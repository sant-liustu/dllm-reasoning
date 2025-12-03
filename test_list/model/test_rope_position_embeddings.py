#!/usr/bin/env python3
"""
Debug RoPE Position Embeddings

检查不同配置下的position embeddings (cos/sin)是否相同
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset
from dllm_reasoning.model.DLLM.modeling_dllm import DLLMForCausalLM
from dllm_reasoning.model.DLLM.configuration_dllm import DLLMConfig


def test_position_embeddings(model, input_ids, position_ids, block_info, prompt_len, seq_len, device):
    """捕获position embeddings (cos/sin)"""

    # Hook rotary_emb
    captured_embeddings = {}

    original_forward = model.model.rotary_emb.forward

    def hooked_forward(x, position_ids):
        cos, sin = original_forward(x, position_ids)
        captured_embeddings['cos'] = cos.detach().clone()
        captured_embeddings['sin'] = sin.detach().clone()
        return cos, sin

    model.model.rotary_emb.forward = hooked_forward

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            block_info=[block_info],
            prompt_len=[prompt_len],
            seq_lens=[seq_len],
            use_cache=False
        )

    # 恢复
    model.model.rotary_emb.forward = original_forward

    return captured_embeddings


def test_config(model, sample, num_blocks_to_keep, device):
    """测试特定配置"""
    input_ids = sample['input_ids'].to(device)
    position_ids = sample['position_ids'].to(device)
    block_info = sample['block_info']
    prompt_len = sample['prompt_len']

    # 截断序列
    current_pos = prompt_len
    blocks_seen = 0
    truncate_pos = None

    for seg_type, seg_idx, seg_len in block_info:
        blocks_seen += 1
        current_pos += seg_len
        if blocks_seen >= num_blocks_to_keep:
            truncate_pos = current_pos
            break

    if truncate_pos is None:
        truncate_pos = input_ids.size(0)

    # 截断
    input_ids_truncated = input_ids[:truncate_pos]
    position_ids_truncated = position_ids[:truncate_pos]

    block_info_truncated = []
    blocks_added = 0
    for seg_type, seg_idx, seg_len in block_info:
        if blocks_added >= num_blocks_to_keep:
            break
        block_info_truncated.append((seg_type, seg_idx, seg_len))
        blocks_added += 1

    # 捕获
    results = test_position_embeddings(
        model=model,
        input_ids=input_ids_truncated,
        position_ids=position_ids_truncated,
        block_info=block_info_truncated,
        prompt_len=prompt_len,
        seq_len=truncate_pos,
        device=device,
    )

    return results, position_ids_truncated


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("RoPE Position Embeddings Debug")
    print("="*80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    config = DLLMConfig.from_pretrained(MODEL_PATH)
    model = DLLMForCausalLM(config)

    from transformers import AutoModelForCausalLM
    pretrained = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.load_state_dict(pretrained.state_dict())
    model = model.to(device).to(torch.bfloat16)
    model.train()

    print("加载数据集...")
    dataset = InterleavedSFTDataset(
        parquet_files=DATA_PATH,
        tokenizer=tokenizer,
        prompt_key="prompt",
        response_key="target",
        block_size=4,
        max_length=6000,
        truncation="right",
    )

    sample = dataset[0]

    print(f"\n样本信息:")
    print(f"  Prompt长度: {sample['prompt_len']}")
    print(f"  M₁位置: {sample['prompt_len']}")
    print()

    # 测试配置
    test_configs = [
        (1, "[P][M₁]"),
        (2, "[P][M₁][R₁]"),
        (len(sample['block_info']), f"完整序列({len(sample['block_info'])}块)"),
    ]

    print("="*80)
    print("捕获Position Embeddings")
    print("="*80)
    print()

    all_results = {}

    for num_blocks, config_name in test_configs:
        print(f"配置: {config_name}")
        embeddings, position_ids = test_config(
            model=model,
            sample=sample,
            num_blocks_to_keep=num_blocks,
            device=device,
        )
        all_results[config_name] = (embeddings, position_ids)

        print(f"  Position IDs shape: {position_ids.shape}")
        print(f"  Position IDs (前10个): {position_ids[:10].tolist()}")
        print(f"  Position IDs (M₁位置277): {position_ids[277].item()}")
        print(f"  Cos shape: {embeddings['cos'].shape}")
        print(f"  Sin shape: {embeddings['sin'].shape}")
        print()

    # 对比
    print("="*80)
    print("对比分析")
    print("="*80)
    print()

    config_names = list(all_results.keys())
    base_config = config_names[0]

    m1_pos = sample['prompt_len']

    for i, compare_config in enumerate(config_names[1:], 1):
        print(f"\n【对比 {i}】基准: {base_config} vs 对比: {compare_config}")
        print("-" * 60)

        base_embeddings, base_position_ids = all_results[base_config]
        compare_embeddings, compare_position_ids = all_results[compare_config]

        # 比较position_ids (前278个位置)
        print("\n1. Position IDs (前278个位置):")
        m1_visible_len = m1_pos + 1
        pos_ids_base = base_position_ids[:m1_visible_len]
        pos_ids_compare = compare_position_ids[:m1_visible_len]

        if torch.equal(pos_ids_base, pos_ids_compare):
            print(f"  ✅ Position IDs 完全相同")
        else:
            diff_count = (pos_ids_base != pos_ids_compare).sum().item()
            print(f"  ❌ Position IDs 有差异: {diff_count}/{m1_visible_len} 个位置不同")

        # 比较cos/sin (前278个位置)
        print("\n2. Cos Embeddings (前278个位置):")
        cos_base = base_embeddings['cos'][:, :m1_visible_len, :]
        cos_compare = compare_embeddings['cos'][:, :m1_visible_len, :]

        diff = torch.norm(cos_base - cos_compare).item()
        max_diff = torch.max(torch.abs(cos_base - cos_compare)).item()

        if diff < 1e-6:
            print(f"  ✅ Cos相同: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")
        else:
            print(f"  ❌ Cos不同: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")

        print("\n3. Sin Embeddings (前278个位置):")
        sin_base = base_embeddings['sin'][:, :m1_visible_len, :]
        sin_compare = compare_embeddings['sin'][:, :m1_visible_len, :]

        diff = torch.norm(sin_base - sin_compare).item()
        max_diff = torch.max(torch.abs(sin_base - sin_compare)).item()

        if diff < 1e-6:
            print(f"  ✅ Sin相同: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")
        else:
            print(f"  ❌ Sin不同: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")

        print()

    print("="*80)
    print("结论")
    print("="*80)
    print()
    print("如果position_ids相同但cos/sin不同,说明:")
    print("  - RoPE实现有问题(不应该发生)")
    print()
    print("如果position_ids和cos/sin都相同,但K/V不同,说明:")
    print("  - 问题可能在apply_rotary_pos_emb函数")
    print("  - 或者是某个其他机制影响了K/V")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
