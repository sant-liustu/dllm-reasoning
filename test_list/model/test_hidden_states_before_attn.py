#!/usr/bin/env python3
"""
Debug hidden states BEFORE attention

检查传入Layer 0 attention的hidden_states是否相同
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


def capture_hidden_states_before_attn(model, input_ids, position_ids, block_info, prompt_len, seq_len, device):
    """捕获Layer 0输入的hidden states"""
    m1_pos = prompt_len
    captured = {}

    # Hook Layer 0 self_attn的输入
    def hook_attn_input(module, args, kwargs):
        # 从kwargs中获取hidden_states
        hidden_states = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
        if hidden_states is not None:
            captured['hidden_states'] = hidden_states.detach().clone()
            captured['m1_hidden'] = hidden_states[0, m1_pos].detach().clone()
        else:
            print(f"Warning: Could not capture hidden_states. args={len(args)}, kwargs keys={list(kwargs.keys())}")

    # Hook embedding layer输出
    def hook_embedding(module, input, output):
        captured['embeddings'] = output.detach().clone()
        captured['m1_embedding'] = output[0, m1_pos].detach().clone()

    # 注册hooks
    h1 = model.model.layers[0].self_attn.register_forward_pre_hook(hook_attn_input, with_kwargs=True)
    h2 = model.model.embed_tokens.register_forward_hook(hook_embedding)

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

    # 移除hooks
    h1.remove()
    h2.remove()

    return captured


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
    results = capture_hidden_states_before_attn(
        model=model,
        input_ids=input_ids_truncated,
        position_ids=position_ids_truncated,
        block_info=block_info_truncated,
        prompt_len=prompt_len,
        seq_len=truncate_pos,
        device=device,
    )

    return results, input_ids_truncated


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("Hidden States Before Attention Debug")
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
    print("捕获Hidden States")
    print("="*80)
    print()

    all_results = {}

    for num_blocks, config_name in test_configs:
        print(f"配置: {config_name}")
        results, input_ids = test_config(
            model=model,
            sample=sample,
            num_blocks_to_keep=num_blocks,
            device=device,
        )
        all_results[config_name] = (results, input_ids)

        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Input IDs (M₁位置277): {input_ids[277].item()}")
        print(f"  Embeddings shape: {results['embeddings'].shape}")
        print(f"  M₁ embedding: shape {results['m1_embedding'].shape}")
        print(f"  Hidden states (before Layer 0 attn) shape: {results['hidden_states'].shape}")
        print(f"  M₁ hidden (before Layer 0 attn): shape {results['m1_hidden'].shape}")
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

        base_results, base_input_ids = all_results[base_config]
        compare_results, compare_input_ids = all_results[compare_config]

        # 比较input_ids (M₁位置)
        print("\n1. Input ID at M₁:")
        if base_input_ids[m1_pos].item() == compare_input_ids[m1_pos].item():
            print(f"  ✅ Input ID相同: {base_input_ids[m1_pos].item()}")
        else:
            print(f"  ❌ Input ID不同: {base_input_ids[m1_pos].item()} vs {compare_input_ids[m1_pos].item()}")

        # 比较embedding (M₁位置)
        print("\n2. Embedding at M₁:")
        diff = torch.norm(base_results['m1_embedding'] - compare_results['m1_embedding']).item()
        max_diff = torch.max(torch.abs(base_results['m1_embedding'] - compare_results['m1_embedding'])).item()

        if diff < 1e-6:
            print(f"  ✅ Embedding相同: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")
        else:
            print(f"  ❌ Embedding不同: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")

        # 比较hidden states (M₁位置, Layer 0输入)
        print("\n3. Hidden States at M₁ (Layer 0输入):")
        diff = torch.norm(base_results['m1_hidden'] - compare_results['m1_hidden']).item()
        max_diff = torch.max(torch.abs(base_results['m1_hidden'] - compare_results['m1_hidden'])).item()

        if diff < 1e-6:
            print(f"  ✅ Hidden States相同: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")
        else:
            print(f"  ❌ Hidden States不同: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")

        print()

    print("="*80)
    print("结论")
    print("="*80)
    print()
    print("如果Input ID和Embedding相同,但Hidden States(Layer 0输入)不同,说明:")
    print("  - 问题不在Embedding层")
    print("  - 但是等等...Layer 0的输入应该就是embedding!")
    print("  - 需要检查是否有其他preprocess")
    print()
    print("如果所有都相同,但Layer 0的输出不同,说明:")
    print("  - 问题在Layer 0的attention计算本身")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
