#!/usr/bin/env python3
"""
使用模型内部的print来debug Layer 0
验证3种配置下Layer 0的hidden_states, K, V是否相同
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


def test_config(model, sample, num_blocks_to_keep, config_name, device):
    """测试特定配置"""
    print("\n" + "="*80)
    print(f"配置: {config_name} (保留前 {num_blocks_to_keep} 个blocks)")
    print("="*80)

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

    print(f"截断后序列长度: {truncate_pos}")
    print(f"Block info: {block_info_truncated[:3]}...")  # 只打印前3个

    # Forward pass - 模型内部的print会自动输出
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_truncated.unsqueeze(0),
            position_ids=position_ids_truncated.unsqueeze(0),
            block_info=[block_info_truncated],
            prompt_len=[prompt_len],
            seq_lens=[truncate_pos],
            use_cache=False
        )


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("Layer 0 Debug with Model Internal Print")
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

    # 重要：使用训练模式
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
    print(f"  总序列长度: {sample['input_ids'].shape[0]}")
    print(f"  Prompt长度: {sample['prompt_len']}")
    print(f"  总Block数: {len(sample['block_info'])}")
    print()

    # 测试3个配置
    test_configs = [
        (1, "[P][M₁]"),
        (2, "[P][M₁][R₁]"),
        (len(sample['block_info']), f"完整序列({len(sample['block_info'])}块)"),
    ]

    for num_blocks, config_name in test_configs:
        test_config(
            model=model,
            sample=sample,
            num_blocks_to_keep=num_blocks,
            config_name=config_name,
            device=device,
        )
        print("\n")  # 分隔不同配置的输出

    print("="*80)
    print("测试完成！")
    print("="*80)
    print()
    print("分析说明:")
    print("1. 对比3种配置下的 hidden_states 是否相同")
    print("2. 对比 Before RoPE 的 key_states 是否相同")
    print("3. 对比 Position embeddings (cos/sin) 是否相同")
    print("4. 对比 After RoPE 的 key_states 是否相同")
    print()
    print("如果 Before RoPE 相同但 After RoPE 不同,说明是RoPE的问题")
    print("如果 Position embeddings 不同,说明是动态RoPE导致的")


if __name__ == "__main__":
    main()
