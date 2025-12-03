#!/usr/bin/env python3
"""
详细对比Layer 0的每个计算步骤
找出精度误差具体在哪里引入的
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


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("Layer 0 详细Debug")
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
    print(f"样本信息: 总长度={sample['input_ids'].shape[0]}, Prompt长度={sample['prompt_len']}")
    print()

    input_ids = sample['input_ids'].to(device)
    position_ids = sample['position_ids'].to(device)
    block_info = sample['block_info']
    prompt_len = sample['prompt_len']

    # 配置1: [P][M₁]
    print("\n" + "="*80)
    print("配置1: [P][M₁] (280 tokens)")
    print("="*80)

    truncate_pos_1 = prompt_len
    for seg_type, seg_idx, seg_len in block_info[:1]:
        truncate_pos_1 += seg_len

    with torch.no_grad():
        outputs_1 = model(
            input_ids=input_ids[:truncate_pos_1].unsqueeze(0),
            position_ids=position_ids[:truncate_pos_1].unsqueeze(0),
            block_info=[block_info[:1]],
            prompt_len=[prompt_len],
            seq_lens=[truncate_pos_1],
            use_cache=False
        )

    # 配置3: 完整序列
    print("\n" + "="*80)
    print("配置3: 完整序列 (5439 tokens)")
    print("="*80)

    with torch.no_grad():
        outputs_3 = model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            block_info=[block_info],
            prompt_len=[prompt_len],
            seq_lens=[input_ids.shape[0]],
            use_cache=False
        )

    print("\n" + "="*80)
    print("完成!")
    print("="*80)


if __name__ == "__main__":
    main()
