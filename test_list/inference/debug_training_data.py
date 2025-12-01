#!/usr/bin/env python3
"""
Debug训练数据格式
"""

import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset


def main():
    MODEL_PATH = "dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
    DATA_PATH = "data/openr1.parquet"

    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")

    print("\n加载数据集...")
    dataset = InterleavedSFTDataset(
        parquet_files=DATA_PATH,
        tokenizer=tokenizer,
        prompt_key="prompt",
        response_key="target",  # 使用target而不是response
        block_size=4,
        max_length=4096,  # 提高max_length
    )

    print(f"数据集大小: {len(dataset)}")

    # 找一个合适长度的样本
    sample = None
    for i in range(100):
        try:
            s = dataset[i]
            if s['input_ids'].shape[0] < 1000:
                sample = s
                print(f"\n使用样本 {i}")
                break
        except ValueError:
            continue

    if sample is None:
        print("未找到合适长度的样本")
        return

    input_ids = sample['input_ids']
    position_ids = sample['position_ids']
    labels = sample['labels']
    block_info = sample['block_info']
    prompt_len = sample['prompt_len']

    print(f"\n样本信息:")
    print(f"  Prompt长度: {prompt_len}")
    print(f"  Input IDs长度: {len(input_ids)}")
    print(f"  Block数量: {len(block_info)}")

    # 打印前2个block的详细信息
    print(f"\n=== 前2个block的详细信息 ===")
    current_pos = prompt_len
    for idx, (seg_type, seg_len) in enumerate(block_info[:4]):
        print(f"\nBlock {idx}: type={seg_type}, start={current_pos}, length={seg_len}")
        block_input_ids = input_ids[current_pos:current_pos+seg_len]
        block_position_ids = position_ids[current_pos:current_pos+seg_len]
        block_labels = labels[current_pos:current_pos+seg_len]

        print(f"  Input IDs: {block_input_ids.tolist()}")
        print(f"  Position IDs: {block_position_ids.tolist()}")
        print(f"  Labels: {block_labels.tolist()}")
        print(f"  Input tokens: {tokenizer.convert_ids_to_tokens(block_input_ids.tolist())}")

        valid_labels = [lbl for lbl in block_labels.tolist() if lbl != -100]
        if len(valid_labels) > 0:
            print(f"  Label tokens: {tokenizer.convert_ids_to_tokens(valid_labels)}")

        current_pos += seg_len


if __name__ == "__main__":
    main()
