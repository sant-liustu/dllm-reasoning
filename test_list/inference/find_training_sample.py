#!/usr/bin/env python3
"""查找训练时使用的样本"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer
from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset

tokenizer = AutoTokenizer.from_pretrained(
    'dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface',
    trust_remote_code=True
)

dataset = InterleavedSFTDataset(
    parquet_files='data/openr1.parquet',
    tokenizer=tokenizer,
    prompt_key='prompt',
    response_key='target',
    block_size=4,
    max_length=6000,
    truncation='right',
)

print('查找prompt长度=244的样本...')
for i in range(min(100, len(dataset))):
    try:
        sample = dataset[i]
        if sample['prompt_len'] == 244:
            print(f'\n找到样本 {i}:')
            print(f'  prompt_len={sample["prompt_len"]}')
            print(f'  total_len={sample["input_ids"].shape[0]}')

            # 检查第一个Mask block（应该在244-246）
            mask_start = 244
            print(f'\n  位置 {mask_start}-{mask_start+2} (应该是Mask):')
            for pos in range(mask_start, mask_start+3):
                token_id = sample['input_ids'][pos].item()
                label_id = sample['labels'][pos].item()
                token_text = tokenizer.decode([token_id])
                print(f'    pos={pos}: input={token_id:6d} "{token_text}", label={label_id:6d}')
            break
    except Exception as e:
        print(f'样本{i}加载失败: {e}')
        continue
else:
    print('\n未找到prompt_len=244的样本')
