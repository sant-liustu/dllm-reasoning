#!/usr/bin/env python3
"""
Debug block_info结构，理解位置对应关系
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import create_interleaved_training_data, create_interleaved_labels

MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
DATA_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

# 加载数据
df = pd.read_parquet(DATA_PATH)
sample = df.iloc[0]

prompt = sample['prompt']
target = sample['target']

messages = list(prompt) if isinstance(prompt, (list, tuple)) else prompt
messages = messages.tolist() if hasattr(messages, 'tolist') else messages

target_list = target.tolist() if hasattr(target, 'tolist') else target
if isinstance(target_list, list) and len(target_list) > 0:
    target_content = target_list[0].get('content', '') if isinstance(target_list[0], dict) else str(target_list[0])
else:
    target_content = str(target_list)

# 去掉<think>
if target_content.strip().startswith('<think>'):
    think_start = target_content.find('<think>')
    target_content = target_content[think_start + 7:].lstrip()

# Tokenize - 对齐训练逻辑
prompt_only_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
full_conversation_str = prompt_only_str + target_content + tokenizer.eos_token

prompt_ids = tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
input_ids = tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

prompt_len = prompt_ids.size(0)
loss_mask = torch.ones(input_ids.size(0), dtype=torch.long, device=device)
loss_mask[:prompt_len] = 0

# 创建interleaved格式
interleaved_ids, position_ids, interleaved_loss_mask, block_info = create_interleaved_training_data(
    input_ids=input_ids,
    loss_mask=loss_mask,
    block_size=4,
    mask_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

print(f"原始序列长度: {input_ids.size(0)}")
print(f"Prompt长度: {prompt_len}")
print(f"Interleaved序列长度: {interleaved_ids.size(0)}")
print(f"Block info数量: {len(block_info)}")

print(f"\n前10个segments的block_info:")
for i, (seg_type, _, seg_len) in enumerate(block_info[:10]):
    print(f"  [{i}] type={seg_type}, length={seg_len}")

print(f"\n检查interleaved_ids的前{prompt_len+20}个位置:")
for i in range(min(prompt_len + 20, interleaved_ids.size(0))):
    token_id = interleaved_ids[i].item()
    token_text = tokenizer.decode([token_id])

    if i == prompt_len - 1:
        print(f"  --- Prompt ends at position {i} ---")

    marker = ""
    if token_id == tokenizer.eos_token_id:
        marker = " <- EOS/MASK token!"

    print(f"  [{i:3d}] ID={token_id:6d} '{token_text}'{marker}")

print(f"\n重要问题：block_info的位置是从0开始还是从prompt_len开始？")
print(f"让我们验证：")

# 根据block_info重建位置映射
print(f"\n如果从position 0开始：")
current_pos = 0
for i, (seg_type, _, seg_len) in enumerate(block_info[:5]):
    print(f"  Segment {i}: [{current_pos}:{current_pos+seg_len}] type={seg_type}")
    if current_pos < interleaved_ids.size(0):
        first_token = interleaved_ids[current_pos].item()
        print(f"    First token: ID={first_token}, is_EOS={first_token == tokenizer.eos_token_id}")
    current_pos += seg_len

print(f"\n如果从position {prompt_len}开始：")
current_pos = prompt_len
for i, (seg_type, _, seg_len) in enumerate(block_info[:5]):
    print(f"  Segment {i}: [{current_pos}:{current_pos+seg_len}] type={seg_type}")
    if current_pos < interleaved_ids.size(0):
        first_token = interleaved_ids[current_pos].item()
        print(f"    First token: ID={first_token}, is_EOS={first_token == tokenizer.eos_token_id}")
    current_pos += seg_len
