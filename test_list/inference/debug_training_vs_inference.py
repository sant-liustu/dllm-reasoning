#!/usr/bin/env python3
"""
对比训练格式和推理格式,找出差异
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import create_interleaved_training_data

MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
DATA_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
df = pd.read_parquet(DATA_PATH)
sample = df.iloc[0]

device = torch.device("cpu")

# ========== 训练时的格式 ==========
print("="*100)
print("训练时的格式")
print("="*100)

prompt = sample['prompt']
target = sample['target']

# 处理成和训练一样
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

# 完整对话
full_messages = messages + [{"role": "assistant", "content": target_content}]

# Tokenize
input_ids = tokenizer.apply_chat_template(
    full_messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)[0].to(device)

prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)[0].to(device)

prompt_len = prompt_ids.size(0)
loss_mask = torch.ones(input_ids.size(0), dtype=torch.long, device=device)

# 创建interleaved格式
interleaved_ids, position_ids, interleaved_loss_mask, block_info = create_interleaved_training_data(
    input_ids=input_ids,
    prompt_len=prompt_len,
    loss_mask=loss_mask,
    block_size=4,
    mask_token_id=tokenizer.eos_token_id,
    ignore_index=-100
)

print(f"原始序列长度: {input_ids.size(0)}")
print(f"Prompt长度: {prompt_len}")
print(f"Response长度: {input_ids.size(0) - prompt_len}")
print(f"Interleaved序列长度: {interleaved_ids.size(0)}")
print(f"\nBlock info (前3个): {block_info[:3]}")

# 第一个block
print(f"\n第一个Block详情:")
print(f"Prompt最后3个tokens (ID): {interleaved_ids[prompt_len-3:prompt_len].tolist()}")
print(f"Prompt最后3个tokens (text): {tokenizer.decode(interleaved_ids[prompt_len-3:prompt_len])}")

# 找到第一个mask block和real block
mask_start = prompt_len
mask_end = mask_start + 3
real_start = mask_end
real_end = real_start + 4

print(f"\nMask×3:")
print(f"  位置: [{mask_start}:{mask_end}]")
print(f"  IDs: {interleaved_ids[mask_start:mask_end].tolist()}")
print(f"  Position IDs: {position_ids[mask_start:mask_end].tolist()}")

print(f"\nReal×4:")
print(f"  位置: [{real_start}:{real_end}]")
print(f"  IDs: {interleaved_ids[real_start:real_end].tolist()}")
print(f"  Text: '{tokenizer.decode(interleaved_ids[real_start:real_end])}'")
print(f"  Position IDs: {position_ids[real_start:real_end].tolist()}")

# ========== 推理时的格式 ==========
print(f"\n{'='*100}")
print("推理时的格式")
print("="*100)

# 推理输入
inference_ids = prompt_ids.clone()
mask_tokens = torch.full((3,), tokenizer.eos_token_id, dtype=torch.long, device=device)
inference_with_mask = torch.cat([inference_ids, mask_tokens])

inference_position_ids = torch.arange(0, inference_with_mask.size(0), device=device)

print(f"推理输入长度: {inference_with_mask.size(0)}")
print(f"  = Prompt({prompt_len}) + Mask(3)")
print(f"\nPrompt最后3个tokens (ID): {inference_with_mask[prompt_len-3:prompt_len].tolist()}")
print(f"Prompt最后3个tokens (text): {tokenizer.decode(inference_with_mask[prompt_len-3:prompt_len])}")
print(f"\nMask×3:")
print(f"  位置: [{prompt_len}:{prompt_len+3}]")
print(f"  IDs: {inference_with_mask[prompt_len:prompt_len+3].tolist()}")
print(f"  Position IDs: {inference_position_ids[prompt_len:prompt_len+3].tolist()}")

# ========== 对比 ==========
print(f"\n{'='*100}")
print("关键差异")
print("="*100)

print("\n1. 序列内容:")
print(f"   训练: [Prompt({prompt_len})] + [Mask×3] + [Real×4] + ... = 长度 {interleaved_ids.size(0)}")
print(f"   推理: [Prompt({prompt_len})] + [Mask×3] = 长度 {inference_with_mask.size(0)}")
print(f"   差异: 推理时缺少Real tokens!")

print("\n2. Position IDs:")
print(f"   训练Mask×3 position_ids: {position_ids[mask_start:mask_end].tolist()}")
print(f"   推理Mask×3 position_ids: {inference_position_ids[prompt_len:prompt_len+3].tolist()}")
print(f"   是否一致: {torch.equal(position_ids[mask_start:mask_end], inference_position_ids[prompt_len:prompt_len+3])}")

print("\n3. 训练时Real tokens的存在:")
print(f"   训练时,虽然Mask不能通过attention看到Real,但Real tokens的embeddings仍在序列中")
print(f"   推理时,完全没有Real tokens")
print(f"   这可能导致hidden states的分布不同!")

print("\n4. 期望的预测:")
print(f"   训练时labels设置:")
print(f"     - Prompt[{prompt_len-1}] 应该预测 Real[0] = {interleaved_ids[real_start].item()}")
print(f"     - Mask[0] 应该预测 Real[1] = {interleaved_ids[real_start+1].item()}")
print(f"     - Mask[1] 应该预测 Real[2] = {interleaved_ids[real_start+2].item()}")
print(f"     - Mask[2] 应该预测 Real[3] = {interleaved_ids[real_start+3].item()}")
print(f"   推理时应该得到相同的预测,但实际预测的是BOS tokens!")
