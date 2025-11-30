#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证标签是否在所有位置都计算了 loss
"""

import sys
sys.path.insert(0, "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning")

import torch
from dllm_reasoning.trainer.interleaved_sft_dataset import (
    create_interleaved_training_data,
    create_interleaved_labels,
)

# 创建测试数据
block_size = 4
prompt_len = 4
response_len = 8

# 原始序列: [0,1,2,3] (prompt) + [100,101,102,103,104,105,106,107] (response)
original_input_ids = torch.cat([
    torch.arange(prompt_len),
    torch.arange(100, 100 + response_len)
])

loss_mask = torch.cat([
    torch.zeros(prompt_len, dtype=torch.long),
    torch.ones(response_len, dtype=torch.long),
])

mask_token_id = 999
pad_token_id = 0

# 创建交错数据
interleaved_ids, interleaved_pos, interleaved_loss, block_info = create_interleaved_training_data(
    input_ids=original_input_ids,
    loss_mask=loss_mask,
    block_size=block_size,
    mask_token_id=mask_token_id,
    pad_token_id=pad_token_id,
)

# 创建标签
labels = create_interleaved_labels(
    original_input_ids=original_input_ids,
    interleaved_input_ids=interleaved_ids,
    interleaved_loss_mask=interleaved_loss,
    block_size=block_size,
    prompt_len=prompt_len,
    pad_token_id=pad_token_id,
    ignore_index=-100,
)

print("=" * 100)
print("标签分析")
print("=" * 100)

print("\n原始序列:")
print(f"  {original_input_ids.tolist()}")

print("\n交错序列:")
print(f"  IDs:  {interleaved_ids.tolist()}")
print(f"  Pos:  {interleaved_pos.tolist()}")

print("\n标签:")
print(f"  {labels.tolist()}")

print("\n详细标签分析:")
print(f"{'位置':<6} {'输入':<8} {'标签':<8} {'说明':<30}")
print("-" * 60)

# 手动标注期望
token_names = []
# Prompt
for i in range(prompt_len):
    token_names.append(f"p{i}")

# 根据 block_info 构建名称
for seg_type, _, seg_len in block_info:
    if seg_type == 'mask':
        mask_idx = sum(1 for s in block_info[:block_info.index((seg_type, _, seg_len))] if s[0] == 'mask')
        for i in range(seg_len):
            token_names.append(f"M{mask_idx}_{i}")
    elif seg_type == 'real':
        block_idx = sum(1 for s in block_info[:block_info.index((seg_type, _, seg_len))] if s[0] == 'real')
        for i in range(seg_len):
            token_names.append(f"r{block_idx * block_size + i}")

for i in range(len(interleaved_ids)):
    input_val = interleaved_ids[i].item()
    label_val = labels[i].item()

    name = token_names[i] if i < len(token_names) else f"?{i}"

    if label_val == -100:
        label_str = "ignore"
        desc = "不计算 loss"
    else:
        label_str = str(label_val)
        desc = f"预测 {label_val}"

    print(f"{i:<6} {name:<8} {input_val:<8} {label_str:<8} {desc:<30}")

print("\n" + "=" * 100)
print("关键位置验证:")
print("=" * 100)

# 找到关键位置
# Prompt 末尾 p3
p3_idx = 3
print(f"\n1. p3 (位置 {p3_idx}):")
print(f"   输入: {interleaved_ids[p3_idx].item()}")
print(f"   标签: {labels[p3_idx].item()}")
print(f"   预期: 应该是 -100 (ignore),因为 p3 不在 response 范围内")
print(f"   ✓" if labels[p3_idx].item() == -100 else f"   ✗")

# Block 0 的 real tokens
print(f"\n2. Block 0 的 real tokens (r0-r3):")
# 根据结构: [p0,p1,p2,p3][M0,M1,M2][r0,r1,r2,r3]
r0_idx = prompt_len + 3  # 跳过 3 个 mask
for i in range(4):
    pos = r0_idx + i
    print(f"   r{i} (位置 {pos}): 输入={interleaved_ids[pos].item()}, 标签={labels[pos].item()}, 预期=10{i+1 if i < 3 else 4}")
    if i < 3:
        expected = 100 + i + 1
    else:
        expected = 104
    print(f"   {'✓' if labels[pos].item() == expected else '✗'}")

# Block 1 的 real tokens
print(f"\n3. Block 1 的 real tokens (r4-r7):")
# 根据结构: ...[M3,M4,M5][r4,r5,r6,r7]
r4_idx = prompt_len + 3 + 4 + 3  # 跳过 M0, Block0, M1
for i in range(4):
    pos = r4_idx + i
    print(f"   r{4+i} (位置 {pos}): 输入={interleaved_ids[pos].item()}, 标签={labels[pos].item()}", end="")
    if i < 3:
        expected = 100 + 4 + i + 1
        print(f", 预期={expected}")
        print(f"   {'✓' if labels[pos].item() == expected else '✗'}")
    else:
        print(f", 预期=ignore (最后一个 token)")
        print(f"   {'✓' if labels[pos].item() == -100 else '✗'}")

print("\n" + "=" * 100)
print("总结:")
print("=" * 100)

# 统计有多少位置计算 loss
num_loss_positions = (labels != -100).sum().item()
total_positions = len(labels)

print(f"\n计算 loss 的位置数: {num_loss_positions} / {total_positions}")
print(f"  - Prompt: {prompt_len} 个位置不计算 loss")
print(f"  - Response: {response_len} 个 token")
print(f"  - Masks: 2 组 × 3 个 = 6 个")
print(f"  - 理论应计算 loss: response_len - 1 (最后一个) + masks = {response_len - 1 + 6} = {response_len + 5}")
print(f"  - 实际计算 loss: {num_loss_positions}")

if num_loss_positions == response_len + 5:
    print("\n✓ 标签设置正确！所有可预测位置都计算了 loss")
else:
    print(f"\n✗ 标签设置可能有问题，预期 {response_len + 5}，实际 {num_loss_positions}")
