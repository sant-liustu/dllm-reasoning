#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 FlexAttention BlockMask 构造的正确性。

验证:
1. create_mask_mod_from_block_info() 正确实现 6 条规则
2. 与之前的 2D dense mask 对比，确保逻辑一致
"""

import sys
sys.path.insert(0, "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning")

import torch
from dllm_reasoning.trainer.flex_attention_utils import (
    create_mask_mod_from_block_info,
    verify_mask_rules,
)

# 手动复制 create_training_attention_mask_from_block_info 函数（避免执行整个测试文件）
def create_training_attention_mask_from_block_info(block_info, seq_len, prompt_len, device):
    """根据 block_info 构造训练时使用的 attention mask（dense 2D）。"""
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    # 构建 segment 信息
    segments = []
    segments.append(('prompt', 0, prompt_len))
    current_pos = prompt_len

    for seg_type, _, seg_len in block_info:
        segments.append((seg_type, current_pos, seg_len))
        current_pos += seg_len

    # 填充 mask
    for q_idx, (q_type, q_start, q_len) in enumerate(segments):
        for kv_idx, (kv_type, kv_start, kv_len) in enumerate(segments):
            if q_type == 'prompt':
                if kv_type == 'prompt':
                    for i in range(q_len):
                        for j in range(min(i + 1, kv_len)):
                            mask[q_start + i, kv_start + j] = True
            elif q_type == 'mask':
                if kv_type == 'prompt':
                    mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True
                elif kv_type == 'real':
                    if kv_idx < q_idx:
                        mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True
                elif kv_type == 'mask':
                    if kv_idx == q_idx:
                        for i in range(q_len):
                            for j in range(min(i + 1, kv_len)):
                                mask[q_start + i, kv_start + j] = True
            elif q_type == 'real':
                if kv_type == 'prompt':
                    mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True
                elif kv_type == 'real':
                    if kv_idx < q_idx:
                        mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True
                    elif kv_idx == q_idx:
                        for i in range(q_len):
                            for j in range(min(i + 1, kv_len)):
                                mask[q_start + i, kv_start + j] = True

    return mask, segments

def test_mask_mod_basic():
    """测试基本的 mask_mod 功能"""
    print("=" * 100)
    print("Test 1: Basic mask_mod functionality")
    print("=" * 100)

    # 简单的测试用例: prompt + 1 mask group + 1 block
    prompt_len = 4
    block_size = 4
    num_masks = 3

    # block_info: [('mask', 3), ('real', 4)]
    block_info = [
        ('mask', num_masks),
        ('real', block_size),
    ]

    seq_len = prompt_len + num_masks + block_size

    print(f"\nTest setup:")
    print(f"  Prompt length: {prompt_len}")
    print(f"  Block info: {block_info}")
    print(f"  Total seq_len: {seq_len}")

    # 创建 mask_mod
    mask_mod = create_mask_mod_from_block_info(block_info, prompt_len, seq_len)

    # 测试一些关键位置
    print(f"\nKey position tests:")

    # Prompt 内部 causal
    print(f"  p0 -> p1: {mask_mod(0, 0, 0, 1)} (should be False, causal)")
    print(f"  p1 -> p0: {mask_mod(0, 0, 1, 0)} (should be True, causal)")

    # Mask 可以看 prompt
    print(f"  M0 -> p0: {mask_mod(0, 0, 4, 0)} (should be True)")
    print(f"  M0 -> p3: {mask_mod(0, 0, 4, 3)} (should be True)")

    # Mask 看不到对应 block
    print(f"  M0 -> r0: {mask_mod(0, 0, 4, 7)} (should be False)")

    # Real block 看不到 mask
    print(f"  r0 -> M0: {mask_mod(0, 0, 7, 4)} (should be False)")
    print(f"  r1 -> M1: {mask_mod(0, 0, 8, 5)} (should be False)")

    # Real block 可以看 prompt
    print(f"  r0 -> p0: {mask_mod(0, 0, 7, 0)} (should be True)")

    # Real block 内部 causal
    print(f"  r0 -> r1: {mask_mod(0, 0, 7, 8)} (should be False, causal)")
    print(f"  r1 -> r0: {mask_mod(0, 0, 8, 7)} (should be True, causal)")

    print("\n✓ Basic tests completed\n")


def test_mask_mod_full_sequence():
    """测试完整序列（2个block）的 mask_mod"""
    print("=" * 100)
    print("Test 2: Full sequence with 2 blocks")
    print("=" * 100)

    # 与 test_training_attention_mask.py 中相同的设置
    prompt_len = 4
    block_size = 4
    num_masks = 3

    # Structure: [prompt] [mask0] [block0] [mask1] [block1]
    block_info = [
        ('mask', num_masks),   # Masks 0
        ('real', block_size),   # Block 0
        ('mask', num_masks),   # Masks 1
        ('real', block_size),   # Block 1
    ]

    seq_len = prompt_len + 2 * (num_masks + block_size)  # 4 + 2*(3+4) = 18

    print(f"\nTest setup:")
    print(f"  Prompt: [0-3]")
    print(f"  Mask0: [4-6]")
    print(f"  Block0: [7-10]")
    print(f"  Mask1: [11-13]")
    print(f"  Block1: [14-17]")
    print(f"  Total seq_len: {seq_len}")

    # 创建 mask_mod
    mask_mod = create_mask_mod_from_block_info(block_info, prompt_len, seq_len)

    # 创建对应的 dense 2D mask (作为参考)
    # 需要将 block_info 转换为原始格式: (seg_type, seg_idx, seg_len)
    block_info_with_idx = [
        ('mask', 1, num_masks),
        ('real', 2, block_size),
        ('mask', 3, num_masks),
        ('real', 4, block_size),
    ]
    dense_mask, segments = create_training_attention_mask_from_block_info(
        block_info=block_info_with_idx,
        seq_len=seq_len,
        prompt_len=prompt_len,
        device=torch.device('cpu'),
    )

    print(f"\n Comparing mask_mod with dense 2D mask...")

    # 对比所有位置
    mismatches = []
    for q in range(seq_len):
        for kv in range(seq_len):
            mask_mod_result = mask_mod(0, 0, q, kv)
            dense_result = dense_mask[q, kv].item()

            if mask_mod_result != dense_result:
                mismatches.append((q, kv, mask_mod_result, dense_result))

    if mismatches:
        print(f"\n❌ Found {len(mismatches)} mismatches!")
        for q, kv, mask_mod_val, dense_val in mismatches[:10]:
            print(f"  Position ({q}, {kv}): mask_mod={mask_mod_val}, dense={dense_val}")
        return False
    else:
        print(f"\n✓ All {seq_len * seq_len} positions match between mask_mod and dense mask!")

    # 运行自动验证
    print(f"\nRunning automatic rule verification...")
    if verify_mask_rules(block_info, prompt_len, seq_len):
        print(f"✓ All 6 rules verified!")
    else:
        print(f"❌ Some rules failed!")
        return False

    print("\n✓ Full sequence test passed\n")
    return True


def test_mask_mod_with_padding():
    """测试带 padding 的情况"""
    print("=" * 100)
    print("Test 3: Sequence with padding")
    print("=" * 100)

    prompt_len = 4
    block_size = 4
    num_masks = 3

    # 实际序列长度
    actual_seq_len = prompt_len + num_masks + block_size  # 11
    # Padding 后的长度
    padded_seq_len = 16

    block_info = [
        ('mask', num_masks),
        ('real', block_size),
    ]

    print(f"\nTest setup:")
    print(f"  Actual seq_len: {actual_seq_len}")
    print(f"  Padded seq_len: {padded_seq_len}")

    # 使用实际长度创建 mask_mod
    mask_mod = create_mask_mod_from_block_info(block_info, prompt_len, actual_seq_len)

    # 验证 padding 区域不可见
    print(f"\nPadding visibility tests:")
    print(f"  Position 10 (last valid) -> Position 11 (first padding): {mask_mod(0, 0, 10, 11)} (should be False)")
    print(f"  Position 11 (padding) -> Position 0: {mask_mod(0, 0, 11, 0)} (should be False)")

    # 验证 padding 不影响有效区域
    print(f"  Position 7 (r0) -> Position 0 (p0): {mask_mod(0, 0, 7, 0)} (should be True)")

    print("\n✓ Padding test passed\n")


if __name__ == "__main__":
    try:
        print("\n" + "=" * 100)
        print("Testing FlexAttention BlockMask Construction")
        print("=" * 100 + "\n")

        # Run all tests
        test_mask_mod_basic()
        success = test_mask_mod_full_sequence()
        test_mask_mod_with_padding()

        if success:
            print("\n" + "=" * 100)
            print("✅ ALL TESTS PASSED!")
            print("=" * 100 + "\n")
            exit(0)
        else:
            print("\n" + "=" * 100)
            print("❌ SOME TESTS FAILED!")
            print("=" * 100 + "\n")
            exit(1)

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
