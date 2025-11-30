#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试交错训练数据构造的正确性。

验证:
1. Position IDs 设计是否正确（mask 位置的 position 感知）
2. Attention mask 是否正确（可见性规则）
3. Labels 是否正确（AR 偏移）
"""

import sys
sys.path.insert(0, "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning")

import torch
from dllm_reasoning.trainer.interleaved_sft_dataset import (
    create_interleaved_training_data,
    create_interleaved_labels,
    create_interleaved_attention_mask,
)


def visualize_attention_mask(mask: torch.Tensor, tokens: list, title: str = "Attention Mask"):
    """可视化 attention mask"""
    print(f"\n{title}")
    print("-" * 60)

    # Header
    header = "     " + " ".join([f"{t:>4}" for t in range(len(tokens))])
    print(header)

    for i, token in enumerate(tokens):
        row = f"{i:>3}: "
        for j in range(mask.shape[1]):
            if mask[i, j]:
                row += "  1  "
            else:
                row += "  .  "
        row += f"  <- {token}"
        print(row)


def test_basic_interleaving():
    """测试基本的交错数据构造"""
    print("\n" + "=" * 60)
    print("Test 1: Basic Interleaving (block_size=4)")
    print("=" * 60)

    # 模拟数据: prompt 4 tokens, response 8 tokens
    # Original: [p0, p1, p2, p3, r0, r1, r2, r3, r4, r5, r6, r7]
    block_size = 4
    prompt_len = 4
    response_len = 8

    input_ids = torch.arange(prompt_len + response_len)  # [0, 1, 2, ..., 11]
    loss_mask = torch.cat([
        torch.zeros(prompt_len, dtype=torch.long),
        torch.ones(response_len, dtype=torch.long),
    ])

    mask_token_id = 999
    pad_token_id = 0

    print(f"\nOriginal input_ids: {input_ids.tolist()}")
    print(f"Original loss_mask: {loss_mask.tolist()}")
    print(f"Prompt length: {prompt_len}")
    print(f"Response length: {response_len}")
    print(f"Block size: {block_size}")

    # 创建交错数据
    interleaved_ids, interleaved_pos, interleaved_loss, block_info = create_interleaved_training_data(
        input_ids=input_ids,
        loss_mask=loss_mask,
        block_size=block_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
    )

    print(f"\n--- After Interleaving ---")
    print(f"Interleaved input_ids: {interleaved_ids.tolist()}")
    print(f"Interleaved position_ids: {interleaved_pos.tolist()}")
    print(f"Interleaved loss_mask: {interleaved_loss.tolist()}")
    print(f"Block info: {block_info}")

    # 验证结构
    # 期望: [p0,p1,p2,p3] [M,M,M] [r0,r1,r2,r3] [M,M,M] [r4,r5,r6,r7]
    # 位置: [0, 1, 2, 3]  [4,5,6] [4, 5, 6, 7]  [8,9,10] [8, 9,10,11]
    # 说明: mask和对应block的position重叠
    expected_ids = [0, 1, 2, 3, 999, 999, 999, 4, 5, 6, 7, 999, 999, 999, 8, 9, 10, 11]
    expected_pos = [0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10, 11]

    print(f"\nExpected input_ids: {expected_ids}")
    print(f"Expected position_ids: {expected_pos}")

    # 检查
    ids_match = interleaved_ids.tolist() == expected_ids
    pos_match = interleaved_pos.tolist() == expected_pos

    print(f"\n✓ Input IDs match: {ids_match}")
    print(f"✓ Position IDs match: {pos_match}")

    return ids_match and pos_match


def test_labels():
    """测试 labels 的 AR 偏移"""
    print("\n" + "=" * 60)
    print("Test 2: Labels with AR Shift")
    print("=" * 60)

    block_size = 4
    prompt_len = 4
    response_len = 8

    # 使用有意义的 token IDs 便于追踪
    # Prompt: [100, 101, 102, 103]
    # Response: [200, 201, 202, 203, 204, 205, 206, 207]
    prompt_ids = torch.tensor([100, 101, 102, 103])
    response_ids = torch.tensor([200, 201, 202, 203, 204, 205, 206, 207])
    input_ids = torch.cat([prompt_ids, response_ids])

    loss_mask = torch.cat([
        torch.zeros(prompt_len, dtype=torch.long),
        torch.ones(response_len, dtype=torch.long),
    ])

    mask_token_id = 999
    pad_token_id = 0

    print(f"\nOriginal sequence:")
    print(f"  Prompt:   {prompt_ids.tolist()}")
    print(f"  Response: {response_ids.tolist()}")

    # 创建交错数据
    interleaved_ids, interleaved_pos, interleaved_loss, _ = create_interleaved_training_data(
        input_ids=input_ids,
        loss_mask=loss_mask,
        block_size=block_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
    )

    # 创建 labels
    labels = create_interleaved_labels(
        original_input_ids=input_ids,
        interleaved_input_ids=interleaved_ids,
        interleaved_loss_mask=interleaved_loss,
        block_size=block_size,
        prompt_len=prompt_len,
        pad_token_id=pad_token_id,
    )

    print(f"\nInterleaved structure:")
    print(f"  Input IDs:    {interleaved_ids.tolist()}")
    print(f"  Position IDs: {interleaved_pos.tolist()}")
    print(f"  Labels:       {labels.tolist()}")

    # 解释每个位置的预测目标
    print(f"\nPrediction targets (position -> predicts):")
    for i, (inp, pos, lab) in enumerate(zip(interleaved_ids.tolist(), interleaved_pos.tolist(), labels.tolist())):
        token_type = "M" if inp == mask_token_id else str(inp)
        target = "ignore" if lab == -100 else str(lab)
        print(f"  [{i:2d}] input={token_type:>4}, pos={pos:2d} -> label={target}")

    # 验证关键点
    # Block 0 最后一个 token (r3=203) 应该预测 r4=204
    # Mask 0 (位置8) 应该预测 r5=205
    # Mask 1 (位置9) 应该预测 r6=206
    # Mask 2 (位置10) 应该预测 r7=207

    print("\n--- Validation ---")

    # 找到 block 0 最后一个 real token 的位置
    block0_last_idx = prompt_len + block_size - 1  # index 7
    print(f"Block 0 last token (idx={block0_last_idx}): input={interleaved_ids[block0_last_idx]}, label={labels[block0_last_idx]}")
    assert labels[block0_last_idx] == 204, f"Expected 204, got {labels[block0_last_idx]}"
    print(f"  ✓ Correctly predicts 204 (next block's first token)")

    # Mask positions
    mask_start = prompt_len + block_size  # index 8
    for i in range(3):
        idx = mask_start + i
        expected_label = 205 + i
        print(f"Mask {i} (idx={idx}): label={labels[idx]}")
        assert labels[idx] == expected_label, f"Expected {expected_label}, got {labels[idx]}"
        print(f"  ✓ Correctly predicts {expected_label}")

    print("\n✓ All label validations passed!")
    return True


def test_attention_mask():
    """测试 attention mask 的可见性规则"""
    print("\n" + "=" * 60)
    print("Test 3: Attention Mask Visibility Rules")
    print("=" * 60)

    block_size = 4
    prompt_len = 4
    num_blocks = 2
    num_masks_per_block = block_size - 1

    # 计算序列长度
    # 新结构: [prompt] [masks0] [block0] [masks1] [block1]
    seq_len = prompt_len + num_masks_per_block + block_size + num_masks_per_block + block_size

    print(f"\nSequence structure:")
    print(f"  Prompt: positions 0-{prompt_len-1}")

    masks0_start = prompt_len
    masks0_end = masks0_start + num_masks_per_block
    print(f"  Masks 0: positions {masks0_start}-{masks0_end-1}")

    block0_start = masks0_end
    block0_end = block0_start + block_size
    print(f"  Block 0: positions {block0_start}-{block0_end-1}")

    masks1_start = block0_end
    masks1_end = masks1_start + num_masks_per_block
    print(f"  Masks 1: positions {masks1_start}-{masks1_end-1}")

    block1_start = masks1_end
    block1_end = block1_start + block_size
    print(f"  Block 1: positions {block1_start}-{block1_end-1}")

    print(f"  Total length: {seq_len}")

    # 创建 attention mask
    mask = create_interleaved_attention_mask(
        seq_len=seq_len,
        prompt_len=prompt_len,
        block_size=block_size,
        num_blocks=num_blocks,
        device=torch.device("cpu"),
    )

    # 创建 token 标签用于可视化
    # 新结构: [p0,p1,p2,p3] [M0,M1,M2] [r0,r1,r2,r3] [M3,M4,M5] [r4,r5,r6,r7]
    tokens = []
    for i in range(prompt_len):
        tokens.append(f"p{i}")
    for i in range(num_masks_per_block):
        tokens.append(f"M0_{i}")
    for i in range(block_size):
        tokens.append(f"r{i}")
    for i in range(num_masks_per_block):
        tokens.append(f"M1_{i}")
    for i in range(block_size):
        tokens.append(f"r{block_size + i}")

    visualize_attention_mask(mask, tokens, "Attention Mask")

    # 验证关键规则
    print("\n--- Rule Validation ---")

    # Rule 1: Real tokens cannot see mask tokens
    # Block 0 不能看到 Masks 0 (它自己前面的 masks)
    block0_sees_masks0 = mask[block0_start:block0_end, masks0_start:masks0_end].any()
    print(f"Rule 1: Block 0 cannot see Masks 0 (before it): {not block0_sees_masks0}")
    assert not block0_sees_masks0, "Block 0 should not see Masks 0!"

    # Block 1 不能看到 Masks 0 和 Masks 1
    block1_sees_masks0 = mask[block1_start:block1_end, masks0_start:masks0_end].any()
    block1_sees_masks1 = mask[block1_start:block1_end, masks1_start:masks1_end].any()
    print(f"Rule 1: Block 1 cannot see Masks 0: {not block1_sees_masks0}")
    print(f"Rule 1: Block 1 cannot see Masks 1: {not block1_sees_masks1}")
    assert not block1_sees_masks0, "Block 1 should not see Masks 0!"
    assert not block1_sees_masks1, "Block 1 should not see Masks 1!"

    # Rule 2: Mask tokens can see prompt and previous real blocks
    # Masks 0 should see Prompt only (no previous blocks)
    masks0_sees_prompt = mask[masks0_start:masks0_end, :prompt_len].all()
    print(f"Rule 2: Masks 0 sees Prompt: {masks0_sees_prompt}")
    assert masks0_sees_prompt, "Masks 0 should see Prompt!"

    # Masks 1 should see Prompt and Block 0
    masks1_sees_prompt = mask[masks1_start:masks1_end, :prompt_len].all()
    masks1_sees_block0 = mask[masks1_start:masks1_end, block0_start:block0_end].all()
    print(f"Rule 2: Masks 1 sees Prompt: {masks1_sees_prompt}")
    print(f"Rule 2: Masks 1 sees Block 0: {masks1_sees_block0}")
    assert masks1_sees_prompt, "Masks 1 should see Prompt!"
    assert masks1_sees_block0, "Masks 1 should see Block 0!"

    # Rule 3: Mask tokens are causal within their group
    # M0_0 cannot see M0_1; M0_1 can see M0_0
    m0_0_sees_m0_1 = mask[masks0_start, masks0_start + 1]
    m0_1_sees_m0_0 = mask[masks0_start + 1, masks0_start]
    print(f"Rule 3: M0_0 sees M0_1: {m0_0_sees_m0_1} (should be False)")
    print(f"Rule 3: M0_1 sees M0_0: {m0_1_sees_m0_0} (should be True)")
    assert not m0_0_sees_m0_1, "M0_0 should not see M0_1!"
    assert m0_1_sees_m0_0, "M0_1 should see M0_0!"

    # Rule 4: Real tokens are causal within their block
    # r0 cannot see r1, r1 can see r0
    r0_idx = block0_start
    r1_idx = block0_start + 1
    r0_sees_r1 = mask[r0_idx, r1_idx]
    r1_sees_r0 = mask[r1_idx, r0_idx]
    print(f"Rule 4: r0 sees r1: {r0_sees_r1} (should be False)")
    print(f"Rule 4: r1 sees r0: {r1_sees_r0} (should be True)")
    assert not r0_sees_r1, "r0 should not see r1!"
    assert r1_sees_r0, "r1 should see r0!"

    # Rule 5: Block 1 can see Block 0 (all of it) and Prompt
    block1_sees_block0 = mask[block1_start:block1_end, block0_start:block0_end].all()
    block1_sees_prompt = mask[block1_start:block1_end, :prompt_len].all()
    print(f"Rule 5: Block 1 sees all of Block 0: {block1_sees_block0}")
    print(f"Rule 5: Block 1 sees Prompt: {block1_sees_prompt}")
    assert block1_sees_block0, "Block 1 should see all of Block 0!"
    assert block1_sees_prompt, "Block 1 should see Prompt!"

    print("\n✓ All attention mask rules validated!")
    return True


def test_position_ids_equivalence():
    """测试 position IDs 是否让每个 block 的计算等价于单独计算"""
    print("\n" + "=" * 60)
    print("Test 4: Position IDs Equivalence")
    print("=" * 60)

    block_size = 4
    prompt_len = 4
    response_len = 8

    input_ids = torch.arange(prompt_len + response_len)
    loss_mask = torch.cat([
        torch.zeros(prompt_len, dtype=torch.long),
        torch.ones(response_len, dtype=torch.long),
    ])

    mask_token_id = 999
    pad_token_id = 0

    interleaved_ids, interleaved_pos, _, _ = create_interleaved_training_data(
        input_ids=input_ids,
        loss_mask=loss_mask,
        block_size=block_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
    )

    print("\nPosition ID design:")
    print("=" * 40)

    # 新结构: [prompt] [masks0] [block0] [masks1] [block1]
    # Masks 0 的 position 应该从 prompt_len 开始
    masks0_start = prompt_len
    masks0_positions = interleaved_pos[masks0_start:masks0_start + 3].tolist()
    print(f"\nMasks 0 positions: {masks0_positions}")
    print(f"  -> Masks 0 'thinks' it follows {masks0_positions[0]} tokens (the prompt)")
    print(f"  -> This equals prompt length = {prompt_len}")

    expected_masks0_start_pos = prompt_len
    assert masks0_positions[0] == expected_masks0_start_pos, \
        f"Expected {expected_masks0_start_pos}, got {masks0_positions[0]}"

    # Block 0 的 position 也从 prompt_len 开始（和 Masks 0 重叠）
    block0_start = masks0_start + 3
    block0_positions = interleaved_pos[block0_start:block0_start + 4].tolist()
    print(f"\nBlock 0 positions: {block0_positions}")
    print(f"  -> Block 0 'thinks' it starts at position {block0_positions[0]}")
    print(f"  -> Same as Masks 0 start position (intentional overlap)")

    assert block0_positions[0] == expected_masks0_start_pos, \
        f"Block 0 should start at same position as Masks 0"

    # Block 1 的 position 应该是 8, 9, 10, 11
    # Masks 1 的 position 也应该从 8 开始
    masks1_start = block0_start + 4
    masks1_positions = interleaved_pos[masks1_start:masks1_start + 3].tolist()
    print(f"\nMasks 1 positions: {masks1_positions}")
    print(f"  -> Masks 1 'thinks' it follows {masks1_positions[0]} tokens")
    print(f"  -> This equals prompt({prompt_len}) + block0({block_size}) = {prompt_len + block_size}")

    expected_masks1_start_pos = prompt_len + block_size
    assert masks1_positions[0] == expected_masks1_start_pos, \
        f"Expected {expected_masks1_start_pos}, got {masks1_positions[0]}"

    # Block 1 position
    block1_start = masks1_start + 3
    block1_positions = interleaved_pos[block1_start:block1_start + 4].tolist()
    print(f"\nBlock 1 positions: {block1_positions}")
    print(f"  -> Block 1 'thinks' it starts at position {block1_positions[0]}")
    print(f"  -> Same as Masks 1 start position (intentional overlap)")

    # 关键: Block 1 和 Masks 1 的起始 position 应该相同
    assert block1_positions[0] == masks1_positions[0], \
        f"Block 1 start pos ({block1_positions[0]}) should equal Masks 1 start pos ({masks1_positions[0]})"

    print(f"\n✓ Position ID equivalence verified!")
    print(f"  Masks/Block pairs share the same starting position:")
    print(f"    - Masks 0 and Block 0 both start at position {masks0_positions[0]} (after prompt)")
    print(f"    - Masks 1 and Block 1 both start at position {masks1_positions[0]} (after prompt + block0)")
    print(f"  This design allows parallel prediction while maintaining AR position semantics")

    return True


def main():
    print("\n" + "=" * 70)
    print("     Testing Interleaved Training Data Construction")
    print("=" * 70)

    results = []

    results.append(("Basic Interleaving", test_basic_interleaving()))
    results.append(("Labels with AR Shift", test_labels()))
    results.append(("Attention Mask Rules", test_attention_mask()))
    results.append(("Position IDs Equivalence", test_position_ids_equivalence()))

    print("\n" + "=" * 70)
    print("     Summary")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ All tests passed! The interleaved training data construction is correct.\n")
    else:
        print("\n✗ Some tests failed. Please check the implementation.\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
