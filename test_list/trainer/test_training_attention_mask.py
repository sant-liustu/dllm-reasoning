#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试实际训练中的 attention mask 构造是否正确。

这个测试模拟实际训练流程，验证：
1. 从 interleaved dataset 获取数据
2. 根据 block_info 构造 FlexAttention mask
3. 验证 mask 的可见性规则是否符合预期
"""

import sys
sys.path.insert(0, "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning")

import torch
from dllm_reasoning.trainer.interleaved_sft_dataset import create_interleaved_training_data

try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    print("Warning: FlexAttention not available, using simplified mask verification")


def create_training_attention_mask_from_block_info(
    block_info,
    seq_len,
    prompt_len,
    device,
):
    """
    根据 block_info 构造训练时使用的 attention mask。

    新结构: [prompt] [mask0] [block0] [mask1] [block1] ...

    规则:
    1. Prompt: 标准 causal
    2. Mask:
       - 可以看 prompt 和之前所有 real blocks
       - 在同组内 causal
       - 看不到任何其他 mask
    3. Real block:
       - 可以看 prompt 和之前所有 real blocks
       - 在同 block 内 causal
       - 看不到任何 mask
    """
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    # 构建 segment 信息
    segments = []
    current_pos = 0

    # Prompt segment
    segments.append(('prompt', current_pos, prompt_len))
    current_pos = prompt_len

    # 从 block_info 构建其他 segments
    for seg_type, _, seg_len in block_info:
        segments.append((seg_type, current_pos, seg_len))
        current_pos += seg_len

    # 填充 mask
    for q_idx, (q_type, q_start, q_len) in enumerate(segments):
        for kv_idx, (kv_type, kv_start, kv_len) in enumerate(segments):

            if q_type == 'prompt':
                # Prompt 内部 causal
                if kv_type == 'prompt':
                    for i in range(q_len):
                        for j in range(min(i + 1, kv_len)):
                            mask[q_start + i, kv_start + j] = True

            elif q_type == 'mask':
                if kv_type == 'prompt':
                    # Mask 可以看整个 prompt
                    mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True

                elif kv_type == 'real':
                    # Mask 可以看之前的所有 real blocks
                    # 找到这个 mask 对应的 block (它后面的第一个 real)
                    mask_block_idx = None
                    for seg_idx in range(q_idx + 1, len(segments)):
                        if segments[seg_idx][0] == 'real':
                            mask_block_idx = seg_idx
                            break

                    # Mask 可以看这个 block 之前的所有 real blocks
                    if kv_idx < q_idx and kv_type == 'real':
                        mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True

                elif kv_type == 'mask':
                    # 同一组 mask 内部 causal
                    if kv_idx == q_idx:
                        for i in range(q_len):
                            for j in range(min(i + 1, kv_len)):
                                mask[q_start + i, kv_start + j] = True
                    # 不同组 mask 互相看不见

            elif q_type == 'real':
                if kv_type == 'prompt':
                    # Real block 可以看整个 prompt
                    mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True

                elif kv_type == 'real':
                    if kv_idx < q_idx:
                        # 可以看之前的所有 real blocks
                        mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True
                    elif kv_idx == q_idx:
                        # 同 block 内 causal
                        for i in range(q_len):
                            for j in range(min(i + 1, kv_len)):
                                mask[q_start + i, kv_start + j] = True

                # Real block 看不到任何 mask (kv_type == 'mask' 时不设置)

    return mask, segments


def visualize_mask(mask, segments, title="Attention Mask"):
    """可视化 attention mask"""
    seq_len = mask.shape[0]

    # 创建 token 标签
    tokens = []
    for seg_type, seg_start, seg_len in segments:
        if seg_type == 'prompt':
            for i in range(seg_len):
                tokens.append(f"p{i}")
        elif seg_type == 'mask':
            # 找到这是第几组 mask
            mask_group_idx = sum(1 for s in segments[:segments.index((seg_type, seg_start, seg_len))] if s[0] == 'mask')
            for i in range(seg_len):
                tokens.append(f"M{mask_group_idx}_{i}")
        elif seg_type == 'real':
            # 找到这是第几个 block
            block_idx = sum(1 for s in segments[:segments.index((seg_type, seg_start, seg_len))] if s[0] == 'real')
            block_offset = block_idx * 4  # 假设 block_size=4
            for i in range(seg_len):
                tokens.append(f"r{block_offset + i}")

    print(f"\n{title}")
    print("-" * 100)

    # Header
    header = "       " + "".join([f"{i:>5}" for i in range(seq_len)])
    print(header)

    for i in range(seq_len):
        row = f"{i:>3}:  "
        for j in range(seq_len):
            if mask[i, j]:
                row += "  1  "
            else:
                row += "  .  "
        row += f"  <- {tokens[i]}"
        print(row)


def test_training_attention_mask():
    """测试实际训练中的 attention mask"""
    print("=" * 100)
    print("Testing Training Attention Mask Construction")
    print("=" * 100)

    # 模拟训练数据
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
    device = torch.device("cpu")

    # 创建交错数据
    interleaved_ids, interleaved_pos, interleaved_loss, block_info = create_interleaved_training_data(
        input_ids=input_ids,
        loss_mask=loss_mask,
        block_size=block_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
    )

    print(f"\nData structure:")
    print(f"  Input IDs: {interleaved_ids.tolist()}")
    print(f"  Position IDs: {interleaved_pos.tolist()}")
    print(f"  Block info: {block_info}")

    # 根据 block_info 构造 attention mask
    seq_len = interleaved_ids.shape[0]
    attn_mask, segments = create_training_attention_mask_from_block_info(
        block_info=block_info,
        seq_len=seq_len,
        prompt_len=prompt_len,
        device=device,
    )

    print(f"\nSegments:")
    for i, (seg_type, seg_start, seg_len) in enumerate(segments):
        print(f"  [{i}] {seg_type}: positions {seg_start}-{seg_start + seg_len - 1} (len={seg_len})")

    # 可视化
    visualize_mask(attn_mask, segments, "Training Attention Mask")

    # 验证规则
    print("\n" + "=" * 100)
    print("Validation")
    print("=" * 100)

    # 找到各个 segment 的位置
    prompt_seg = segments[0]
    assert prompt_seg[0] == 'prompt'

    mask0_seg = segments[1]
    assert mask0_seg[0] == 'mask'
    mask0_start, mask0_len = mask0_seg[1], mask0_seg[2]

    block0_seg = segments[2]
    assert block0_seg[0] == 'real'
    block0_start, block0_len = block0_seg[1], block0_seg[2]

    mask1_seg = segments[3]
    assert mask1_seg[0] == 'mask'
    mask1_start, mask1_len = mask1_seg[1], mask1_seg[2]

    block1_seg = segments[4]
    assert block1_seg[0] == 'real'
    block1_start, block1_len = block1_seg[1], block1_seg[2]

    # Rule 1: Real blocks 看不到 mask
    print("\n[Rule 1] Real blocks cannot see mask tokens")
    block0_sees_mask0 = attn_mask[block0_start:block0_start + block0_len, mask0_start:mask0_start + mask0_len].any().item()
    block1_sees_mask0 = attn_mask[block1_start:block1_start + block1_len, mask0_start:mask0_start + mask0_len].any().item()
    block1_sees_mask1 = attn_mask[block1_start:block1_start + block1_len, mask1_start:mask1_start + mask1_len].any().item()

    print(f"  Block 0 sees Mask 0: {block0_sees_mask0} (should be False)")
    print(f"  Block 1 sees Mask 0: {block1_sees_mask0} (should be False)")
    print(f"  Block 1 sees Mask 1: {block1_sees_mask1} (should be False)")

    assert not block0_sees_mask0, "❌ Block 0 should not see Mask 0!"
    assert not block1_sees_mask0, "❌ Block 1 should not see Mask 0!"
    assert not block1_sees_mask1, "❌ Block 1 should not see Mask 1!"
    print("  ✓ All checks passed!")

    # Rule 2: Masks 可以看 prompt 和之前的 real blocks
    print("\n[Rule 2] Masks can see prompt and previous real blocks")
    mask0_sees_prompt = attn_mask[mask0_start:mask0_start + mask0_len, :prompt_len].all().item()
    mask1_sees_prompt = attn_mask[mask1_start:mask1_start + mask1_len, :prompt_len].all().item()
    mask1_sees_block0 = attn_mask[mask1_start:mask1_start + mask1_len, block0_start:block0_start + block0_len].all().item()

    print(f"  Mask 0 sees Prompt: {mask0_sees_prompt} (should be True)")
    print(f"  Mask 1 sees Prompt: {mask1_sees_prompt} (should be True)")
    print(f"  Mask 1 sees Block 0: {mask1_sees_block0} (should be True)")

    assert mask0_sees_prompt, "❌ Mask 0 should see Prompt!"
    assert mask1_sees_prompt, "❌ Mask 1 should see Prompt!"
    assert mask1_sees_block0, "❌ Mask 1 should see Block 0!"
    print("  ✓ All checks passed!")

    # Rule 3: Masks 在同组内 causal
    print("\n[Rule 3] Masks are causal within their group")
    m0_0_sees_m0_1 = attn_mask[mask0_start, mask0_start + 1].item()
    m0_1_sees_m0_0 = attn_mask[mask0_start + 1, mask0_start].item()

    print(f"  M0_0 sees M0_1: {m0_0_sees_m0_1} (should be False)")
    print(f"  M0_1 sees M0_0: {m0_1_sees_m0_0} (should be True)")

    assert not m0_0_sees_m0_1, "❌ M0_0 should not see M0_1!"
    assert m0_1_sees_m0_0, "❌ M0_1 should see M0_0!"
    print("  ✓ All checks passed!")

    # Rule 4: Real blocks 在同 block 内 causal
    print("\n[Rule 4] Real blocks are causal within their block")
    r0_sees_r1 = attn_mask[block0_start, block0_start + 1].item()
    r1_sees_r0 = attn_mask[block0_start + 1, block0_start].item()

    print(f"  r0 sees r1: {r0_sees_r1} (should be False)")
    print(f"  r1 sees r0: {r1_sees_r0} (should be True)")

    assert not r0_sees_r1, "❌ r0 should not see r1!"
    assert r1_sees_r0, "❌ r1 should see r0!"
    print("  ✓ All checks passed!")

    # Rule 5: Block 1 可以看 Block 0
    print("\n[Rule 5] Block 1 can see Block 0")
    block1_sees_block0 = attn_mask[block1_start:block1_start + block1_len, block0_start:block0_start + block0_len].all().item()

    print(f"  Block 1 sees Block 0: {block1_sees_block0} (should be True)")

    assert block1_sees_block0, "❌ Block 1 should see Block 0!"
    print("  ✓ All checks passed!")

    # Rule 6: Masks 之间互相看不见 (不同组)
    print("\n[Rule 6] Different mask groups cannot see each other")
    mask0_sees_mask1 = attn_mask[mask0_start:mask0_start + mask0_len, mask1_start:mask1_start + mask1_len].any().item()
    mask1_sees_mask0 = attn_mask[mask1_start:mask1_start + mask1_len, mask0_start:mask0_start + mask0_len].any().item()

    print(f"  Mask 0 sees Mask 1: {mask0_sees_mask1} (should be False)")
    print(f"  Mask 1 sees Mask 0: {mask1_sees_mask0} (should be False)")

    assert not mask0_sees_mask1, "❌ Mask 0 should not see Mask 1!"
    assert not mask1_sees_mask0, "❌ Mask 1 should not see Mask 0!"
    print("  ✓ All checks passed!")

    print("\n" + "=" * 100)
    print("✓ ALL TRAINING ATTENTION MASK RULES VALIDATED!")
    print("=" * 100)

    return True


if __name__ == "__main__":
    try:
        success = test_training_attention_mask()
        if success:
            print("\n✅ Test completed successfully!")
            exit(0)
        else:
            print("\n❌ Test failed!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
