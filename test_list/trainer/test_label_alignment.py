"""
标签对齐验证测试

验证交错训练中 labels 和 input_ids 的对齐关系是否正确。
这是最关键的测试之一，确保模型学习到正确的预测目标。

测试策略：
1. 使用已知的简单样本，手动计算期望的 labels
2. 使用 Dataset 的 create_interleaved_labels 函数生成实际的 labels
3. 逐位置对比，确保完全一致
"""

import sys
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import create_interleaved_labels


def test_label_alignment_case1():
    """
    测试用例 1: 简单的单 block 情况

    原始序列: [P1, P2, P3, R1, R2, R3, R4]
    交错序列: [P1, P2, P3, M1, M2, M3, R1, R2, R3, R4]

    期望 labels:
    - P1, P2, P3: -100 (prompt 不计算 loss)
    - M1: R2 (mask 预测 block 中第2个token)
    - M2: R3 (mask 预测 block 中第3个token)
    - M3: R4 (mask 预测 block 中第4个token)
    - R1: R2 (real token 预测下一个 token)
    - R2: R3
    - R3: R4
    - R4: -100 (最后一个 token 没有下一个)
    """
    print("=" * 80)
    print("测试用例 1: 单 block 情况")
    print("=" * 80)

    device = torch.device("cpu")
    block_size = 4
    prompt_len = 3
    pad_token_id = 0
    mask_token_id = 999

    # 原始序列（不含 mask）
    original_input_ids = torch.tensor([1, 2, 3, 10, 11, 12, 13], device=device)

    # 交错序列（插入了 mask）
    # [P1, P2, P3, M1, M2, M3, R1, R2, R3, R4]
    interleaved_input_ids = torch.tensor([1, 2, 3, 999, 999, 999, 10, 11, 12, 13], device=device)

    # Loss mask（哪些位置需要计算 loss）
    # Prompt (3) 不计算，Masks (3) 计算，Real (4) 计算
    interleaved_loss_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], device=device)

    # 手动计算期望的 labels
    expected_labels = torch.tensor([
        -100, -100, -100,  # Prompt
        11, 12, 13,         # M1->R2, M2->R3, M3->R4
        11, 12, 13, -100    # R1->R2, R2->R3, R3->R4, R4->-100
    ], device=device)

    # 使用函数生成实际的 labels
    actual_labels = create_interleaved_labels(
        original_input_ids=original_input_ids,
        interleaved_input_ids=interleaved_input_ids,
        interleaved_loss_mask=interleaved_loss_mask,
        block_size=block_size,
        prompt_len=prompt_len,
        pad_token_id=pad_token_id,
        ignore_index=-100,
    )

    print(f"\n原始序列:   {original_input_ids.tolist()}")
    print(f"交错序列:   {interleaved_input_ids.tolist()}")
    print(f"期望 labels: {expected_labels.tolist()}")
    print(f"实际 labels: {actual_labels.tolist()}")

    # 逐位置验证
    mismatches = []
    for i in range(len(expected_labels)):
        if expected_labels[i] != actual_labels[i]:
            mismatches.append({
                'position': i,
                'expected': expected_labels[i].item(),
                'actual': actual_labels[i].item(),
                'input_token': interleaved_input_ids[i].item(),
            })

    if mismatches:
        print(f"\n❌ 发现 {len(mismatches)} 个不匹配:")
        for m in mismatches:
            print(f"  位置 {m['position']}: input={m['input_token']}, 期望={m['expected']}, 实际={m['actual']}")
        return False
    else:
        print("\n✅ 所有位置的标签都正确对齐")
        return True


def test_label_alignment_case2():
    """
    测试用例 2: 多 block 情况

    原始序列: [P1, P2, R1, R2, R3, R4, R5, R6, R7, R8]
    Block size = 4

    交错序列: [P1, P2, M1, M2, M3, R1, R2, R3, R4, M4, M5, M6, R5, R6, R7, R8]

    Block 0: R1, R2, R3, R4
    Block 1: R5, R6, R7, R8
    """
    print("\n" + "=" * 80)
    print("测试用例 2: 多 block 情况")
    print("=" * 80)

    device = torch.device("cpu")
    block_size = 4
    prompt_len = 2
    pad_token_id = 0
    mask_token_id = 999

    # 原始序列
    original_input_ids = torch.tensor([1, 2, 10, 11, 12, 13, 20, 21, 22, 23], device=device)

    # 交错序列
    interleaved_input_ids = torch.tensor([
        1, 2,                    # Prompt
        999, 999, 999,           # Masks before block 0
        10, 11, 12, 13,          # Block 0
        999, 999, 999,           # Masks before block 1
        20, 21, 22, 23           # Block 1
    ], device=device)

    interleaved_loss_mask = torch.tensor([
        0, 0,           # Prompt
        1, 1, 1,        # Masks
        1, 1, 1, 1,     # Block 0
        1, 1, 1,        # Masks
        1, 1, 1, 1      # Block 1
    ], device=device)

    # 手动计算期望的 labels
    expected_labels = torch.tensor([
        -100, -100,      # Prompt
        11, 12, 13,      # M1->R2, M2->R3, M3->R4
        11, 12, 13, 20,  # R1->R2, R2->R3, R3->R4, R4->R5
        21, 22, 23,      # M4->R6, M5->R7, M6->R8
        21, 22, 23, -100 # R5->R6, R6->R7, R7->R8, R8->-100
    ], device=device)

    actual_labels = create_interleaved_labels(
        original_input_ids=original_input_ids,
        interleaved_input_ids=interleaved_input_ids,
        interleaved_loss_mask=interleaved_loss_mask,
        block_size=block_size,
        prompt_len=prompt_len,
        pad_token_id=pad_token_id,
        ignore_index=-100,
    )

    print(f"\n原始序列:   {original_input_ids.tolist()}")
    print(f"交错序列:   {interleaved_input_ids.tolist()}")
    print(f"期望 labels: {expected_labels.tolist()}")
    print(f"实际 labels: {actual_labels.tolist()}")

    # 逐位置验证
    mismatches = []
    for i in range(len(expected_labels)):
        if expected_labels[i] != actual_labels[i]:
            mismatches.append({
                'position': i,
                'expected': expected_labels[i].item(),
                'actual': actual_labels[i].item(),
                'input_token': interleaved_input_ids[i].item(),
            })

    if mismatches:
        print(f"\n❌ 发现 {len(mismatches)} 个不匹配:")
        for m in mismatches:
            print(f"  位置 {m['position']}: input={m['input_token']}, 期望={m['expected']}, 实际={m['actual']}")
        return False
    else:
        print("\n✅ 所有位置的标签都正确对齐")
        return True


def test_label_alignment_case3():
    """
    测试用例 3: 最后一个 block 不完整的情况

    原始序列: [P1, R1, R2, R3, R4, R5]
    Block size = 4

    最后一个 block 只有 2 个 token (R5, R6)
    按照逻辑，不足 block_size-1 = 3 个 token，不应该插入 mask

    交错序列: [P1, M1, M2, M3, R1, R2, R3, R4, R5]
    """
    print("\n" + "=" * 80)
    print("测试用例 3: 最后 block 不完整（少于 block_size-1）")
    print("=" * 80)

    device = torch.device("cpu")
    block_size = 4
    prompt_len = 1
    pad_token_id = 0

    # 原始序列：6 个 token（1个prompt + 5个response）
    # Block 0: R1, R2, R3, R4 (完整)
    # Block 1: R5 (只有1个，不足3个，不插入mask)
    original_input_ids = torch.tensor([1, 10, 11, 12, 13, 20], device=device)

    # 交错序列：只在第一个 block 前插入 mask
    interleaved_input_ids = torch.tensor([
        1,                  # Prompt
        999, 999, 999,      # Masks before block 0
        10, 11, 12, 13,     # Block 0
        20                  # Block 1 (no masks, too short)
    ], device=device)

    interleaved_loss_mask = torch.tensor([
        0,              # Prompt
        1, 1, 1,        # Masks
        1, 1, 1, 1,     # Block 0
        1               # Block 1
    ], device=device)

    # 手动计算期望的 labels
    expected_labels = torch.tensor([
        -100,           # Prompt
        11, 12, 13,     # M1->R2, M2->R3, M3->R4
        11, 12, 13, 20, # R1->R2, R2->R3, R3->R4, R4->R5
        -100            # R5->-100 (最后一个)
    ], device=device)

    actual_labels = create_interleaved_labels(
        original_input_ids=original_input_ids,
        interleaved_input_ids=interleaved_input_ids,
        interleaved_loss_mask=interleaved_loss_mask,
        block_size=block_size,
        prompt_len=prompt_len,
        pad_token_id=pad_token_id,
        ignore_index=-100,
    )

    print(f"\n原始序列:   {original_input_ids.tolist()}")
    print(f"交错序列:   {interleaved_input_ids.tolist()}")
    print(f"期望 labels: {expected_labels.tolist()}")
    print(f"实际 labels: {actual_labels.tolist()}")

    # 逐位置验证
    mismatches = []
    for i in range(len(expected_labels)):
        if expected_labels[i] != actual_labels[i]:
            mismatches.append({
                'position': i,
                'expected': expected_labels[i].item(),
                'actual': actual_labels[i].item(),
                'input_token': interleaved_input_ids[i].item(),
            })

    if mismatches:
        print(f"\n❌ 发现 {len(mismatches)} 个不匹配:")
        for m in mismatches:
            print(f"  位置 {m['position']}: input={m['input_token']}, 期望={m['expected']}, 实际={m['actual']}")
        return False
    else:
        print("\n✅ 所有位置的标签都正确对齐")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("标签对齐验证测试")
    print("=" * 100 + "\n")

    results = []

    # 运行所有测试用例
    results.append(("用例1: 单block", test_label_alignment_case1()))
    results.append(("用例2: 多block", test_label_alignment_case2()))
    results.append(("用例3: 最后block不完整", test_label_alignment_case3()))

    # 总结
    print("\n" + "=" * 100)
    print("测试总结")
    print("=" * 100)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    if all_passed:
        print("\n✅ 所有标签对齐测试通过！")
        print("=" * 100 + "\n")
        exit(0)
    else:
        print("\n❌ 部分测试失败！请检查 create_interleaved_labels 的实现")
        print("=" * 100 + "\n")
        exit(1)
