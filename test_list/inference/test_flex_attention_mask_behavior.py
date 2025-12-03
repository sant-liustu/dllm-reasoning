#!/usr/bin/env python3
"""
测试FlexAttention的mask应用时机

验证mask是在softmax之前还是之后应用：
- 正确：score -> mask(-inf) -> softmax -> matmul(V)
- 错误：score -> softmax -> mask(0) -> matmul(V)
"""

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def test_flex_attention_mask_timing():
    """
    测试FlexAttention的mask应用时机

    如果mask在softmax之前应用（正确）：
    - 被mask的位置应该完全不影响attention输出
    - 不同序列长度但mask pattern相同时，输出应该相同

    如果mask在softmax之后应用（错误）：
    - 被mask的位置会影响softmax归一化
    - 不同序列长度会导致不同的输出
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 简单设置
    batch_size = 1
    num_heads = 2
    head_dim = 4

    # 配置1: 短序列 (seq_len = 4)
    seq_len_1 = 4
    Q1 = torch.randn(batch_size, num_heads, seq_len_1, head_dim, device=device)
    K1 = torch.randn(batch_size, num_heads, seq_len_1, head_dim, device=device)
    V1 = torch.randn(batch_size, num_heads, seq_len_1, head_dim, device=device)

    # 配置2: 长序列 (seq_len = 6)
    # 前4个位置的Q/K/V与配置1完全相同，后2个位置是新的
    seq_len_2 = 6
    Q2 = torch.cat([Q1[0], torch.randn(num_heads, 2, head_dim, device=device)], dim=1).unsqueeze(0)
    K2 = torch.cat([K1[0], torch.randn(num_heads, 2, head_dim, device=device)], dim=1).unsqueeze(0)
    V2 = torch.cat([V1[0], torch.randn(num_heads, 2, head_dim, device=device)], dim=1).unsqueeze(0)

    # Mask1: 每个位置只能看到自己和之前的位置（causal mask）
    def causal_mask_short(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask_1 = create_block_mask(
        causal_mask_short,
        B=batch_size, H=None,
        Q_LEN=seq_len_1, KV_LEN=seq_len_1,
    )

    # Mask2: 前4个位置只能看到前4个位置（后2个位置被mask）
    def mask_later_positions(b, h, q_idx, kv_idx):
        # 使用条件表达式而非if-else，避免data-dependent control flow
        # 前4个query位置只能看到前4个kv位置
        mask_first4 = (q_idx >= kv_idx) & (kv_idx < 4)
        # 后面的位置使用标准causal mask
        mask_rest = q_idx >= kv_idx
        # 根据q_idx选择mask
        return (q_idx < 4) * mask_first4 + (q_idx >= 4) * mask_rest

    block_mask_2 = create_block_mask(
        mask_later_positions,
        B=batch_size, H=None,
        Q_LEN=seq_len_2, KV_LEN=seq_len_2,
    )

    # 前向传播
    output_1 = flex_attention(Q1, K1, V1, block_mask=block_mask_1)
    output_2 = flex_attention(Q2, K2, V2, block_mask=block_mask_2)

    # 比较前4个位置的输出
    output_1_first4 = output_1[0, :, :4, :]  # [num_heads, 4, head_dim]
    output_2_first4 = output_2[0, :, :4, :]  # [num_heads, 4, head_dim]

    print("="*80)
    print("FlexAttention Mask应用时机测试")
    print("="*80)

    print(f"\n配置1 (短序列):")
    print(f"  序列长度: {seq_len_1}")
    print(f"  Mask: 每个位置只能看到自己和之前的位置")
    print(f"  输出前4个位置: {output_1_first4[0, :, 0]}")  # 第一个head的所有位置的第一维

    print(f"\n配置2 (长序列):")
    print(f"  序列长度: {seq_len_2}")
    print(f"  Mask: 前4个位置只能看到前4个位置（后2个被mask）")
    print(f"  输出前4个位置: {output_2_first4[0, :, 0]}")

    # 计算差异
    diff = torch.norm(output_1_first4 - output_2_first4).item()
    print(f"\n前4个位置的L2差异: {diff:.10f}")

    print("\n" + "="*80)
    print("结论分析")
    print("="*80)

    if diff < 1e-5:
        print("✅ 差异极小！")
        print("   这说明：mask在softmax**之前**应用（正确）")
        print("   - 被mask的位置（4-5）完全不影响前4个位置的attention输出")
        print("   - 即使序列长度不同，只要可见部分相同，输出就相同")
        print("\n   ⚠️ 这意味着我们之前的分析有误！")
        print("   FlexAttention的实现是正确的，问题可能在其他地方...")
    else:
        print("❌ 差异显著！")
        print("   这说明：mask在softmax**之后**应用（错误）")
        print("   - 被mask的位置（4-5）虽然权重为0，但影响了softmax归一化")
        print("   - 不同序列长度会导致不同的attention分布")
        print("\n   ⚠️ 这是bug！需要修复FlexAttention的mask应用方式")

    return diff


def test_manual_attention_with_mask():
    """
    手动实现attention来验证正确的mask应用方式
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    num_heads = 2
    seq_len = 4
    head_dim = 4

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # 计算attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, H, seq_len, seq_len]

    print("\n" + "="*80)
    print("手动Attention计算示例")
    print("="*80)

    print(f"\n原始scores (第一个head，位置0对所有位置):")
    print(scores[0, 0, 0, :])

    # 应用causal mask (-inf)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    masked_scores = scores[0, 0] + causal_mask

    print(f"\n应用mask后的scores (位置0):")
    print(masked_scores[0, :])

    # Softmax
    attn_weights = torch.softmax(masked_scores, dim=-1)

    print(f"\nSoftmax后的attention weights (位置0):")
    print(attn_weights[0, :])
    print(f"权重和: {attn_weights[0, :].sum().item():.6f} (应该为1.0)")

    print("\n说明：")
    print("  - 被mask位置的score为-inf")
    print("  - Softmax后这些位置的权重自动变为0")
    print("  - 只有可见位置参与归一化")


if __name__ == "__main__":
    # 测试FlexAttention的mask应用时机
    diff = test_flex_attention_mask_timing()

    # 手动演示正确的mask应用方式
    test_manual_attention_with_mask()
