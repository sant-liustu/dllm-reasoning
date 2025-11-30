#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Mask Token 的上下文隔离性

验证关键设计：
对于交错序列 [P][M0][R0][M1][R1]：
- M1 只能看到 P（看不到 R0）
- 因此 M1 的 attention mask 应该和只有 [P][M1] 时完全一样
- 这验证了 attention mask 正确隔离了不同 block 之间的信息流

测试方法：
1. 构造完整序列的 mask：[P][M0][R0][M1]
2. 构造截断序列的 mask：[P][M1]（M1 位置相同）
3. 验证 M1 对应行的 mask 完全一致
"""

import sys
sys.path.insert(0, "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning")

import torch
from transformers import AutoTokenizer
from dllm_reasoning.model.DLLM import DLLMForCausalLM
from dllm_reasoning.trainer.flex_attention_utils import create_mask_mod_from_block_info

try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    print("⚠️  FlexAttention 不可用，将跳过测试")


def create_simple_block_mask(block_info, prompt_len, seq_len, device):
    """为单个样本创建 BlockMask"""
    if not FLEX_ATTN_AVAILABLE:
        raise RuntimeError("FlexAttention not available")

    # 创建 mask_mod
    mask_mod = create_mask_mod_from_block_info(
        block_info=block_info,
        prompt_len=prompt_len,
        seq_len=seq_len,
    )

    # 预计算 2D mask
    mask_2d = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for q in range(seq_len):
        for kv in range(seq_len):
            mask_2d[q, kv] = mask_mod(0, 0, q, kv)

    # 扩展为 batch
    mask_2d = mask_2d.unsqueeze(0)  # [1, seq_len, seq_len]

    # 创建 BlockMask
    block_mask = create_block_mask(
        lambda b, h, q_idx, kv_idx: mask_2d[b, q_idx, kv_idx],
        B=1,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    return block_mask


def test_mask_isolation_basic():
    """
    测试基本的 mask 隔离性

    场景：[P][M0][R0][M1]
    验证：M1 只能看到 P，看不到 M0 和 R0
    因此：M1 的计算应该和 [P][M1] 完全一样
    """
    print("=" * 100)
    print("Test: Mask Context Isolation (Basic)")
    print("=" * 100)

    if not FLEX_ATTN_AVAILABLE:
        print("⚠️  跳过测试（FlexAttention 不可用）\n")
        return

    # 加载 DLLM 模型和 tokenizer
    print("\n加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DLLMForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    # 保持训练模式以使用 FlexAttention（推理模式不支持 BlockMask）
    model.train()
    print(f"✓ 模型加载成功（设备: {device}，训练模式）")

    # 构造测试数据
    prompt_len = 5
    mask_len = 3
    block_len = 4

    # Prompt tokens (随机)
    prompt_tokens = torch.randint(1000, 5000, (prompt_len,), device=device)

    # Mask tokens (EOS)
    mask_token_id = tokenizer.eos_token_id
    mask_tokens = torch.full((mask_len,), mask_token_id, device=device)

    # Real block tokens (随机)
    block_tokens = torch.randint(5000, 10000, (block_len,), device=device)

    print(f"\n测试设置:")
    print(f"  Prompt 长度: {prompt_len}")
    print(f"  Mask 长度: {mask_len}")
    print(f"  Block 长度: {block_len}")
    print(f"  Mask token ID: {mask_token_id}")

    # ========== 场景 1: 完整序列 [P][M0][R0][M1] ==========
    print(f"\n场景 1: 完整序列 [P][M0][R0][M1]")

    # 构造 input_ids
    full_seq = torch.cat([
        prompt_tokens,      # [0-4]: Prompt
        mask_tokens,        # [5-7]: M0
        block_tokens,       # [8-11]: R0
        mask_tokens,        # [12-14]: M1
    ], dim=0)

    full_seq_len = len(full_seq)
    print(f"  序列长度: {full_seq_len}")
    print(f"  结构: [0-{prompt_len-1}] Prompt, [{prompt_len}-{prompt_len+mask_len-1}] M0, "
          f"[{prompt_len+mask_len}-{prompt_len+mask_len+block_len-1}] R0, "
          f"[{prompt_len+mask_len+block_len}-{full_seq_len-1}] M1")

    # 构造 position_ids（M0 和 R0 共享位置，M1 在新位置）
    full_position_ids = torch.cat([
        torch.arange(0, prompt_len, device=device),                                           # Prompt: 0-4
        torch.arange(prompt_len, prompt_len + mask_len, device=device),                      # M0: 5-7
        torch.arange(prompt_len, prompt_len + block_len, device=device),                      # R0: 5-8
        torch.arange(prompt_len + block_len, prompt_len + block_len + mask_len, device=device), # M1: 9-11
    ], dim=0)

    print(f"  Position IDs: {full_position_ids.tolist()}")

    # 构造 block_info
    block_info_full = [
        ('mask', mask_len),
        ('real', block_len),
        ('mask', mask_len),
    ]

    # 创建 BlockMask
    block_mask_full = create_simple_block_mask(
        block_info=block_info_full,
        prompt_len=prompt_len,
        seq_len=full_seq_len,
        device=device,
    )

    # 前向传播（获取 hidden states）
    with torch.no_grad():
        outputs_full = model(
            input_ids=full_seq.unsqueeze(0),
            position_ids=full_position_ids.unsqueeze(0),
            attention_mask=block_mask_full,
            output_hidden_states=True,
        )

    # 提取 M1 位置的 hidden states
    m1_start = prompt_len + mask_len + block_len
    m1_end = m1_start + mask_len

    hidden_states_full = outputs_full.hidden_states[-1]  # 最后一层
    m1_hidden_full = hidden_states_full[0, m1_start:m1_end, :]  # [mask_len, hidden_dim]

    print(f"  M1 位置: [{m1_start}-{m1_end-1}]")
    print(f"  M1 hidden states shape: {m1_hidden_full.shape}")
    print(f"  M1 hidden states 范数: {m1_hidden_full.norm(dim=-1).tolist()}")

    # ========== 场景 2: 截断序列 [P][M1] ==========
    print(f"\n场景 2: 截断序列 [P][M1]（M1 位置对齐）")

    # 构造 input_ids（只有 Prompt + M1）
    short_seq = torch.cat([
        prompt_tokens,      # [0-4]: Prompt
        mask_tokens,        # [5-7]: M1
    ], dim=0)

    short_seq_len = len(short_seq)
    print(f"  序列长度: {short_seq_len}")
    print(f"  结构: [0-{prompt_len-1}] Prompt, [{prompt_len}-{short_seq_len-1}] M1")

    # 构造 position_ids（M1 的位置要和完整序列中的 M1 一致）
    short_position_ids = torch.cat([
        torch.arange(0, prompt_len, device=device),                                           # Prompt: 0-4
        torch.arange(prompt_len + block_len, prompt_len + block_len + mask_len, device=device), # M1: 9-11（和完整序列一致！）
    ], dim=0)

    print(f"  Position IDs: {short_position_ids.tolist()}")

    # 构造 block_info
    block_info_short = [
        ('mask', mask_len),
    ]

    # 创建 BlockMask
    block_mask_short = create_simple_block_mask(
        block_info=block_info_short,
        prompt_len=prompt_len,
        seq_len=short_seq_len,
        device=device,
    )

    # 前向传播
    with torch.no_grad():
        outputs_short = model(
            input_ids=short_seq.unsqueeze(0),
            position_ids=short_position_ids.unsqueeze(0),
            attention_mask=block_mask_short,
            output_hidden_states=True,
        )

    # 提取 M1 位置的 hidden states
    m1_start_short = prompt_len
    m1_end_short = m1_start_short + mask_len

    hidden_states_short = outputs_short.hidden_states[-1]
    m1_hidden_short = hidden_states_short[0, m1_start_short:m1_end_short, :]

    print(f"  M1 位置: [{m1_start_short}-{m1_end_short-1}]")
    print(f"  M1 hidden states shape: {m1_hidden_short.shape}")
    print(f"  M1 hidden states 范数: {m1_hidden_short.norm(dim=-1).tolist()}")

    # ========== 验证：两个场景中 M1 的 hidden states 应该完全一致 ==========
    print(f"\n" + "=" * 80)
    print(f"验证结果")
    print(f"=" * 80)

    # 计算差异
    diff = (m1_hidden_full - m1_hidden_short).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nM1 hidden states 差异:")
    print(f"  最大差异: {max_diff:.6e}")
    print(f"  平均差异: {mean_diff:.6e}")

    # 验证阈值（考虑浮点误差）
    threshold = 1e-5

    if max_diff < threshold:
        print(f"\n✅ 验证通过！")
        print(f"   M1 的计算在两个场景中完全一致（差异 < {threshold}）")
        print(f"   这证明 attention mask 正确隔离了 M1 和 R0 之间的信息流")
        return True
    else:
        print(f"\n❌ 验证失败！")
        print(f"   M1 的计算在两个场景中不一致（最大差异 {max_diff:.6e} >= {threshold}）")
        print(f"   这可能说明 attention mask 没有正确隔离信息流")

        # 打印详细的差异信息
        print(f"\n详细差异（每个 token 的平均差异）:")
        for i in range(mask_len):
            token_diff = diff[i].mean().item()
            print(f"  M1[{i}]: {token_diff:.6e}")

        return False


def test_mask_isolation_multiple_blocks():
    """
    测试多个 block 的 mask 隔离性

    场景：[P][M0][R0][M1][R1][M2]
    验证：
    - M0 只看到 P
    - M1 只看到 P + R0（看不到 M0）
    - M2 只看到 P + R0 + R1（看不到 M0, M1）

    对于 M2，应该和 [P][R0][R1][M2] 的计算一致
    """
    print("\n" + "=" * 100)
    print("Test: Mask Context Isolation (Multiple Blocks)")
    print("=" * 100)

    if not FLEX_ATTN_AVAILABLE:
        print("⚠️  跳过测试（FlexAttention 不可用）\n")
        return

    print("\n加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DLLMForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    # 保持训练模式以使用 FlexAttention
    model.train()
    print(f"✓ 模型加载成功（训练模式）")

    # 构造测试数据
    prompt_len = 5
    mask_len = 3
    block_len = 4

    prompt_tokens = torch.randint(1000, 5000, (prompt_len,), device=device)
    mask_token_id = tokenizer.eos_token_id
    mask_tokens = torch.full((mask_len,), mask_token_id, device=device)
    block0_tokens = torch.randint(5000, 6000, (block_len,), device=device)
    block1_tokens = torch.randint(6000, 7000, (block_len,), device=device)

    # ========== 场景 1: 完整序列 [P][M0][R0][M1][R1][M2] ==========
    print(f"\n场景 1: 完整序列 [P][M0][R0][M1][R1][M2]")

    full_seq = torch.cat([
        prompt_tokens,   # [0-4]: P
        mask_tokens,     # [5-7]: M0
        block0_tokens,   # [8-11]: R0
        mask_tokens,     # [12-14]: M1
        block1_tokens,   # [15-18]: R1
        mask_tokens,     # [19-21]: M2
    ], dim=0)

    full_seq_len = len(full_seq)

    # Position IDs
    full_position_ids = torch.cat([
        torch.arange(0, prompt_len, device=device),                                           # P: 0-4
        torch.arange(prompt_len, prompt_len + mask_len, device=device),                      # M0: 5-7
        torch.arange(prompt_len, prompt_len + block_len, device=device),                      # R0: 5-8
        torch.arange(prompt_len + block_len, prompt_len + block_len + mask_len, device=device), # M1: 9-11
        torch.arange(prompt_len + block_len, prompt_len + 2*block_len, device=device),        # R1: 9-12
        torch.arange(prompt_len + 2*block_len, prompt_len + 2*block_len + mask_len, device=device), # M2: 13-15
    ], dim=0)

    # Block info
    block_info_full = [
        ('mask', mask_len),
        ('real', block_len),
        ('mask', mask_len),
        ('real', block_len),
        ('mask', mask_len),
    ]

    block_mask_full = create_simple_block_mask(
        block_info=block_info_full,
        prompt_len=prompt_len,
        seq_len=full_seq_len,
        device=device,
    )

    with torch.no_grad():
        outputs_full = model(
            input_ids=full_seq.unsqueeze(0),
            position_ids=full_position_ids.unsqueeze(0),
            attention_mask=block_mask_full,
            output_hidden_states=True,
        )

    # M2 位置
    m2_start = prompt_len + mask_len + block_len + mask_len + block_len
    m2_end = m2_start + mask_len

    hidden_states_full = outputs_full.hidden_states[-1]
    m2_hidden_full = hidden_states_full[0, m2_start:m2_end, :]

    print(f"  M2 位置: [{m2_start}-{m2_end-1}]")
    print(f"  M2 hidden states 范数: {m2_hidden_full.norm(dim=-1).tolist()}")

    # ========== 场景 2: 去掉所有 mask 的序列 [P][R0][R1][M2] ==========
    print(f"\n场景 2: 去掉所有 mask 的序列 [P][R0][R1][M2]")

    short_seq = torch.cat([
        prompt_tokens,   # [0-4]: P
        block0_tokens,   # [5-8]: R0
        block1_tokens,   # [9-12]: R1
        mask_tokens,     # [13-15]: M2
    ], dim=0)

    short_seq_len = len(short_seq)

    # Position IDs（M2 的位置要对齐）
    short_position_ids = torch.cat([
        torch.arange(0, prompt_len, device=device),                                           # P: 0-4
        torch.arange(prompt_len, prompt_len + block_len, device=device),                      # R0: 5-8
        torch.arange(prompt_len + block_len, prompt_len + 2*block_len, device=device),        # R1: 9-12
        torch.arange(prompt_len + 2*block_len, prompt_len + 2*block_len + mask_len, device=device), # M2: 13-15
    ], dim=0)

    # Block info
    block_info_short = [
        ('real', block_len),
        ('real', block_len),
        ('mask', mask_len),
    ]

    block_mask_short = create_simple_block_mask(
        block_info=block_info_short,
        prompt_len=prompt_len,
        seq_len=short_seq_len,
        device=device,
    )

    with torch.no_grad():
        outputs_short = model(
            input_ids=short_seq.unsqueeze(0),
            position_ids=short_position_ids.unsqueeze(0),
            attention_mask=block_mask_short,
            output_hidden_states=True,
        )

    # M2 位置
    m2_start_short = prompt_len + 2*block_len
    m2_end_short = m2_start_short + mask_len

    hidden_states_short = outputs_short.hidden_states[-1]
    m2_hidden_short = hidden_states_short[0, m2_start_short:m2_end_short, :]

    print(f"  M2 位置: [{m2_start_short}-{m2_end_short-1}]")
    print(f"  M2 hidden states 范数: {m2_hidden_short.norm(dim=-1).tolist()}")

    # 验证
    print(f"\n" + "=" * 80)
    print(f"验证结果")
    print(f"=" * 80)

    diff = (m2_hidden_full - m2_hidden_short).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nM2 hidden states 差异:")
    print(f"  最大差异: {max_diff:.6e}")
    print(f"  平均差异: {mean_diff:.6e}")

    threshold = 1e-5

    if max_diff < threshold:
        print(f"\n✅ 验证通过！")
        print(f"   M2 看不到之前的所有 mask tokens (M0, M1)")
        print(f"   M2 只依赖于 Prompt + R0 + R1")
        return True
    else:
        print(f"\n❌ 验证失败！")
        print(f"   M2 的计算可能受到了 M0 或 M1 的影响")
        return False


if __name__ == "__main__":
    try:
        print("\n" + "=" * 100)
        print("Mask Context Isolation Tests")
        print("=" * 100 + "\n")

        # 运行测试
        result1 = test_mask_isolation_basic()
        result2 = test_mask_isolation_multiple_blocks()

        if result1 and result2:
            print("\n" + "=" * 100)
            print("✅ 所有验证通过！")
            print("=" * 100 + "\n")
            exit(0)
        else:
            print("\n" + "=" * 100)
            print("❌ 部分验证失败！")
            print("=" * 100 + "\n")
            exit(1)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
