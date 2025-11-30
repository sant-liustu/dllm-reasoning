#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Mask Token 的上下文隔离性（简化版）

验证关键设计：
对于交错序列 [P][M0][R0][M1]：
- M1 只能看到 P（看不到 R0 和 M0）
- 因此 M1 位置的计算应该和只有 [P][M1] 时完全一样

测试方法：
1. 直接调用模型的第一个 attention 层（避开完整模型的 block diffusion 逻辑）
2. 验证包括 position embedding 在内的完整计算流程
3. 对比两个场景中 M1 位置的 attention output
"""

import sys
sys.path.insert(0, "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning")

import os
# Disable torch.compile to avoid warnings.warn issue
os.environ["TORCH_COMPILE_DISABLE"] = "1"

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


def test_mask_isolation_with_multiple_layers(num_layers=5):
    """
    使用多个 attention 层测试隔离性，验证误差不会累积

    Args:
        num_layers: 要测试的层数（默认 5 层）
    """
    print("=" * 100)
    print(f"Test: Mask Context Isolation (Multiple Layers, testing {num_layers} layers)")
    print("=" * 100)

    if not FLEX_ATTN_AVAILABLE:
        print("⚠️  跳过测试（FlexAttention 不可用）\n")
        return False

    # 加载模型
    print("\n加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DLLMForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)

    print(f"✓ 模型加载成功（设备: {device}）")

    # 提取需要的组件
    embedding_layer = model.model.embed_tokens
    rotary_emb = model.model.rotary_emb
    norm_layer = model.model.norm

    # 获取所有层（或指定层数）
    if num_layers is None or num_layers <= 0:
        decoder_layers = model.model.layers  # 所有层
        num_layers = len(decoder_layers)
    else:
        decoder_layers = model.model.layers[:num_layers]

    # 设置为训练模式（使用 FlexAttention）
    for layer in decoder_layers:
        layer.train()

    print(f"将测试 {num_layers} 个 decoder 层（模型共 {len(model.model.layers)} 层）")

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

    # 获取 embeddings
    with torch.no_grad():
        hidden_states_full = embedding_layer(full_seq.unsqueeze(0))  # [1, seq_len, hidden_dim]

    # ✅ 关键修复：RoPE 只计算一次，所有层共享（和模型实际实现一致）
    with torch.no_grad():
        position_embeddings_full = rotary_emb(hidden_states_full, position_ids=full_position_ids.unsqueeze(0))

    # 逐层前向传播
    print(f"\n逐层前向传播（场景 1）:")
    layer_outputs_full = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(decoder_layers):
            # Decoder layer forward（使用预计算的 position_embeddings）
            hidden_states_full = layer(
                hidden_states_full,
                attention_mask=block_mask_full,
                position_ids=full_position_ids.unsqueeze(0),
                position_embeddings=position_embeddings_full,  # ✅ 所有层共享
            )[0]

            # 提取 M1 位置
            m1_start = prompt_len + mask_len + block_len
            m1_end = m1_start + mask_len
            m1_output = hidden_states_full[0, m1_start:m1_end, :]

            layer_outputs_full.append(m1_output)
            print(f"  Layer {layer_idx}: M1 output 范数 = {m1_output.norm(dim=-1).tolist()}")

    # ========== 场景 2: 去掉 M0 的序列 [P][R0][M1] ==========
    print(f"\n场景 2: 去掉 M0 的序列 [P][R0][M1]（M1 位置对齐）")

    # 构造 input_ids（Prompt + R0 + M1，去掉 M0）
    short_seq = torch.cat([
        prompt_tokens,      # [0-4]: Prompt
        block_tokens,       # [5-8]: R0
        mask_tokens,        # [9-11]: M1
    ], dim=0)

    short_seq_len = len(short_seq)
    print(f"  序列长度: {short_seq_len}")
    print(f"  结构: [0-{prompt_len-1}] Prompt, [{prompt_len}-{prompt_len+block_len-1}] R0, [{prompt_len+block_len}-{short_seq_len-1}] M1")

    # 构造 position_ids（M1 的位置要和完整序列中的 M1 一致）
    short_position_ids = torch.cat([
        torch.arange(0, prompt_len, device=device),                                           # Prompt: 0-4
        torch.arange(prompt_len, prompt_len + block_len, device=device),                      # R0: 5-8
        torch.arange(prompt_len + block_len, prompt_len + block_len + mask_len, device=device), # M1: 9-11（和完整序列一致！）
    ], dim=0)

    print(f"  Position IDs: {short_position_ids.tolist()}")

    # 构造 block_info
    block_info_short = [
        ('real', block_len),
        ('mask', mask_len),
    ]

    # 创建 BlockMask
    block_mask_short = create_simple_block_mask(
        block_info=block_info_short,
        prompt_len=prompt_len,
        seq_len=short_seq_len,
        device=device,
    )

    # 获取 embeddings
    with torch.no_grad():
        hidden_states_short = embedding_layer(short_seq.unsqueeze(0))

    # ✅ 关键修复：RoPE 只计算一次，所有层共享（和模型实际实现一致）
    with torch.no_grad():
        position_embeddings_short = rotary_emb(hidden_states_short, position_ids=short_position_ids.unsqueeze(0))

    # 逐层前向传播
    print(f"\n逐层前向传播（场景 2）:")
    layer_outputs_short = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(decoder_layers):
            # Decoder layer forward（使用预计算的 position_embeddings）
            hidden_states_short = layer(
                hidden_states_short,
                attention_mask=block_mask_short,
                position_ids=short_position_ids.unsqueeze(0),
                position_embeddings=position_embeddings_short,  # ✅ 所有层共享
            )[0]

            # 提取 M1 位置
            m1_start_short = prompt_len + block_len
            m1_end_short = m1_start_short + mask_len
            m1_output = hidden_states_short[0, m1_start_short:m1_end_short, :]

            layer_outputs_short.append(m1_output)
            print(f"  Layer {layer_idx}: M1 output 范数 = {m1_output.norm(dim=-1).tolist()}")

    # ========== 验证：逐层对比差异，检查误差累积 ==========
    print(f"\n" + "=" * 80)
    print(f"逐层验证结果")
    print(f"=" * 80)

    # 逐层计算差异
    layer_diffs = []
    # 对于多层计算，使用稍宽松的阈值（考虑浮点累积误差）
    threshold = 1e-4
    all_passed = True

    print(f"\n每层的 M1 输出差异:")
    print(f"{'Layer':<8} {'最大差异':<15} {'平均差异':<15} {'状态':<10}")
    print(f"-" * 60)

    for layer_idx in range(num_layers):
        diff = (layer_outputs_full[layer_idx] - layer_outputs_short[layer_idx]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        layer_diffs.append((max_diff, mean_diff))

        status = "✅ PASS" if max_diff < threshold else "❌ FAIL"
        if max_diff >= threshold:
            all_passed = False

        print(f"Layer {layer_idx:<2} {max_diff:<15.6e} {mean_diff:<15.6e} {status:<10}")

    # 分析误差趋势
    print(f"\n误差趋势分析:")
    max_diffs = [d[0] for d in layer_diffs]
    mean_diffs = [d[1] for d in layer_diffs]

    print(f"  第一层最大差异: {max_diffs[0]:.6e}")
    print(f"  最后层最大差异: {max_diffs[-1]:.6e}")
    print(f"  差异增长倍数: {max_diffs[-1] / max_diffs[0]:.2f}x")

    # 检查是否有明显的误差累积
    if max_diffs[-1] > max_diffs[0] * 10:
        print(f"  ⚠️  警告：误差有显著累积趋势！")
    elif max_diffs[-1] > max_diffs[0] * 2:
        print(f"  ⚠️  注意：误差有轻微累积")
    else:
        print(f"  ✅ 误差没有明显累积")

    # 总体验证
    print(f"\n" + "=" * 80)
    print(f"总体验证结果")
    print(f"=" * 80)

    if all_passed:
        print(f"\n✅ 所有 {num_layers} 层验证通过！")
        print(f"   M1 的计算在两个场景中完全一致（所有层差异 < {threshold}）")
        print(f"   这证明:")
        print(f"     1. attention mask 正确隔离了 M1 和 M0 之间的信息流")
        print(f"     2. position embedding 正确处理了位置对齐")
        print(f"     3. 误差不会随层数累积")
        return True
    else:
        print(f"\n❌ 验证失败！")
        print(f"   部分层的 M1 计算在两个场景中不一致（差异 >= {threshold}）")
        print(f"   失败的层: {[i for i in range(num_layers) if layer_diffs[i][0] >= threshold]}")
        return False


if __name__ == "__main__":
    try:
        print("\n" + "=" * 100)
        print("Mask Context Isolation Test (Multi-Layer)")
        print("=" * 100 + "\n")

        # 运行测试（测试所有层，传入 None 或负数表示测试所有层）
        result = test_mask_isolation_with_multiple_layers(num_layers=None)

        if result:
            print("\n" + "=" * 100)
            print("✅ 所有层验证通过！")
            print("=" * 100 + "\n")
            exit(0)
        else:
            print("\n" + "=" * 100)
            print("❌ 验证失败！")
            print("=" * 100 + "\n")
            exit(1)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
