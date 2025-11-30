#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Mask Token 的上下文隔离性（使用 float64 精度）

验证：使用更高精度（float64）来确认误差累积是否纯粹由浮点精度限制导致
如果 float64 下误差显著降低，则证明 float32 的误差是可接受的精度问题
"""

import sys
sys.path.insert(0, "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning")

import os
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

    mask_mod = create_mask_mod_from_block_info(
        block_info=block_info,
        prompt_len=prompt_len,
        seq_len=seq_len,
    )

    mask_2d = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for q in range(seq_len):
        for kv in range(seq_len):
            mask_2d[q, kv] = mask_mod(0, 0, q, kv)

    mask_2d = mask_2d.unsqueeze(0)

    block_mask = create_block_mask(
        lambda b, h, q_idx, kv_idx: mask_2d[b, q_idx, kv_idx],
        B=1,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    return block_mask


def test_isolation_fp64():
    """使用 float64 测试所有层"""
    print("=" * 100)
    print("Test: Mask Context Isolation with float64 precision")
    print("=" * 100)

    if not FLEX_ATTN_AVAILABLE:
        print("⚠️  跳过测试（FlexAttention 不可用）\n")
        return False

    print("\n加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 使用 float64
    model = DLLMForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float64,  # ✅ 使用 float64
        trust_remote_code=True,
    ).to(device)

    print(f"✓ 模型加载成功（设备: {device}, dtype: float64）")

    embedding_layer = model.model.embed_tokens
    rotary_emb = model.model.rotary_emb
    decoder_layers = model.model.layers
    num_layers = len(decoder_layers)

    for layer in decoder_layers:
        layer.train()

    print(f"将测试 {num_layers} 个 decoder 层")

    # 构造测试数据
    prompt_len = 5
    mask_len = 3
    block_len = 4

    prompt_tokens = torch.randint(1000, 5000, (prompt_len,), device=device)
    mask_token_id = tokenizer.eos_token_id
    mask_tokens = torch.full((mask_len,), mask_token_id, device=device)
    block_tokens = torch.randint(5000, 10000, (block_len,), device=device)

    print(f"\n测试设置:")
    print(f"  Prompt 长度: {prompt_len}")
    print(f"  Mask 长度: {mask_len}")
    print(f"  Block 长度: {block_len}")

    # ========== 场景 1: [P][M0][R0][M1] ==========
    print(f"\n场景 1: 完整序列 [P][M0][R0][M1]")

    full_seq = torch.cat([
        prompt_tokens,
        mask_tokens,
        block_tokens,
        mask_tokens,
    ], dim=0)

    full_seq_len = len(full_seq)
    print(f"  序列长度: {full_seq_len}")

    full_position_ids = torch.cat([
        torch.arange(0, prompt_len, device=device),
        torch.arange(prompt_len, prompt_len + mask_len, device=device),
        torch.arange(prompt_len, prompt_len + block_len, device=device),
        torch.arange(prompt_len + block_len, prompt_len + block_len + mask_len, device=device),
    ], dim=0)

    print(f"  Position IDs: {full_position_ids.tolist()}")

    block_info_full = [
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
        hidden_states_full = embedding_layer(full_seq.unsqueeze(0))
        position_embeddings_full = rotary_emb(hidden_states_full, position_ids=full_position_ids.unsqueeze(0))

    print(f"\n逐层前向传播（场景 1）:")
    layer_outputs_full = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(decoder_layers):
            hidden_states_full = layer(
                hidden_states_full,
                attention_mask=block_mask_full,
                position_ids=full_position_ids.unsqueeze(0),
                position_embeddings=position_embeddings_full,
            )[0]

            m1_start = prompt_len + mask_len + block_len
            m1_end = m1_start + mask_len
            m1_output = hidden_states_full[0, m1_start:m1_end, :]

            layer_outputs_full.append(m1_output)

            # 只打印部分层以减少输出
            if layer_idx < 3 or layer_idx >= num_layers - 3:
                print(f"  Layer {layer_idx}: M1 output 范数 = {m1_output.norm(dim=-1).tolist()}")
            elif layer_idx == 3:
                print(f"  ...")

    # ========== 场景 2: [P][R0][M1] ==========
    print(f"\n场景 2: 去掉 M0 的序列 [P][R0][M1]（M1 位置对齐）")

    short_seq = torch.cat([
        prompt_tokens,
        block_tokens,
        mask_tokens,
    ], dim=0)

    short_seq_len = len(short_seq)
    print(f"  序列长度: {short_seq_len}")

    short_position_ids = torch.cat([
        torch.arange(0, prompt_len, device=device),
        torch.arange(prompt_len, prompt_len + block_len, device=device),
        torch.arange(prompt_len + block_len, prompt_len + block_len + mask_len, device=device),
    ], dim=0)

    print(f"  Position IDs: {short_position_ids.tolist()}")

    block_info_short = [
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
        hidden_states_short = embedding_layer(short_seq.unsqueeze(0))
        position_embeddings_short = rotary_emb(hidden_states_short, position_ids=short_position_ids.unsqueeze(0))

    print(f"\n逐层前向传播（场景 2）:")
    layer_outputs_short = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(decoder_layers):
            hidden_states_short = layer(
                hidden_states_short,
                attention_mask=block_mask_short,
                position_ids=short_position_ids.unsqueeze(0),
                position_embeddings=position_embeddings_short,
            )[0]

            m1_start_short = prompt_len + block_len
            m1_end_short = m1_start_short + mask_len
            m1_output = hidden_states_short[0, m1_start_short:m1_end_short, :]

            layer_outputs_short.append(m1_output)

            if layer_idx < 3 or layer_idx >= num_layers - 3:
                print(f"  Layer {layer_idx}: M1 output 范数 = {m1_output.norm(dim=-1).tolist()}")
            elif layer_idx == 3:
                print(f"  ...")

    # ========== 验证 ==========
    print(f"\n" + "=" * 80)
    print(f"逐层验证结果（float64 精度）")
    print(f"=" * 80)

    layer_diffs = []
    threshold = 1e-10  # float64 下使用更严格的阈值
    all_passed = True

    print(f"\n每层的 M1 输出差异:")
    print(f"{'Layer':<8} {'最大差异':<20} {'平均差异':<20} {'状态':<10}")
    print(f"-" * 70)

    for layer_idx in range(num_layers):
        diff = (layer_outputs_full[layer_idx] - layer_outputs_short[layer_idx]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        layer_diffs.append((max_diff, mean_diff))

        status = "✅ PASS" if max_diff < threshold else "❌ FAIL"
        if max_diff >= threshold:
            all_passed = False

        print(f"Layer {layer_idx:<2} {max_diff:<20.12e} {mean_diff:<20.12e} {status:<10}")

    # 分析误差趋势
    print(f"\n误差趋势分析:")
    max_diffs = [d[0] for d in layer_diffs]

    print(f"  第一层最大差异: {max_diffs[0]:.12e}")
    print(f"  最后层最大差异: {max_diffs[-1]:.12e}")
    if max_diffs[0] > 0:
        print(f"  差异增长倍数: {max_diffs[-1] / max_diffs[0]:.2f}x")

    # 与 float32 对比
    print(f"\n与 float32 对比:")
    print(f"  float32 最后层误差: ~2.14e-04")
    print(f"  float64 最后层误差: {max_diffs[-1]:.12e}")
    if max_diffs[-1] > 0:
        improvement = 2.14e-04 / max_diffs[-1]
        print(f"  精度提升: {improvement:.1f}x")

    print(f"\n" + "=" * 80)
    print(f"总体验证结果")
    print(f"=" * 80)

    if all_passed:
        print(f"\n✅ 所有 {num_layers} 层验证通过（float64 精度）！")
        print(f"   这证明 float32 的误差确实是浮点精度限制导致的")
        return True
    else:
        print(f"\n⚠️  部分层失败（阈值 {threshold}）")
        print(f"   但如果误差远小于 float32，仍然说明是精度问题")
        failed_layers = [i for i in range(num_layers) if layer_diffs[i][0] >= threshold]
        print(f"   失败的层: {failed_layers}")
        return False


if __name__ == "__main__":
    try:
        result = test_isolation_fp64()
        exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
