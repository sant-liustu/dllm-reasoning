#!/usr/bin/env python3
"""
深入debug attention计算的每一步

Hook捕获：
1. Q/K/V projection之后的值
2. RoPE应用之后的Q/K
3. Attention scores (Q @ K^T)
4. Attention weights (after softmax)
5. Attention output (weights @ V)
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset


def patch_attention_forward(model, prompt_len, captured_dict):
    """
    Patch Layer 0的attention forward函数来捕获中间计算结果
    """
    layer0_attn = model.model.layers[0].self_attn
    original_forward = layer0_attn.forward
    m1_pos = prompt_len

    def patched_forward(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs
    ):
        """Patched forward函数"""
        # 获取基本信息
        bsz, q_len = hidden_states.shape[:-1]
        hidden_shape = (bsz, q_len, -1, layer0_attn.head_dim)

        # ===== Step 1: Q/K/V Projection =====
        query_states = layer0_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = layer0_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = layer0_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        captured_dict['q_before_rope'] = query_states[0, :, m1_pos, :].detach().clone()
        captured_dict['k_before_rope'] = key_states[0, :, m1_pos, :].detach().clone()
        captured_dict['v'] = value_states[0, :, m1_pos, :].detach().clone()

        # ===== Step 2: Apply RoPE =====
        # Import from the cached transformers module
        import importlib
        modeling_dllm = importlib.import_module('transformers_modules.huggingface.modeling_dllm')
        apply_rotary_pos_emb = modeling_dllm.apply_rotary_pos_emb

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        captured_dict['q_after_rope'] = query_states[0, :, m1_pos, :].detach().clone()
        captured_dict['k_after_rope'] = key_states[0, :, m1_pos, :].detach().clone()

        # ===== Step 3: Attention Computation =====
        if layer0_attn.training:
            # Use FlexAttention
            from dllm_reasoning.checkpoints.interleaved_sft.global_step_17172.huggingface.modeling_dllm import fused_flex_attention

            # 保存K/V的完整矩阵（用于分析）
            captured_dict['k_full_shape'] = key_states.shape
            captured_dict['v_full_shape'] = value_states.shape

            attn_output, attn_weights = fused_flex_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                enable_gqa=True,
                scale=layer0_attn.scaling,
                return_lse=True
            )

            from einops import rearrange
            attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')

            captured_dict['attn_output_before_o_proj'] = attn_output[0, m1_pos, :].detach().clone()
            if attn_weights is not None:
                captured_dict['attn_weights'] = attn_weights[0, :, m1_pos, :].detach().clone()
        else:
            raise NotImplementedError("Only training mode supported")

        # ===== Step 4: Output Projection =====
        attn_output = layer0_attn.o_proj(attn_output)
        captured_dict['attn_output_after_o_proj'] = attn_output[0, m1_pos, :].detach().clone()

        return attn_output, attn_weights

    # Patch the forward method
    layer0_attn.forward = patched_forward
    return original_forward


def analyze_two_configs(model, tokenizer, sample, device):
    """
    对比[P][M₁]和[P][M₁][R₁]两种配置的attention计算过程
    """
    print("="*80)
    print("详细Attention计算对比")
    print("="*80)

    prompt_len = sample['prompt_len']
    input_ids_full = sample['input_ids'].to(device)
    position_ids_full = sample['position_ids'].to(device)

    # 配置1: [P][M₁]
    print("\n" + "#"*80)
    print("配置1: [P][M₁]")
    print("#"*80)

    truncate_pos_1 = prompt_len + 3
    input_ids_1 = input_ids_full[:truncate_pos_1]
    position_ids_1 = position_ids_full[:truncate_pos_1]
    block_info_1 = [('mask', 3)]

    captured_1 = {}
    original_forward = patch_attention_forward(model, prompt_len, captured_1)

    with torch.no_grad():
        outputs_1 = model(
            input_ids_1.unsqueeze(0),
            position_ids=position_ids_1.unsqueeze(0),
            block_info=[block_info_1],
            prompt_len=[prompt_len],
            seq_lens=[input_ids_1.size(0)],
            use_cache=False,
        )

    # Restore original forward
    model.model.layers[0].self_attn.forward = original_forward

    print(f"序列长度: {input_ids_1.size(0)}")
    print(f"K shape: {captured_1['k_full_shape']}")
    print(f"V shape: {captured_1['v_full_shape']}")

    # 配置2: [P][M₁][R₁]
    print("\n" + "#"*80)
    print("配置2: [P][M₁][R₁]")
    print("#"*80)

    truncate_pos_2 = prompt_len + 7
    input_ids_2 = input_ids_full[:truncate_pos_2]
    position_ids_2 = position_ids_full[:truncate_pos_2]
    block_info_2 = [('mask', 3), ('real', 4)]

    captured_2 = {}
    original_forward = patch_attention_forward(model, prompt_len, captured_2)

    with torch.no_grad():
        outputs_2 = model(
            input_ids_2.unsqueeze(0),
            position_ids=position_ids_2.unsqueeze(0),
            block_info=[block_info_2],
            prompt_len=[prompt_len],
            seq_lens=[input_ids_2.size(0)],
            use_cache=False,
        )

    # Restore original forward
    model.model.layers[0].self_attn.forward = original_forward

    print(f"序列长度: {input_ids_2.size(0)}")
    print(f"K shape: {captured_2['k_full_shape']}")
    print(f"V shape: {captured_2['v_full_shape']}")

    # ========== 对比分析 ==========
    print("\n" + "="*80)
    print("📊 逐步对比分析（M₁位置）")
    print("="*80)

    comparisons = [
        ('q_before_rope', 'Q (before RoPE)'),
        ('k_before_rope', 'K (before RoPE)'),
        ('v', 'V'),
        ('q_after_rope', 'Q (after RoPE)'),
        ('k_after_rope', 'K (after RoPE)'),
        ('attn_output_before_o_proj', 'Attention output (before o_proj)'),
        ('attn_output_after_o_proj', 'Attention output (after o_proj)'),
    ]

    print(f"\n{'步骤':^40} | {'L2距离':^15} | {'余弦相似度':^15} | {'是否相同':^10}")
    print("-"*90)

    for key, name in comparisons:
        if key not in captured_1 or key not in captured_2:
            continue

        t1 = captured_1[key]
        t2 = captured_2[key]

        # Flatten
        if t1.dim() > 1:
            t1 = t1.flatten()
            t2 = t2.flatten()

        l2_dist = torch.norm(t1 - t2).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            t1.unsqueeze(0), t2.unsqueeze(0)
        ).item()

        is_same = "✅" if l2_dist < 1e-5 else "❌"

        print(f"{name:^40} | {l2_dist:>12.6f}  | {cos_sim:>12.6f}  | {is_same:^10}")

    # ========== 关键分析 ==========
    print("\n" + "="*80)
    print("🔍 关键发现")
    print("="*80)

    # 检查Q/K/V before RoPE
    q_before_diff = torch.norm(captured_1['q_before_rope'].flatten() - captured_2['q_before_rope'].flatten()).item()
    k_before_diff = torch.norm(captured_1['k_before_rope'].flatten() - captured_2['k_before_rope'].flatten()).item()
    v_diff = torch.norm(captured_1['v'].flatten() - captured_2['v'].flatten()).item()

    if q_before_diff < 1e-5 and k_before_diff < 1e-5 and v_diff < 1e-5:
        print("✅ Q/K/V projection完全相同（RoPE之前）")
    else:
        print(f"❌ Q/K/V projection不同！这不应该发生...")
        return

    # 检查RoPE之后
    q_after_diff = torch.norm(captured_1['q_after_rope'].flatten() - captured_2['q_after_rope'].flatten()).item()
    k_after_diff = torch.norm(captured_1['k_after_rope'].flatten() - captured_2['k_after_rope'].flatten()).item()

    if q_after_diff < 1e-5 and k_after_diff < 1e-5:
        print("✅ RoPE应用后Q/K完全相同")
        print("   这说明M₁位置的position_id确实相同")
    else:
        print(f"❌ RoPE应用后Q/K不同！")
        print(f"   Q差异: {q_after_diff:.6f}")
        print(f"   K差异: {k_after_diff:.6f}")
        print("   这说明position_id有问题")
        return

    # 检查attention output
    attn_before_diff = torch.norm(
        captured_1['attn_output_before_o_proj'].flatten() -
        captured_2['attn_output_before_o_proj'].flatten()
    ).item()

    if attn_before_diff > 1e-5:
        print(f"\n❌ Attention output不同！差异: {attn_before_diff:.6f}")
        print("\n⚠️ 关键矛盾：")
        print(f"   1. M₁的Q/K/V完全相同（包括RoPE后）")
        print(f"   2. 但是Attention output却不同！")
        print(f"\n可能的原因：")
        print(f"   A. K/V矩阵的**其他位置**不同")
        print(f"      - 配置1: K/V shape = {captured_1['k_full_shape']}")
        print(f"      - 配置2: K/V shape = {captured_2['k_full_shape']}")
        print(f"      - 虽然M₁位置的K/V相同，但其他位置可能不同")
        print(f"\n   B. Attention mask的影响")
        print(f"      - 虽然理论上mask在softmax前应用（我们验证过）")
        print(f"      - 但实际的K/V矩阵大小不同，可能影响FlexAttention的底层实现")
        print(f"\n   C. 需要检查M₁**可见**的K/V位置")
        print(f"      - M₁应该只能看到Prompt + M₁自己")
        print(f"      - 我们需要验证这些位置的K/V是否真的相同")

        # 额外检查：检查前278个K/V是否相同（Prompt + M₁）
        print(f"\n🔎 额外检查：前278个位置的K/V是否相同？")
        # 这需要在forward中保存完整的K/V矩阵


def main():
    MODEL_PATH = "dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
    DATA_PATH = "data/openr1.parquet"

    print("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    model.train()

    print("加载数据集...")
    dataset = InterleavedSFTDataset(
        parquet_files=DATA_PATH,
        tokenizer=tokenizer,
        prompt_key="prompt",
        response_key="target",
        block_size=4,
        max_length=6000,
        truncation="right",
    )

    sample = dataset[0]

    print(f"\n样本信息:")
    print(f"  Prompt长度: {sample['prompt_len']}")
    print(f"  总序列长度: {sample['input_ids'].shape[0]}")

    # 运行详细分析
    analyze_two_configs(model, tokenizer, sample, device)


if __name__ == "__main__":
    main()
