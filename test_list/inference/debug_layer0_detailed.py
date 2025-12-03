#!/usr/bin/env python3
"""
详细拆解Layer 0的计算过程，找出[P][M₁]和[P][M₁][R₁]差异的根源

逐步检查：
1. Embedding
2. Position embeddings (RoPE)
3. Attention (Q, K, V, attention weights, attention output)
4. MLP
5. LayerNorm
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


def detailed_layer0_analysis(model, input_ids, position_ids, block_info, prompt_len, device):
    """
    详细分析Layer 0的每一步计算

    Returns:
        各个中间步骤的M₁位置的tensor
    """
    m1_pos = prompt_len
    results = {}

    # ========== 1. Embedding ==========
    print("\n" + "="*80)
    print("1️⃣  Embedding层")
    print("="*80)

    inputs_embeds = model.model.embed_tokens(input_ids)  # [seq_len, hidden_size]
    m1_embedding = inputs_embeds[m1_pos]
    results['embedding'] = m1_embedding

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Embeddings shape: {inputs_embeds.shape}")
    print(f"M₁ embedding (前5维): {m1_embedding[:5]}")
    print(f"M₁ embedding norm: {torch.norm(m1_embedding).item():.6f}")

    # ========== 2. Position IDs ==========
    print("\n" + "="*80)
    print("2️⃣  Position IDs")
    print("="*80)

    m1_position_id = position_ids[m1_pos]
    results['position_id'] = m1_position_id

    print(f"Position IDs: {position_ids.tolist()}")
    print(f"M₁ position ID: {m1_position_id.item()}")

    # ========== 3. Layer 0 详细分析 ==========
    print("\n" + "="*80)
    print("3️⃣  Layer 0 详细分析")
    print("="*80)

    layer0 = model.model.layers[0]

    # 3.1 Input LayerNorm
    print("\n--- 3.1 Input LayerNorm ---")
    normed_inputs = layer0.input_layernorm(inputs_embeds.unsqueeze(0))[0]  # [seq_len, hidden_size]
    m1_normed = normed_inputs[m1_pos]
    results['layer0_input_normed'] = m1_normed

    print(f"Normed inputs shape: {normed_inputs.shape}")
    print(f"M₁ normed (前5维): {m1_normed[:5]}")
    print(f"M₁ normed norm: {torch.norm(m1_normed).item():.6f}")

    # 3.2 Attention - Q, K, V projections
    print("\n--- 3.2 Attention Q/K/V projections ---")
    attn = layer0.self_attn

    # Get attention dimensions
    head_dim = attn.head_dim
    num_heads = attn.num_attention_heads
    num_kv_heads = attn.num_key_value_heads

    # Project to Q, K, V
    query_states = attn.q_proj(normed_inputs).view(-1, num_heads, head_dim).transpose(0, 1)  # [num_heads, seq_len, head_dim]
    key_states = attn.k_proj(normed_inputs).view(-1, num_kv_heads, head_dim).transpose(0, 1)
    value_states = attn.v_proj(normed_inputs).view(-1, num_kv_heads, head_dim).transpose(0, 1)

    m1_q = query_states[:, m1_pos, :]  # [num_heads, head_dim]
    m1_k = key_states[:, m1_pos, :]    # [num_kv_heads, head_dim]
    m1_v = value_states[:, m1_pos, :]

    results['layer0_q'] = m1_q
    results['layer0_k'] = m1_k
    results['layer0_v'] = m1_v

    print(f"Q shape: {query_states.shape}")
    print(f"K shape: {key_states.shape}")
    print(f"V shape: {value_states.shape}")
    print(f"M₁ Q[0] (前5维): {m1_q[0, :5]}")
    print(f"M₁ K[0] (前5维): {m1_k[0, :5]}")
    print(f"M₁ V[0] (前5维): {m1_v[0, :5]}")

    # 3.3 Forward through layer 0 to get output
    print("\n--- 3.3 Layer 0 Full Forward ---")

    # Run through the full layer
    with torch.no_grad():
        layer_output = layer0(
            inputs_embeds.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0)
        )[0][0]  # [seq_len, hidden_size]

    m1_layer_output = layer_output[m1_pos]
    results['layer0_output'] = m1_layer_output

    print(f"Layer 0 output shape: {layer_output.shape}")
    print(f"M₁ layer 0 output (前5维): {m1_layer_output[:5]}")
    print(f"M₁ layer 0 output norm: {torch.norm(m1_layer_output).item():.6f}")

    return results


def compare_two_configs(model, tokenizer, sample, device):
    """
    对比[P][M₁]和[P][M₁][R₁]两种配置的详细计算过程
    """
    print("="*80)
    print("详细对比：[P][M₁] vs [P][M₁][R₁]")
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

    print(f"\n序列长度: {input_ids_1.size(0)}")
    print(f"Block info: {block_info_1}")
    print(f"Input IDs: {input_ids_1.tolist()}")

    results_1 = detailed_layer0_analysis(
        model, input_ids_1, position_ids_1, block_info_1, prompt_len, device
    )

    # 配置2: [P][M₁][R₁]
    print("\n" + "#"*80)
    print("配置2: [P][M₁][R₁]")
    print("#"*80)

    truncate_pos_2 = prompt_len + 7
    input_ids_2 = input_ids_full[:truncate_pos_2]
    position_ids_2 = position_ids_full[:truncate_pos_2]
    block_info_2 = [('mask', 3), ('real', 4)]

    print(f"\n序列长度: {input_ids_2.size(0)}")
    print(f"Block info: {block_info_2}")
    print(f"Input IDs: {input_ids_2.tolist()}")

    results_2 = detailed_layer0_analysis(
        model, input_ids_2, position_ids_2, block_info_2, prompt_len, device
    )

    # ========== 对比分析 ==========
    print("\n" + "="*80)
    print("📊 对比分析")
    print("="*80)

    comparisons = [
        ('embedding', 'Embedding'),
        ('rope_cos', 'RoPE cos'),
        ('rope_sin', 'RoPE sin'),
        ('layer0_input_normed', 'Input LayerNorm'),
        ('layer0_q', 'Query (before RoPE)'),
        ('layer0_k', 'Key (before RoPE)'),
        ('layer0_v', 'Value'),
        ('layer0_q_rope', 'Query (after RoPE)'),
        ('layer0_k_rope', 'Key (after RoPE)'),
    ]

    print(f"\n{'步骤':^30} | {'L2距离':^15} | {'余弦相似度':^15} | {'是否相同':^10}")
    print("-"*80)

    for key, name in comparisons:
        if key not in results_1 or key not in results_2:
            continue

        t1 = results_1[key]
        t2 = results_2[key]

        # Flatten if multi-dimensional
        if t1.dim() > 1:
            t1 = t1.flatten()
            t2 = t2.flatten()

        l2_dist = torch.norm(t1 - t2).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            t1.unsqueeze(0), t2.unsqueeze(0)
        ).item()

        is_same = "✅" if l2_dist < 1e-5 else "❌"

        print(f"{name:^30} | {l2_dist:>12.6f}  | {cos_sim:>12.6f}  | {is_same:^10}")

    # ========== 关键发现 ==========
    print("\n" + "="*80)
    print("🔍 关键发现")
    print("="*80)

    # 检查embedding是否相同
    emb_diff = torch.norm(results_1['embedding'] - results_2['embedding']).item()
    if emb_diff < 1e-5:
        print("✅ Embedding层：M₁的embedding完全相同")
    else:
        print(f"❌ Embedding层：M₁的embedding不同！差异: {emb_diff:.6f}")
        print("   这不应该发生，因为M₁的input_id应该相同")
        return

    # 检查position embeddings
    cos_diff = torch.norm(results_1['rope_cos'] - results_2['rope_cos']).item()
    sin_diff = torch.norm(results_1['rope_sin'] - results_2['rope_sin']).item()

    if cos_diff < 1e-5 and sin_diff < 1e-5:
        print("✅ RoPE：M₁的position embeddings完全相同")
    else:
        print(f"❌ RoPE：M₁的position embeddings不同！")
        print(f"   cos差异: {cos_diff:.6f}")
        print(f"   sin差异: {sin_diff:.6f}")
        print(f"   position_id配置1: {position_ids_1[prompt_len].item()}")
        print(f"   position_id配置2: {position_ids_2[prompt_len].item()}")
        print("\n   ⚠️ 这说明position_ids不同导致了差异！")
        print("   虽然M₁看不到R₁，但position_id本身的差异影响了RoPE encoding")


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

    # 运行详细对比
    compare_two_configs(model, tokenizer, sample, device)


if __name__ == "__main__":
    main()
