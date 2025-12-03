#!/usr/bin/env python3
"""
捕获完整的attention计算过程

目标：
1. 捕获完整的K/V矩阵 (所有位置)
2. 手动计算M₁的attention scores (Q @ K^T)
3. 对比[P][M₁]和[P][M₁][R₁]两种配置
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


def capture_attention_details(model, input_ids, position_ids, block_info, prompt_len, device):
    """
    捕获Layer 0的完整attention计算细节
    """
    m1_pos = prompt_len
    captured = {}

    # Hook to capture Q/K/V after projection and RoPE
    def hook_forward(module, args, kwargs, output):
        """Hook self_attn.forward"""
        # Attention forward signature: forward(hidden_states, position_embeddings, attention_mask, ...)
        # In PyTorch hooks with_kwargs=True, positional args are in args, keyword args in kwargs
        if len(args) >= 3:
            hidden_states = args[0]
            position_embeddings = args[1]
            attention_mask = args[2]
        else:
            # kwargs passed
            hidden_states = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
            position_embeddings = kwargs.get('position_embeddings', args[1] if len(args) > 1 else None)
            attention_mask = kwargs.get('attention_mask', args[2] if len(args) > 2 else None)

        bsz, q_len = hidden_states.shape[:-1]
        hidden_shape = (bsz, q_len, -1, module.head_dim)

        # Q/K/V projection
        query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        # 直接从model的模块中获取apply_rotary_pos_emb
        import sys
        modeling_module = sys.modules.get(module.__class__.__module__)
        if modeling_module and hasattr(modeling_module, 'apply_rotary_pos_emb'):
            apply_rotary_pos_emb = modeling_module.apply_rotary_pos_emb
        else:
            # Fallback: 从transformers_modules中导入
            import importlib
            modeling_module = importlib.import_module('transformers_modules.huggingface.modeling_dllm')
            apply_rotary_pos_emb = modeling_module.apply_rotary_pos_emb

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # 保存完整的Q/K/V矩阵 [B, num_heads, seq_len, head_dim]
        captured['query_full'] = query_states[0].detach().clone()  # [num_heads, seq_len, head_dim]
        captured['key_full'] = key_states[0].detach().clone()
        captured['value_full'] = value_states[0].detach().clone()
        captured['seq_len'] = q_len

        # 保存attention_mask (BlockMask)
        captured['attention_mask'] = attention_mask

        return output

    # Register hook using forward_hook with_kwargs
    layer0_attn = model.model.layers[0].self_attn
    hook_handle = layer0_attn.register_forward_hook(hook_forward, with_kwargs=True)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            block_info=[block_info],
            prompt_len=[prompt_len],
            seq_lens=[input_ids.size(0)],
            use_cache=False,
        )

    # Remove hook
    hook_handle.remove()

    return captured


def analyze_attention_scores(captured_1, captured_2, m1_pos):
    """
    分析M₁位置的attention scores

    Args:
        captured_1: [P][M₁]配置的捕获数据
        captured_2: [P][M₁][R₁]配置的捕获数据
        m1_pos: M₁的位置
    """
    print("\n" + "="*80)
    print("📊 Attention Scores详细分析")
    print("="*80)

    # 提取M₁位置的Q
    q1 = captured_1['query_full'][:, m1_pos, :]  # [num_heads, head_dim]
    q2 = captured_2['query_full'][:, m1_pos, :]

    # 验证Q相同
    q_diff = torch.norm(q1 - q2).item()
    print(f"\nM₁的Query向量:")
    print(f"  L2距离: {q_diff:.10f}")
    print(f"  {'✅ 完全相同' if q_diff < 1e-5 else '❌ 不同'}")

    if q_diff >= 1e-5:
        print(f"  ⚠️ Query不同，后续分析无意义")
        return

    # 提取完整的K矩阵
    k1 = captured_1['key_full']  # [num_heads, seq_len1, head_dim]
    k2 = captured_2['key_full']  # [num_heads, seq_len2, head_dim]

    seq_len1 = captured_1['seq_len']
    seq_len2 = captured_2['seq_len']

    print(f"\nK矩阵形状:")
    print(f"  配置1 [P][M₁]: {k1.shape} (seq_len={seq_len1})")
    print(f"  配置2 [P][M₁][R₁]: {k2.shape} (seq_len={seq_len2})")

    # 对比M₁可见位置的K (0 to m1_pos)
    visible_len = m1_pos + 1  # Prompt + M₁自己
    k1_visible = k1[:, :visible_len, :]  # [num_heads, visible_len, head_dim]
    k2_visible = k2[:, :visible_len, :]

    k_visible_diff = torch.norm(k1_visible - k2_visible).item()
    print(f"\nM₁可见位置的K (0-{m1_pos}):")
    print(f"  L2距离: {k_visible_diff:.10f}")
    print(f"  {'✅ 完全相同' if k_visible_diff < 1e-5 else '❌ 不同'}")

    if k_visible_diff >= 1e-5:
        print(f"  ⚠️ 可见K不同，这不应该发生！")
        return

    # 计算attention scores: Q @ K^T
    # q1/q2: [num_q_heads, head_dim]
    # k1_visible: [num_kv_heads, visible_len, head_dim]

    # GQA: 需要重复K/V来匹配Q的head数量
    num_q_heads = q1.shape[0]
    num_kv_heads = k1_visible.shape[0]
    num_groups = num_q_heads // num_kv_heads

    # Repeat K/V: [num_kv_heads, visible_len, head_dim] -> [num_q_heads, visible_len, head_dim]
    k1_visible_repeated = k1_visible.repeat_interleave(num_groups, dim=0)
    k2_visible_repeated = k2_visible.repeat_interleave(num_groups, dim=0)

    scaling = 1.0 / (q1.shape[-1] ** 0.5)  # 1/sqrt(head_dim)

    # [num_q_heads, head_dim] @ [num_q_heads, head_dim, visible_len] -> [num_q_heads, visible_len]
    scores1 = torch.bmm(
        q1.unsqueeze(1),  # [num_q_heads, 1, head_dim]
        k1_visible_repeated.transpose(1, 2)  # [num_q_heads, head_dim, visible_len]
    ).squeeze(1) * scaling  # [num_q_heads, visible_len]

    scores2 = torch.bmm(
        q2.unsqueeze(1),
        k2_visible_repeated.transpose(1, 2)
    ).squeeze(1) * scaling

    scores_diff = torch.norm(scores1 - scores2).item()
    print(f"\nAttention Scores (Q @ K^T, scaled):")
    print(f"  配置1: {scores1.shape}")
    print(f"  配置2: {scores2.shape}")
    print(f"  L2距离: {scores_diff:.10f}")
    print(f"  {'✅ 完全相同' if scores_diff < 1e-5 else '❌ 不同'}")

    if scores_diff >= 1e-5:
        print(f"  ⚠️ Scores不同，这不应该发生（Q和K都相同）！")
        # 打印逐位置差异
        print(f"\n  逐位置检查 (第一个head):")
        head0_scores1 = scores1[0]  # [visible_len]
        head0_scores2 = scores2[0]
        print(f"  {'位置':^8} | {'Scores1':^15} | {'Scores2':^15} | {'差异':^15}")
        print("-"*60)
        for pos in range(min(10, visible_len)):
            s1 = head0_scores1[pos].item()
            s2 = head0_scores2[pos].item()
            diff = abs(s1 - s2)
            print(f"  {pos:^8} | {s1:>12.6f}  | {s2:>12.6f}  | {diff:>12.6f}")
        return

    # 计算softmax后的attention weights
    weights1 = torch.softmax(scores1, dim=-1)  # [num_heads, visible_len]
    weights2 = torch.softmax(scores2, dim=-1)

    weights_diff = torch.norm(weights1 - weights2).item()
    print(f"\nAttention Weights (after softmax):")
    print(f"  L2距离: {weights_diff:.10f}")
    print(f"  {'✅ 完全相同' if weights_diff < 1e-5 else '❌ 不同'}")

    # 对比V矩阵
    v1_visible = captured_1['value_full'][:, :visible_len, :]  # [num_heads, visible_len, head_dim]
    v2_visible = captured_2['value_full'][:, :visible_len, :]

    v_visible_diff = torch.norm(v1_visible - v2_visible).item()
    print(f"\nM₁可见位置的V (0-{m1_pos}):")
    print(f"  L2距离: {v_visible_diff:.10f}")
    print(f"  {'✅ 完全相同' if v_visible_diff < 1e-5 else '❌ 不同'}")

    if v_visible_diff >= 1e-5:
        print(f"  ⚠️ 可见V不同，这不应该发生！")
        return

    # 计算最终的attention output: weights @ V
    # weights: [num_q_heads, visible_len]
    # v_visible: [num_kv_heads, visible_len, head_dim]

    # Repeat V: [num_kv_heads, visible_len, head_dim] -> [num_q_heads, visible_len, head_dim]
    v1_visible_repeated = v1_visible.repeat_interleave(num_groups, dim=0)
    v2_visible_repeated = v2_visible.repeat_interleave(num_groups, dim=0)

    output1 = torch.bmm(
        weights1.unsqueeze(1),  # [num_q_heads, 1, visible_len]
        v1_visible_repeated  # [num_q_heads, visible_len, head_dim]
    ).squeeze(1)  # [num_q_heads, head_dim]

    output2 = torch.bmm(
        weights2.unsqueeze(1),
        v2_visible_repeated
    ).squeeze(1)

    output_diff = torch.norm(output1 - output2).item()
    print(f"\nAttention Output (weights @ V):")
    print(f"  L2距离: {output_diff:.10f}")
    print(f"  {'✅ 完全相同' if output_diff < 1e-5 else '❌ 不同'}")

    print("\n" + "="*80)
    print("🔍 结论")
    print("="*80)

    if scores_diff < 1e-5 and weights_diff < 1e-5 and output_diff < 1e-5:
        print("\n✅ 手动计算的attention完全相同！")
        print("\n这说明：")
        print("  - M₁可见位置的Q/K/V确实相同")
        print("  - 基于这些可见位置的attention计算结果相同")
        print("\n⚠️ 但实际模型输出却不同！")
        print("\n可能的原因：")
        print("  1. FlexAttention实际上并没有完全隔离不可见位置")
        print("  2. 或者FlexAttention的实现有某些数值稳定性问题")
        print("  3. 需要检查FlexAttention的实际行为")
    else:
        print("\n❌ 发现差异！")
        if scores_diff >= 1e-5:
            print(f"  - Attention scores不同 (L2={scores_diff:.6f})")
        if weights_diff >= 1e-5:
            print(f"  - Attention weights不同 (L2={weights_diff:.6f})")
        if output_diff >= 1e-5:
            print(f"  - Attention output不同 (L2={output_diff:.6f})")


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

    prompt_len = sample['prompt_len']
    input_ids_full = sample['input_ids'].to(device)
    position_ids_full = sample['position_ids'].to(device)

    print(f"\n样本信息:")
    print(f"  Prompt长度: {prompt_len}")
    print(f"  总序列长度: {input_ids_full.shape[0]}")
    print(f"  M₁位置: {prompt_len}")

    # 配置1: [P][M₁]
    print("\n" + "#"*80)
    print("配置1: [P][M₁]")
    print("#"*80)

    truncate_pos_1 = prompt_len + 3
    input_ids_1 = input_ids_full[:truncate_pos_1]
    position_ids_1 = position_ids_full[:truncate_pos_1]
    block_info_1 = [('mask', 3)]

    captured_1 = capture_attention_details(
        model, input_ids_1, position_ids_1, block_info_1, prompt_len, device
    )

    print(f"序列长度: {captured_1['seq_len']}")
    print(f"Q shape: {captured_1['query_full'].shape}")
    print(f"K shape: {captured_1['key_full'].shape}")
    print(f"V shape: {captured_1['value_full'].shape}")

    # 配置2: [P][M₁][R₁]
    print("\n" + "#"*80)
    print("配置2: [P][M₁][R₁]")
    print("#"*80)

    truncate_pos_2 = prompt_len + 7
    input_ids_2 = input_ids_full[:truncate_pos_2]
    position_ids_2 = position_ids_full[:truncate_pos_2]
    block_info_2 = [('mask', 3), ('real', 4)]

    captured_2 = capture_attention_details(
        model, input_ids_2, position_ids_2, block_info_2, prompt_len, device
    )

    print(f"序列长度: {captured_2['seq_len']}")
    print(f"Q shape: {captured_2['query_full'].shape}")
    print(f"K shape: {captured_2['key_full'].shape}")
    print(f"V shape: {captured_2['value_full'].shape}")

    # 分析attention scores
    analyze_attention_scores(captured_1, captured_2, prompt_len)


if __name__ == "__main__":
    main()
