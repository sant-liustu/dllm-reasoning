#!/usr/bin/env python3
"""
使用hooks详细拆解Layer 0的计算过程，找出[P][M₁]和[P][M₁][R₁]差异的根源

通过在模型关键位置注册hooks来捕获：
1. Embedding输出
2. Position IDs
3. Layer 0的各个中间步骤
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


def capture_layer0_details(model, input_ids, position_ids, block_info, prompt_len, device):
    """
    使用hooks捕获Layer 0的详细计算过程

    Returns:
        包含各步骤M₁位置tensor的字典
    """
    m1_pos = prompt_len
    captured = {}

    # Hook 1: Capture embedding output
    def hook_embedding(module, input, output):
        # output shape: [batch, seq_len, hidden_dim]
        captured['embedding'] = output[0, m1_pos].detach().clone()

    # Hook 2: Capture Layer 0 input (after input_layernorm)
    def hook_layer0_input_norm(module, input, output):
        captured['layer0_input_normed'] = output[0, m1_pos].detach().clone()

    # Hook 3: Capture Layer 0 Q/K/V projections
    def hook_q_proj(module, input, output):
        # output shape: [batch, seq_len, num_heads * head_dim]
        captured['layer0_q_proj'] = output[0, m1_pos].detach().clone()

    def hook_k_proj(module, input, output):
        captured['layer0_k_proj'] = output[0, m1_pos].detach().clone()

    def hook_v_proj(module, input, output):
        captured['layer0_v_proj'] = output[0, m1_pos].detach().clone()

    # Hook 4: Capture attention output (before o_proj)
    def hook_attn_output(module, input, output):
        # output is tuple: (attn_output, attn_weights)
        # attn_output shape: [batch, seq_len, hidden_dim]
        captured['layer0_attn_output'] = output[0][0, m1_pos].detach().clone()

    # Hook 5: Capture Layer 0 output
    def hook_layer0_output(module, input, output):
        # output is tuple: (hidden_states, ...)
        captured['layer0_output'] = output[0][0, m1_pos].detach().clone()

    # Register hooks
    layer0 = model.model.layers[0]
    hooks = []

    hooks.append(model.model.embed_tokens.register_forward_hook(hook_embedding))
    hooks.append(layer0.input_layernorm.register_forward_hook(hook_layer0_input_norm))
    hooks.append(layer0.self_attn.q_proj.register_forward_hook(hook_q_proj))
    hooks.append(layer0.self_attn.k_proj.register_forward_hook(hook_k_proj))
    hooks.append(layer0.self_attn.v_proj.register_forward_hook(hook_v_proj))
    hooks.append(layer0.self_attn.register_forward_hook(hook_attn_output))
    hooks.append(layer0.register_forward_hook(hook_layer0_output))

    # Capture position IDs
    captured['position_id'] = position_ids[m1_pos].item()

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

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return captured


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
    print(f"M₁ position: {prompt_len}")
    print(f"M₁ position_id: {position_ids_1[prompt_len].item()}")

    results_1 = capture_layer0_details(
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
    print(f"M₁ position: {prompt_len}")
    print(f"M₁ position_id: {position_ids_2[prompt_len].item()}")

    results_2 = capture_layer0_details(
        model, input_ids_2, position_ids_2, block_info_2, prompt_len, device
    )

    # ========== 对比分析 ==========
    print("\n" + "="*80)
    print("📊 对比分析")
    print("="*80)

    # 1. Position IDs对比
    print("\n1️⃣  Position IDs")
    print(f"  配置1 M₁ position_id: {results_1['position_id']}")
    print(f"  配置2 M₁ position_id: {results_2['position_id']}")
    if results_1['position_id'] == results_2['position_id']:
        print("  ✅ Position IDs相同")
    else:
        print(f"  ❌ Position IDs不同！差异: {abs(results_1['position_id'] - results_2['position_id'])}")
        print("  ⚠️ 这是RoPE编码不同的根源！")

    # 2. 其他中间步骤对比
    comparisons = [
        ('embedding', 'Embedding'),
        ('layer0_input_normed', 'Input LayerNorm'),
        ('layer0_q_proj', 'Q projection'),
        ('layer0_k_proj', 'K projection'),
        ('layer0_v_proj', 'V projection'),
        ('layer0_attn_output', 'Attention output'),
        ('layer0_output', 'Layer 0 output'),
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

    # 检查position IDs
    if results_1['position_id'] != results_2['position_id']:
        print(f"\n❌ Position IDs不同！")
        print(f"   配置1 [P][M₁]: position_id = {results_1['position_id']}")
        print(f"   配置2 [P][M₁][R₁]: position_id = {results_2['position_id']}")
        print(f"\n   ⚠️ 关键发现：虽然M₁看不到R₁（通过FlexAttention mask阻止），")
        print(f"   但position_id本身的差异导致RoPE编码不同！")
        print(f"   RoPE在Q/K投影后应用，position_id不同会导致Q/K的位置编码不同，")
        print(f"   进而影响attention计算，最终导致M₁的输出不同。")
    else:
        print("✅ Position IDs相同")

        # 进一步分析attention output差异
        attn_diff = torch.norm(results_1['layer0_attn_output'] - results_2['layer0_attn_output']).item()
        if attn_diff > 1e-5:
            print(f"\n❌ Attention Output不同！差异: {attn_diff:.6f}")
            print(f"\n   ⚠️ 关键发现：虽然Position IDs相同，Q/K/V projection也相同，")
            print(f"   但Attention输出却不同！")
            print(f"\n   原因分析：")
            print(f"   1. 配置1 [P][M₁]: 序列长度280，M₁的Q可以attend到的K/V来自280个位置")
            print(f"   2. 配置2 [P][M₁][R₁]: 序列长度284，M₁的Q可以attend到的K/V来自284个位置")
            print(f"\n   虽然FlexAttention的BlockMask阻止M₁直接attend到R₁的token，")
            print(f"   但是K/V矩阵的**形状不同**（280 vs 284），这会影响attention计算！")
            print(f"\n   更深层的原因：")
            print(f"   - FlexAttention mask是动态的，取决于block_info和序列长度")
            print(f"   - [P][M₁]的mask: 只有prompt+mask的280个位置")
            print(f"   - [P][M₁][R₁]的mask: 有prompt+mask+real的284个位置")
            print(f"   - 即使M₁不能直接attend到R₁，但mask的整体结构不同，")
            print(f"     导致attention pattern发生变化（例如attention weights的归一化）")
        else:
            print("   但attention output相同，差异可能在MLP或residual中...")


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
