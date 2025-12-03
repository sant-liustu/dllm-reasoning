#!/usr/bin/env python3
"""
验证FlexAttention的bug

已经证明:
1. 手动计算的attention output完全相同 ✅
2. 但FlexAttention的实际output不同 ❌

现在直接对比:
- 手动计算的attention output
- FlexAttention的实际output
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


def capture_both_outputs(model, input_ids, position_ids, block_info, prompt_len, device):
    """
    同时捕获:
    1. FlexAttention的实际输出
    2. 手动计算的attention output (基于可见位置的Q/K/V)
    """
    m1_pos = prompt_len
    captured = {}

    def hook_forward(module, args, kwargs, output):
        """Hook self_attn.forward"""
        if len(args) >= 3:
            hidden_states = args[0]
            position_embeddings = args[1]
            attention_mask = args[2]
        else:
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
        import sys
        modeling_module = sys.modules.get(module.__class__.__module__)
        if modeling_module and hasattr(modeling_module, 'apply_rotary_pos_emb'):
            apply_rotary_pos_emb = modeling_module.apply_rotary_pos_emb
        else:
            import importlib
            modeling_module = importlib.import_module('transformers_modules.huggingface.modeling_dllm')
            apply_rotary_pos_emb = modeling_module.apply_rotary_pos_emb

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # 提取M₁位置的Q
        q_m1 = query_states[0, :, m1_pos, :]  # [num_q_heads, head_dim]

        # 提取可见位置的K/V (0 to m1_pos)
        visible_len = m1_pos + 1
        k_visible = key_states[0, :, :visible_len, :]  # [num_kv_heads, visible_len, head_dim]
        v_visible = value_states[0, :, :visible_len, :]

        # GQA: 重复K/V
        num_q_heads = q_m1.shape[0]
        num_kv_heads = k_visible.shape[0]
        num_groups = num_q_heads // num_kv_heads

        k_visible_repeated = k_visible.repeat_interleave(num_groups, dim=0)
        v_visible_repeated = v_visible.repeat_interleave(num_groups, dim=0)

        # 手动计算attention
        scaling = 1.0 / (q_m1.shape[-1] ** 0.5)
        scores = torch.bmm(
            q_m1.unsqueeze(1),  # [num_q_heads, 1, head_dim]
            k_visible_repeated.transpose(1, 2)  # [num_q_heads, head_dim, visible_len]
        ).squeeze(1) * scaling  # [num_q_heads, visible_len]

        weights = torch.softmax(scores, dim=-1)  # [num_q_heads, visible_len]

        manual_output = torch.bmm(
            weights.unsqueeze(1),  # [num_q_heads, 1, visible_len]
            v_visible_repeated  # [num_q_heads, visible_len, head_dim]
        ).squeeze(1)  # [num_q_heads, head_dim]

        # 展平成 [num_q_heads * head_dim]
        manual_output_flat = manual_output.flatten()
        captured['manual_output'] = manual_output_flat.detach().clone()

        # FlexAttention的实际输出会在output中
        # output[0] 是 attn_output (after o_proj)
        # 我们需要捕获before o_proj的输出
        # 所以需要保存o_proj的输入

        return output

    # 同时hook o_proj的输入来捕获FlexAttention的实际输出(before o_proj)
    def hook_o_proj_input(module, input, output):
        # input[0] is attention output before o_proj
        # Shape: [batch, seq_len, num_q_heads * head_dim]
        attn_out_before_o = input[0][0, m1_pos, :].detach().clone()  # [num_q_heads * head_dim]
        captured['flex_output'] = attn_out_before_o

    layer0_attn = model.model.layers[0].self_attn
    hook_handle1 = layer0_attn.register_forward_hook(hook_forward, with_kwargs=True)
    hook_handle2 = layer0_attn.o_proj.register_forward_hook(hook_o_proj_input)

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
    hook_handle1.remove()
    hook_handle2.remove()

    return captured


def main():
    MODEL_PATH = "dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("🔍 FlexAttention Bug验证")
    print("="*80)
    print("\n目标: 对比手动计算和FlexAttention的实际输出")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    model.train()

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

    print(f"样本信息:")
    print(f"  Prompt长度: {prompt_len}")
    print(f"  M₁位置: {prompt_len}")

    # 配置1: [P][M₁]
    print("\n" + "#"*80)
    print("配置1: [P][M₁]")
    print("#"*80)

    truncate_pos_1 = prompt_len + 3
    input_ids_1 = input_ids_full[:truncate_pos_1]
    position_ids_1 = position_ids_full[:truncate_pos_1]
    block_info_1 = [('mask', 3)]

    captured_1 = capture_both_outputs(
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

    captured_2 = capture_both_outputs(
        model, input_ids_2, position_ids_2, block_info_2, prompt_len, device
    )

    # 对比
    print("\n" + "="*80)
    print("📊 对比结果 (M₁位置)")
    print("="*80)

    # 手动计算的输出
    manual_1 = captured_1['manual_output']
    manual_2 = captured_2['manual_output']
    manual_diff = torch.norm(manual_1 - manual_2).item()

    print(f"\n手动计算的Attention Output (基于可见Q/K/V):")
    print(f"  L2距离: {manual_diff:.10f}")
    print(f"  {'✅ 完全相同' if manual_diff < 1e-5 else '❌ 不同'}")

    # FlexAttention的实际输出
    flex_1 = captured_1['flex_output']
    flex_2 = captured_2['flex_output']
    flex_diff = torch.norm(flex_1 - flex_2).item()

    print(f"\nFlexAttention的实际输出:")
    print(f"  L2距离: {flex_diff:.10f}")
    print(f"  {'✅ 完全相同' if flex_diff < 1e-5 else '❌ 不同'}")

    # 对比配置1: 手动 vs FlexAttention
    diff_1 = torch.norm(manual_1 - flex_1).item()
    print(f"\n配置1 [P][M₁]: 手动 vs FlexAttention:")
    print(f"  L2距离: {diff_1:.10f}")
    print(f"  {'✅ 相同' if diff_1 < 1e-5 else '❌ 不同'}")

    # 对比配置2: 手动 vs FlexAttention
    diff_2 = torch.norm(manual_2 - flex_2).item()
    print(f"\n配置2 [P][M₁][R₁]: 手动 vs FlexAttention:")
    print(f"  L2距离: {diff_2:.10f}")
    print(f"  {'✅ 相同' if diff_2 < 1e-5 else '❌ 不同'}")

    print("\n" + "="*80)
    print("🔍 最终结论")
    print("="*80)

    if manual_diff < 1e-5 and flex_diff >= 1e-5:
        print("\n✅ 确认FlexAttention有bug！")
        print("\n证据:")
        print(f"  1. 手动计算(基于可见Q/K/V): 两种配置完全相同 (L2={manual_diff:.2e})")
        print(f"  2. FlexAttention实际输出: 两种配置不同 (L2={flex_diff:.2e})")
        print("\n这说明:")
        print("  - FlexAttention的mask并没有完全隔离不可见位置")
        print("  - 或者FlexAttention在不同序列长度下有数值稳定性问题")
        print("  - R₁的存在影响了FlexAttention的内部计算,即使M₁理论上看不到R₁")

        if diff_1 < 1e-5 and diff_2 >= 1e-5:
            print("\n进一步分析:")
            print(f"  - 配置1 [P][M₁]: 手动计算 = FlexAttention ✅ (L2={diff_1:.2e})")
            print(f"  - 配置2 [P][M₁][R₁]: 手动计算 ≠ FlexAttention ❌ (L2={diff_2:.2e})")
            print("\n  这说明只有当序列长度增加时,FlexAttention的行为才异常!")
        elif diff_1 >= 1e-5 and diff_2 < 1e-5:
            print("\n进一步分析:")
            print(f"  - 配置1 [P][M₁]: 手动计算 ≠ FlexAttention ❌ (L2={diff_1:.2e})")
            print(f"  - 配置2 [P][M₁][R₁]: 手动计算 = FlexAttention ✅ (L2={diff_2:.2e})")
            print("\n  奇怪！这不符合预期...")
        elif diff_1 >= 1e-5 and diff_2 >= 1e-5:
            print("\n进一步分析:")
            print(f"  - 配置1 [P][M₁]: 手动计算 ≠ FlexAttention ❌ (L2={diff_1:.2e})")
            print(f"  - 配置2 [P][M₁][R₁]: 手动计算 ≠ FlexAttention ❌ (L2={diff_2:.2e})")
            print("\n  FlexAttention在两种配置下都与手动计算不同")
            print("  但手动计算本身是一致的")
            print("\n  可能的原因:")
            print("    - FlexAttention的mask实现与我们的假设不同")
            print("    - 需要检查BlockMask的实际构造逻辑")
    elif manual_diff >= 1e-5:
        print("\n❌ 意外：手动计算不同！")
        print(f"  这不应该发生 (L2={manual_diff:.2e})")
    else:
        print("\n❓ 意外：FlexAttention也相同？")
        print(f"  FlexAttention L2={flex_diff:.2e}")
        print("  这与之前的结果矛盾...")


if __name__ == "__main__":
    main()
