#!/usr/bin/env python3
"""
使用简单的hook捕获attention的输入和输出，找出差异点
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


def capture_with_hooks(model, input_ids, position_ids, block_info, prompt_len, device):
    """
    使用hooks捕获Layer 0 attention的输入和输出
    """
    m1_pos = prompt_len
    captured = {}

    # Hook 1: Capture attention input (normalized hidden states)
    def hook_attn_input(module, input, output):
        # input[0] is hidden_states after layer norm
        captured['attn_input'] = input[0][0, m1_pos, :].detach().clone()

    # Hook 2: Capture attention output
    def hook_attn_output(module, input, output):
        # output[0] is attention output
        captured['attn_output'] = output[0][0, m1_pos, :].detach().clone()

    # Hook 3: Capture full K/V shapes (from q_proj input)
    def hook_input_for_shape(module, input, output):
        # input[0] is the hidden states, shape tells us seq_len
        captured['seq_len'] = input[0].shape[1]

    layer0 = model.model.layers[0]
    hooks = []

    hooks.append(layer0.input_layernorm.register_forward_hook(lambda m, i, o:
        captured.update({'attn_input': o[0, m1_pos, :].detach().clone()})))

    hooks.append(layer0.self_attn.register_forward_hook(lambda m, i, o:
        captured.update({'attn_output': o[0][0, m1_pos, :].detach().clone()})))

    hooks.append(layer0.self_attn.q_proj.register_forward_hook(lambda m, i, o:
        captured.update({'seq_len': i[0].shape[1]})))

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


def compare_attention(model, tokenizer, sample, device):
    """
    对比两种配置的attention输入和输出
    """
    print("="*80)
    print("Attention输入输出对比")
    print("="*80)

    prompt_len = sample['prompt_len']
    input_ids_full = sample['input_ids'].to(device)
    position_ids_full = sample['position_ids'].to(device)

    # 配置1: [P][M₁]
    print("\n配置1: [P][M₁]")
    truncate_pos_1 = prompt_len + 3
    input_ids_1 = input_ids_full[:truncate_pos_1]
    position_ids_1 = position_ids_full[:truncate_pos_1]
    block_info_1 = [('mask', 3)]

    captured_1 = capture_with_hooks(
        model, input_ids_1, position_ids_1, block_info_1, prompt_len, device
    )
    print(f"  序列长度: {captured_1['seq_len']}")

    # 配置2: [P][M₁][R₁]
    print("\n配置2: [P][M₁][R₁]")
    truncate_pos_2 = prompt_len + 7
    input_ids_2 = input_ids_full[:truncate_pos_2]
    position_ids_2 = position_ids_full[:truncate_pos_2]
    block_info_2 = [('mask', 3), ('real', 4)]

    captured_2 = capture_with_hooks(
        model, input_ids_2, position_ids_2, block_info_2, prompt_len, device
    )
    print(f"  序列长度: {captured_2['seq_len']}")

    # 对比
    print("\n" + "="*80)
    print("对比结果（M₁位置）")
    print("="*80)

    # Attention input
    input_diff = torch.norm(captured_1['attn_input'] - captured_2['attn_input']).item()
    print(f"\nAttention输入（LayerNorm后）:")
    print(f"  L2距离: {input_diff:.10f}")
    if input_diff < 1e-5:
        print(f"  ✅ 完全相同")
    else:
        print(f"  ❌ 不同！")

    # Attention output
    output_diff = torch.norm(captured_1['attn_output'] - captured_2['attn_output']).item()
    print(f"\nAttention输出:")
    print(f"  L2距离: {output_diff:.10f}")
    if output_diff < 1e-5:
        print(f"  ✅ 完全相同")
    else:
        print(f"  ❌ 不同！")

    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if input_diff < 1e-5 and output_diff > 1e-5:
        print("\n⚠️ 关键发现：")
        print("  - Attention的**输入完全相同**")
        print("  - Attention的**输出却不同**")
        print("\n这说明问题出在attention计算本身！")
        print("\n可能的原因：")
        print("  1. 序列长度不同 (280 vs 284)")
        print("  2. 虽然M₁的输入相同，但K/V矩阵包含不同数量的token")
        print("  3. FlexAttention在不同序列长度下的行为可能不同")
        print("\n需要进一步检查：")
        print("  - M₁可以attend到哪些位置？")
        print("  - 这些位置的K/V值是否相同？")


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

    # 运行对比
    compare_attention(model, tokenizer, sample, device)


if __name__ == "__main__":
    main()
