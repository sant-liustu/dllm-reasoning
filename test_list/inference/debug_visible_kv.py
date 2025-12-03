#!/usr/bin/env python3
"""
检查M₁可见位置的K/V是否相同

M₁应该只能看到：
- Prompt (0-276): 277个位置
- M₁自己 (277): 1个位置

总共278个位置
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


def capture_full_kv(model, input_ids, position_ids, block_info, prompt_len, device):
    """
    捕获完整的K/V矩阵
    """
    captured = {}

    # Hook to capture K/V after projection (before RoPE is applied in attention)
    def hook_k_proj(module, input, output):
        # output: [batch, seq_len, num_kv_heads * head_dim]
        captured['k_proj_output'] = output.detach().clone()

    def hook_v_proj(module, input, output):
        captured['v_proj_output'] = output.detach().clone()

    layer0 = model.model.layers[0]
    hooks = []

    hooks.append(layer0.self_attn.k_proj.register_forward_hook(hook_k_proj))
    hooks.append(layer0.self_attn.v_proj.register_forward_hook(hook_v_proj))

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


def compare_visible_kv(model, tokenizer, sample, device):
    """
    对比M₁可见位置的K/V
    """
    print("="*80)
    print("M₁可见位置的K/V对比")
    print("="*80)

    prompt_len = sample['prompt_len']
    m1_pos = prompt_len
    input_ids_full = sample['input_ids'].to(device)
    position_ids_full = sample['position_ids'].to(device)

    print(f"\nM₁位置: {m1_pos}")
    print(f"M₁应该可以看到的位置: 0-{m1_pos} (共{m1_pos+1}个位置)")
    print(f"  - Prompt: 0-{prompt_len-1} ({prompt_len}个)")
    print(f"  - M₁自己: {m1_pos} (1个)")

    # 配置1: [P][M₁]
    print("\n" + "#"*80)
    print("配置1: [P][M₁]")
    print("#"*80)

    truncate_pos_1 = prompt_len + 3
    input_ids_1 = input_ids_full[:truncate_pos_1]
    position_ids_1 = position_ids_full[:truncate_pos_1]
    block_info_1 = [('mask', 3)]

    captured_1 = capture_full_kv(
        model, input_ids_1, position_ids_1, block_info_1, prompt_len, device
    )

    k1 = captured_1['k_proj_output'][0]  # [seq_len, hidden]
    v1 = captured_1['v_proj_output'][0]

    print(f"K矩阵 shape: {k1.shape}")
    print(f"V矩阵 shape: {v1.shape}")

    # 配置2: [P][M₁][R₁]
    print("\n" + "#"*80)
    print("配置2: [P][M₁][R₁]")
    print("#"*80)

    truncate_pos_2 = prompt_len + 7
    input_ids_2 = input_ids_full[:truncate_pos_2]
    position_ids_2 = position_ids_full[:truncate_pos_2]
    block_info_2 = [('mask', 3), ('real', 4)]

    captured_2 = capture_full_kv(
        model, input_ids_2, position_ids_2, block_info_2, prompt_len, device
    )

    k2 = captured_2['k_proj_output'][0]  # [seq_len, hidden]
    v2 = captured_2['v_proj_output'][0]

    print(f"K矩阵 shape: {k2.shape}")
    print(f"V矩阵 shape: {v2.shape}")

    # 对比M₁可见的位置 (0 to m1_pos)
    print("\n" + "="*80)
    print(f"对比M₁可见位置 (0-{m1_pos}) 的K/V")
    print("="*80)

    visible_k1 = k1[:m1_pos+1]  # [m1_pos+1, hidden]
    visible_k2 = k2[:m1_pos+1]

    visible_v1 = v1[:m1_pos+1]
    visible_v2 = v2[:m1_pos+1]

    k_diff = torch.norm(visible_k1 - visible_k2).item()
    v_diff = torch.norm(visible_v1 - visible_v2).item()

    print(f"\n前{m1_pos+1}个位置的K:")
    print(f"  L2距离: {k_diff:.10f}")
    if k_diff < 1e-5:
        print(f"  ✅ 完全相同")
    else:
        print(f"  ❌ 不同！")

    print(f"\n前{m1_pos+1}个位置的V:")
    print(f"  L2距离: {v_diff:.10f}")
    if v_diff < 1e-5:
        print(f"  ✅ 完全相同")
    else:
        print(f"  ❌ 不同！")

    # 逐位置检查
    if k_diff > 1e-5 or v_diff > 1e-5:
        print(f"\n逐位置检查差异:")
        print(f"{'位置':^8} | {'K L2距离':^15} | {'V L2距离':^15}")
        print("-"*50)

        for pos in range(min(10, m1_pos+1)):  # 只显示前10个
            k_pos_diff = torch.norm(k1[pos] - k2[pos]).item()
            v_pos_diff = torch.norm(v1[pos] - v2[pos]).item()
            print(f"{pos:^8} | {k_pos_diff:>12.6f}  | {v_pos_diff:>12.6f}")

        if m1_pos+1 > 10:
            print("...")
            # 显示最后几个
            for pos in range(max(10, m1_pos-3), m1_pos+1):
                k_pos_diff = torch.norm(k1[pos] - k2[pos]).item()
                v_pos_diff = torch.norm(v1[pos] - v2[pos]).item()
                print(f"{pos:^8} | {k_pos_diff:>12.6f}  | {v_pos_diff:>12.6f}")

    print("\n" + "="*80)
    print("最终结论")
    print("="*80)

    if k_diff < 1e-5 and v_diff < 1e-5:
        print("\n✅ M₁可见位置的K/V完全相同！")
        print("\n这非常奇怪！因为：")
        print("  - Attention输入相同 ✅")
        print("  - M₁可见的K/V相同 ✅")
        print("  - 但Attention输出不同 ❌")
        print("\n⚠️ 问题可能在于：")
        print("  1. FlexAttention的BlockMask构造有误")
        print("  2. M₁实际上attend到了不应该看到的位置")
        print("  3. 或者是FlexAttention实现的bug")
    else:
        print("\n❌ M₁可见位置的K/V不同！")
        print("\n这说明：")
        print("  - 虽然输入相同，但K/V projection的输出不同")
        print("  - 可能是因为：")
        print("    a) K/V projection使用了全局信息？")
        print("    b) 或者我们对'可见位置'的理解有误？")


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
    compare_visible_kv(model, tokenizer, sample, device)


if __name__ == "__main__":
    main()
