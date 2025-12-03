#!/usr/bin/env python3
"""
简化版float32测试: 直接在float32下运行整个模型
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset
from dllm_reasoning.model.DLLM.modeling_dllm import DLLMForCausalLM
from dllm_reasoning.model.DLLM.configuration_dllm import DLLMConfig


def test_with_dtype(model, sample, device):
    """测试特定dtype下的差异"""
    input_ids = sample['input_ids'].to(device)
    position_ids = sample['position_ids'].to(device)
    block_info = sample['block_info']
    prompt_len = sample['prompt_len']

    # 配置1: [P][M₁]
    truncate_pos_1 = prompt_len
    for seg_type, seg_idx, seg_len in block_info[:1]:
        truncate_pos_1 += seg_len

    print(f"配置1: [P][M₁], 长度={truncate_pos_1}")
    with torch.no_grad():
        outputs_1 = model(
            input_ids=input_ids[:truncate_pos_1].unsqueeze(0),
            position_ids=position_ids[:truncate_pos_1].unsqueeze(0),
            block_info=[block_info[:1]],
            prompt_len=[prompt_len],
            seq_lens=[truncate_pos_1],
            use_cache=False
        )

    # 配置3: 完整序列
    print(f"配置3: 完整序列, 长度={input_ids.shape[0]}")
    with torch.no_grad():
        outputs_3 = model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            block_info=[block_info],
            prompt_len=[prompt_len],
            seq_lens=[input_ids.shape[0]],
            use_cache=False
        )

    # 计算差异
    logits_1 = outputs_1.logits[0, 277, :5].cpu()
    logits_3 = outputs_3.logits[0, 277, :5].cpu()

    l2_diff = torch.norm(logits_1 - logits_3).item()
    max_abs_diff = torch.max(torch.abs(logits_1 - logits_3)).item()

    print(f"\n差异统计:")
    print(f"  Logits (前5维):")
    print(f"    配置1: {logits_1}")
    print(f"    配置3: {logits_3}")
    print(f"  L2差异: {l2_diff:.6e}")
    print(f"  最大绝对差异: {max_abs_diff:.6e}")

    return l2_diff, max_abs_diff


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("Float32 完整模型测试")
    print("="*80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    print("加载数据集...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
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
    print(f"样本信息: 总长度={sample['input_ids'].shape[0]}, Prompt长度={sample['prompt_len']}")
    print()

    config = DLLMConfig.from_pretrained(MODEL_PATH)

    from transformers import AutoModelForCausalLM
    pretrained = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    results_summary = {}

    # 测试1: bfloat16
    print("\n" + "="*80)
    print("测试 1/2: 全部bfloat16")
    print("="*80 + "\n")

    model_bf16 = DLLMForCausalLM(config)
    model_bf16.load_state_dict(pretrained.state_dict())
    model_bf16 = model_bf16.to(device).to(torch.bfloat16)
    model_bf16.train()

    l2_bf16, max_bf16 = test_with_dtype(model_bf16, sample, device)
    results_summary['bfloat16'] = {'l2': l2_bf16, 'max_abs': max_bf16}

    del model_bf16
    torch.cuda.empty_cache()

    # 测试2: float32
    print("\n" + "="*80)
    print("测试 2/2: 全部float32")
    print("="*80 + "\n")

    model_fp32 = DLLMForCausalLM(config)
    model_fp32.load_state_dict(pretrained.state_dict())
    model_fp32 = model_fp32.to(device).to(torch.float32)  # 整个模型转为float32
    model_fp32.train()

    l2_fp32, max_fp32 = test_with_dtype(model_fp32, sample, device)
    results_summary['float32'] = {'l2': l2_fp32, 'max_abs': max_fp32}

    del model_fp32
    torch.cuda.empty_cache()

    # 最终对比
    print("\n" + "="*80)
    print("最终对比结果")
    print("="*80)
    print()
    print(f"{'精度类型':<15} {'L2差异':<15} {'最大绝对差异':<15} {'相对于bfloat16':<15}")
    print("-" * 65)

    for dtype_name, metrics in results_summary.items():
        l2 = metrics['l2']
        max_abs = metrics['max_abs']
        ratio = l2 / results_summary['bfloat16']['l2']
        print(f"{dtype_name:<15} {l2:<15.6e} {max_abs:<15.6e} {ratio:<15.2%}")

    print()
    print("="*80)
    print("结论")
    print("="*80)
    print()

    reduction_ratio = results_summary['float32']['l2'] / results_summary['bfloat16']['l2']
    print(f"Float32将差异减少了: {(1 - reduction_ratio) * 100:.2f}%")
    print(f"差异减小倍数: {1/reduction_ratio:.1f}x")
    print()

    if reduction_ratio < 0.01:
        print("✅ Float32几乎完全消除了差异")
        print("   → 差异主要来自bfloat16的数值精度")
    else:
        print("⚠️  Float32未能完全消除差异")
        print("   → 可能还有其他因素")
    print()


if __name__ == "__main__":
    main()
