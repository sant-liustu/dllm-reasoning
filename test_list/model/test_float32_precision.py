#!/usr/bin/env python3
"""
关键实验: 验证bfloat16精度假设
如果差异真的来自bfloat16精度,那么使用float32应该显著减小差异
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


def run_test_with_dtype(model, sample, dtype_name, device):
    """在特定dtype下运行测试"""
    print(f"\n{'='*80}")
    print(f"测试精度: {dtype_name}")
    print(f"{'='*80}\n")

    input_ids = sample['input_ids'].to(device)
    position_ids = sample['position_ids'].to(device)
    block_info = sample['block_info']
    prompt_len = sample['prompt_len']

    results = {}

    # 配置1: [P][M₁] (1个block)
    config1_blocks = 1
    current_pos = prompt_len
    for i, (seg_type, seg_idx, seg_len) in enumerate(block_info):
        current_pos += seg_len
        if i + 1 >= config1_blocks:
            break

    truncate_pos_1 = current_pos
    input_ids_1 = input_ids[:truncate_pos_1]
    position_ids_1 = position_ids[:truncate_pos_1]
    block_info_1 = block_info[:config1_blocks]

    print(f"配置1: [P][M₁], 长度={truncate_pos_1}")
    with torch.no_grad():
        outputs_1 = model(
            input_ids=input_ids_1.unsqueeze(0),
            position_ids=position_ids_1.unsqueeze(0),
            block_info=[block_info_1],
            prompt_len=[prompt_len],
            seq_lens=[truncate_pos_1],
            use_cache=False
        )

    results['config1'] = {
        'logits': outputs_1.logits[0, 277, :5].cpu(),
        'top5': outputs_1.logits[0, 277].topk(5).indices.tolist()
    }

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

    results['config3'] = {
        'logits': outputs_3.logits[0, 277, :5].cpu(),
        'top5': outputs_3.logits[0, 277].topk(5).indices.tolist()
    }

    # 计算差异
    logits_1 = results['config1']['logits']
    logits_3 = results['config3']['logits']

    l2_diff = torch.norm(logits_1 - logits_3).item()
    max_abs_diff = torch.max(torch.abs(logits_1 - logits_3)).item()

    print(f"\n差异统计:")
    print(f"  Logits (前5维):")
    print(f"    配置1: {logits_1}")
    print(f"    配置3: {logits_3}")
    print(f"  L2差异: {l2_diff:.6e}")
    print(f"  最大绝对差异: {max_abs_diff:.6e}")
    print(f"  Top-5预测:")
    print(f"    配置1: {results['config1']['top5']}")
    print(f"    配置3: {results['config3']['top5']}")
    print(f"  预测是否相同: {'✅ 是' if results['config1']['top5'] == results['config3']['top5'] else '❌ 否'}")

    return l2_diff, max_abs_diff


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("Float32 精度验证实验")
    print("="*80)
    print()
    print("目的: 验证差异是否来自bfloat16的数值精度限制")
    print("预期: 如果是精度问题,float32下差异应该显著减小")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载tokenizer和数据
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

    print(f"样本信息: 总长度={sample['input_ids'].shape[0]}, Prompt长度={sample['prompt_len']}, Block数={len(sample['block_info'])}")
    print()

    # 加载模型配置
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
    print("测试 1/2: bfloat16 (原始)")
    print("="*80)

    model_bf16 = DLLMForCausalLM(config)
    model_bf16.load_state_dict(pretrained.state_dict())
    model_bf16 = model_bf16.to(device).to(torch.bfloat16)
    model_bf16.train()

    l2_bf16, max_bf16 = run_test_with_dtype(model_bf16, sample, "bfloat16", device)
    results_summary['bfloat16'] = {'l2': l2_bf16, 'max_abs': max_bf16}

    del model_bf16
    torch.cuda.empty_cache()

    # 测试2: float32
    print("\n" + "="*80)
    print("测试 2/2: float32")
    print("="*80)

    model_fp32 = DLLMForCausalLM(config)
    model_fp32.load_state_dict(pretrained.state_dict())
    model_fp32 = model_fp32.to(device).to(torch.float32)
    model_fp32.train()

    l2_fp32, max_fp32 = run_test_with_dtype(model_fp32, sample, "float32", device)
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
        print(f"{dtype_name:<15} {l2:<15.6e} {max_abs:<15.6e} {ratio:<15.2f}x")

    print()
    print("="*80)
    print("结论分析")
    print("="*80)
    print()

    reduction_ratio = results_summary['float32']['l2'] / results_summary['bfloat16']['l2']

    if reduction_ratio < 0.1:
        print("✅ float32显著减小了差异 (减小到10%以下)")
        print("   → 结论: 差异主要来自bfloat16的数值精度限制")
        print("   → 这是预期的数值行为,不需要修复")
    elif reduction_ratio < 0.5:
        print("⚠️  float32部分减小了差异 (减小到50%以下)")
        print("   → 结论: 差异部分来自精度,但可能还有其他因素")
        print("   → 需要进一步调查")
    else:
        print("❌ float32几乎没有减小差异")
        print("   → 结论: 差异不是精度问题!")
        print("   → 必须深入调查模型内部,找到受序列长度影响的变量")
        print()
        print("可能的原因:")
        print("  1. FlexAttention内部某个参数受序列长度影响")
        print("  2. LayerNorm的统计量受整体序列影响")
        print("  3. 某个hidden的全局状态被修改")
        print("  4. BlockMask的构建逻辑有问题")
    print()


if __name__ == "__main__":
    main()
