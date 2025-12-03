#!/usr/bin/env python3
"""
验证: 将o_proj改为float32是否能有效抑制差异
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


def convert_oproj_to_float32(model):
    """将所有attention层的o_proj改为float32"""
    count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        # 将o_proj的权重和偏置转为float32
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data.float()
        if layer.self_attn.o_proj.bias is not None:
            layer.self_attn.o_proj.bias.data = layer.self_attn.o_proj.bias.data.float()
        count += 1
    return count


def test_with_oproj_precision(model, sample, dtype_name, device):
    """测试特定o_proj精度下的差异"""
    print(f"\n{'='*80}")
    print(f"测试配置: o_proj使用{dtype_name}, 其余使用bfloat16")
    print(f"{'='*80}\n")

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
    print(f"  Top-5预测:")
    top5_1 = outputs_1.logits[0, 277].topk(5).indices.tolist()
    top5_3 = outputs_3.logits[0, 277].topk(5).indices.tolist()
    print(f"    配置1: {top5_1}")
    print(f"    配置3: {top5_3}")
    print(f"  预测是否相同: {'✅ 是' if top5_1 == top5_3 else '❌ 否'}")

    return l2_diff, max_abs_diff


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("o_proj Float32 精度实验")
    print("="*80)
    print()
    print("目的: 验证将o_proj改为float32是否能有效抑制差异")
    print("预期: 如果o_proj是差异的根源,float32应该显著减小差异")
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

    # 加载模型配置
    config = DLLMConfig.from_pretrained(MODEL_PATH)

    from transformers import AutoModelForCausalLM
    pretrained = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    results_summary = {}

    # 测试1: 全部bfloat16 (baseline)
    print("\n" + "="*80)
    print("测试 1/2: 全部bfloat16 (基准)")
    print("="*80)

    model_bf16 = DLLMForCausalLM(config)
    model_bf16.load_state_dict(pretrained.state_dict())
    model_bf16 = model_bf16.to(device).to(torch.bfloat16)
    model_bf16.train()

    l2_bf16, max_bf16 = test_with_oproj_precision(model_bf16, sample, "bfloat16", device)
    results_summary['全部bfloat16'] = {'l2': l2_bf16, 'max_abs': max_bf16}

    del model_bf16
    torch.cuda.empty_cache()

    # 测试2: o_proj用float32, 其余bfloat16
    print("\n" + "="*80)
    print("测试 2/2: o_proj用float32, 其余bfloat16")
    print("="*80)

    model_mixed = DLLMForCausalLM(config)
    model_mixed.load_state_dict(pretrained.state_dict())
    model_mixed = model_mixed.to(device).to(torch.bfloat16)

    # 将o_proj改为float32
    count = convert_oproj_to_float32(model_mixed)
    print(f"已将{count}个层的o_proj权重转为float32\n")

    model_mixed.train()

    l2_mixed, max_mixed = test_with_oproj_precision(model_mixed, sample, "float32", device)
    results_summary['o_proj float32'] = {'l2': l2_mixed, 'max_abs': max_mixed}

    del model_mixed
    torch.cuda.empty_cache()

    # 最终对比
    print("\n" + "="*80)
    print("最终对比结果")
    print("="*80)
    print()
    print(f"{'配置':<25} {'L2差异':<15} {'最大绝对差异':<15} {'差异减少比例':<15}")
    print("-" * 75)

    for config_name, metrics in results_summary.items():
        l2 = metrics['l2']
        max_abs = metrics['max_abs']
        ratio = l2 / results_summary['全部bfloat16']['l2']
        print(f"{config_name:<25} {l2:<15.6e} {max_abs:<15.6e} {ratio:<15.2%}")

    print()
    print("="*80)
    print("结论分析")
    print("="*80)
    print()

    reduction_ratio = results_summary['o_proj float32']['l2'] / results_summary['全部bfloat16']['l2']

    print(f"将o_proj改为float32后,差异变化:")
    print(f"  L2差异: {results_summary['全部bfloat16']['l2']:.6e} → {results_summary['o_proj float32']['l2']:.6e}")
    print(f"  减少比例: {(1 - reduction_ratio) * 100:.2f}%")
    print()

    if reduction_ratio < 0.1:
        print("✅ o_proj float32显著减小了差异 (减小90%以上)")
        print("   → 结论: 差异主要来自o_proj的bfloat16精度")
        print("   → 建议: 可以考虑在训练时将o_proj保持为float32")
        print("   → 性能影响: 仅o_proj使用float32,额外开销很小")
    elif reduction_ratio < 0.5:
        print("⚠️  o_proj float32部分减小了差异 (减小50-90%)")
        print("   → 结论: o_proj是主要但不是唯一的差异来源")
        print("   → 可能还有其他层(如MLP)也贡献了部分差异")
    else:
        print("❌ o_proj float32没有显著减小差异")
        print("   → 结论: 差异可能主要来自其他地方")
        print("   → 需要进一步调查MLP或其他线性层")
    print()


if __name__ == "__main__":
    main()
