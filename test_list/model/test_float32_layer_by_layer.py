#!/usr/bin/env python3
"""
在float32精度下逐层验证误差累积
目的: 验证在float32下,所有28层的输出是否都能对齐
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
import re

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset
from dllm_reasoning.model.DLLM.modeling_dllm import DLLMForCausalLM
from dllm_reasoning.model.DLLM.configuration_dllm import DLLMConfig


def parse_layer_outputs(log_text):
    """从日志中解析所有层的输出"""
    layers_data = {}
    lines = log_text.split('\n')

    for i, line in enumerate(lines):
        # 匹配 [Layer XX Output]
        layer_match = re.search(r'\[Layer\s+(\d+)\s+Output\]', line)
        if layer_match:
            layer_idx = int(layer_match.group(1))
            # 提取tensor值
            tensor_match = re.search(r'tensor\(\[(.*?)\]', line)
            if tensor_match:
                values_str = tensor_match.group(1)
                try:
                    # 解析数值,去掉dtype等信息
                    values = []
                    for val in values_str.split(','):
                        val = val.strip()
                        # 只取数字部分
                        if val and not val.startswith('device') and not val.startswith('dtype'):
                            try:
                                values.append(float(val))
                            except:
                                pass
                    if len(values) >= 5:
                        layers_data[layer_idx] = torch.tensor(values[:5])
                except Exception as e:
                    print(f"解析Layer {layer_idx}失败: {e}")

    return layers_data


def test_with_dtype(model, sample, dtype_name, device):
    """测试特定dtype下的逐层输出"""
    print(f"\n{'='*80}")
    print(f"测试精度: {dtype_name}")
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

    # 捕获stdout来获取layer输出
    import io
    from contextlib import redirect_stdout

    f1 = io.StringIO()
    with redirect_stdout(f1):
        with torch.no_grad():
            outputs_1 = model(
                input_ids=input_ids[:truncate_pos_1].unsqueeze(0),
                position_ids=position_ids[:truncate_pos_1].unsqueeze(0),
                block_info=[block_info[:1]],
                prompt_len=[prompt_len],
                seq_lens=[truncate_pos_1],
                use_cache=False
            )

    output_1 = f1.getvalue()

    # 配置3: 完整序列
    print(f"配置3: 完整序列, 长度={input_ids.shape[0]}")

    f3 = io.StringIO()
    with redirect_stdout(f3):
        with torch.no_grad():
            outputs_3 = model(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
                block_info=[block_info],
                prompt_len=[prompt_len],
                seq_lens=[input_ids.shape[0]],
                use_cache=False
            )

    output_3 = f3.getvalue()

    # 解析layer输出
    print("\n解析各层输出...")
    layers_1 = parse_layer_outputs(output_1)
    layers_3 = parse_layer_outputs(output_3)

    print(f"配置1解析到 {len(layers_1)} 层")
    print(f"配置3解析到 {len(layers_3)} 层")

    return layers_1, layers_3, outputs_1, outputs_3


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("Float32 逐层误差累积验证")
    print("="*80)
    print()
    print("目的: 验证在float32精度下,所有层的输出是否都能对齐")
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

    results = {}

    # 测试1: bfloat16
    print("\n" + "="*80)
    print("测试 1/2: bfloat16 (对比baseline)")
    print("="*80)

    model_bf16 = DLLMForCausalLM(config)
    model_bf16.load_state_dict(pretrained.state_dict())
    model_bf16 = model_bf16.to(device).to(torch.bfloat16)
    model_bf16.train()

    layers_1_bf16, layers_3_bf16, out1_bf16, out3_bf16 = test_with_dtype(model_bf16, sample, "bfloat16", device)
    results['bfloat16'] = {
        'layers_1': layers_1_bf16,
        'layers_3': layers_3_bf16
    }

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

    layers_1_fp32, layers_3_fp32, out1_fp32, out3_fp32 = test_with_dtype(model_fp32, sample, "float32", device)
    results['float32'] = {
        'layers_1': layers_1_fp32,
        'layers_3': layers_3_fp32
    }

    del model_fp32
    torch.cuda.empty_cache()

    # 分析对比
    print("\n" + "="*80)
    print("逐层差异分析")
    print("="*80)
    print()

    print(f"{'层号':<8} {'bfloat16 L2差异':<20} {'float32 L2差异':<20} {'bfloat16对齐':<15} {'float32对齐':<15}")
    print("-" * 85)

    for layer_idx in range(28):
        # bfloat16
        if layer_idx in results['bfloat16']['layers_1'] and layer_idx in results['bfloat16']['layers_3']:
            v1_bf16 = results['bfloat16']['layers_1'][layer_idx]
            v3_bf16 = results['bfloat16']['layers_3'][layer_idx]
            diff_bf16 = torch.norm(v1_bf16 - v3_bf16).item()
            aligned_bf16 = "✅" if diff_bf16 < 1e-4 else "❌"
        else:
            diff_bf16 = float('nan')
            aligned_bf16 = "N/A"

        # float32
        if layer_idx in results['float32']['layers_1'] and layer_idx in results['float32']['layers_3']:
            v1_fp32 = results['float32']['layers_1'][layer_idx]
            v3_fp32 = results['float32']['layers_3'][layer_idx]
            diff_fp32 = torch.norm(v1_fp32 - v3_fp32).item()
            aligned_fp32 = "✅" if diff_fp32 < 1e-6 else "❌"
        else:
            diff_fp32 = float('nan')
            aligned_fp32 = "N/A"

        print(f"Layer {layer_idx:<2} {diff_bf16:<20.6e} {diff_fp32:<20.6e} {aligned_bf16:<15} {aligned_fp32:<15}")

    print()
    print("="*80)
    print("统计总结")
    print("="*80)
    print()

    # 统计对齐情况
    bf16_aligned_count = 0
    fp32_aligned_count = 0
    total_layers = 0

    for layer_idx in range(28):
        if layer_idx in results['bfloat16']['layers_1'] and layer_idx in results['bfloat16']['layers_3']:
            total_layers += 1
            v1_bf16 = results['bfloat16']['layers_1'][layer_idx]
            v3_bf16 = results['bfloat16']['layers_3'][layer_idx]
            diff_bf16 = torch.norm(v1_bf16 - v3_bf16).item()
            if diff_bf16 < 1e-4:
                bf16_aligned_count += 1

        if layer_idx in results['float32']['layers_1'] and layer_idx in results['float32']['layers_3']:
            v1_fp32 = results['float32']['layers_1'][layer_idx]
            v3_fp32 = results['float32']['layers_3'][layer_idx]
            diff_fp32 = torch.norm(v1_fp32 - v3_fp32).item()
            if diff_fp32 < 1e-6:
                fp32_aligned_count += 1

    print(f"总层数: {total_layers}")
    print()
    print(f"bfloat16:")
    print(f"  对齐层数: {bf16_aligned_count}/{total_layers}")
    print(f"  对齐比例: {bf16_aligned_count/total_layers*100:.1f}%")
    print()
    print(f"float32:")
    print(f"  对齐层数: {fp32_aligned_count}/{total_layers}")
    print(f"  对齐比例: {fp32_aligned_count/total_layers*100:.1f}%")
    print()

    # 结论
    print("="*80)
    print("结论")
    print("="*80)
    print()

    if fp32_aligned_count == total_layers:
        print("✅ Float32下所有层完全对齐!")
        print("   → 说明在float32精度下,不同序列长度不会导致M₁位置的误差累积")
        print("   → 证明模型逻辑完全正确,差异纯粹来自bfloat16的数值精度")
    elif fp32_aligned_count > bf16_aligned_count:
        print("⚠️  Float32显著改善了对齐情况")
        print(f"   → 对齐层数从 {bf16_aligned_count} 增加到 {fp32_aligned_count}")
        print("   → 主要差异来自bfloat16精度,但可能还有其他微小因素")
    else:
        print("❌ Float32未能改善对齐情况")
        print("   → 可能存在其他非精度相关的因素")
    print()


if __name__ == "__main__":
    main()
