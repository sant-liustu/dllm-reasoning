#!/usr/bin/env python3
"""
对比在bfloat16和float32下,o_proj的输出是否对齐
目的: 验证在float32下,配置1和配置3在M₁位置的o_proj输出是否相同
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


def extract_oproj_outputs(log_lines):
    """从日志中提取o_proj输出"""
    results = {}
    current_config = None

    for line in log_lines:
        if "配置1:" in line:
            current_config = "config1"
        elif "配置3:" in line:
            current_config = "config3"
        elif "After o_proj (attention final output):" in line:
            # 下一行包含actual值
            continue
        elif current_config and "attn_output[0, 277, :5] =" in line and "After o_proj" in log_lines[log_lines.index(line)-2]:
            # 提取tensor值
            import re
            match = re.search(r'tensor\((.*?)\)', line)
            if match:
                values_str = match.group(1)
                # 解析数值
                values_str = values_str.replace('[', '').replace(']', '').replace('device=', '').replace('cuda:0', '').replace('dtype=', '').replace('torch.bfloat16', '').replace('torch.float32', '').replace(',', '')
                try:
                    values = [float(x) for x in values_str.split() if x and not x.startswith('device') and not x.startswith('dtype')][:5]
                    results[current_config] = values
                except:
                    pass

    return results


def test_with_dtype(model, sample, dtype_name, device):
    """测试特定dtype下o_proj的输出对齐情况"""
    print(f"\n{'='*80}")
    print(f"测试: {dtype_name}")
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
    print(f"\n配置3: 完整序列, 长度={input_ids.shape[0]}")
    with torch.no_grad():
        outputs_3 = model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            block_info=[block_info],
            prompt_len=[prompt_len],
            seq_lens=[input_ids.shape[0]],
            use_cache=False
        )

    return outputs_1, outputs_3


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("o_proj输出对齐测试")
    print("="*80)
    print()
    print("目的: 验证在不同精度下,o_proj输出是否能对齐")
    print("对齐的定义: 配置1和配置3在M₁位置的o_proj输出数值相同")
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
    print("测试 1/2: bfloat16")
    print("="*80)

    model_bf16 = DLLMForCausalLM(config)
    model_bf16.load_state_dict(pretrained.state_dict())
    model_bf16 = model_bf16.to(device).to(torch.bfloat16)
    model_bf16.train()

    # 捕获输出
    import io
    import sys as system
    from contextlib import redirect_stdout, redirect_stderr

    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        outputs_1_bf16, outputs_3_bf16 = test_with_dtype(model_bf16, sample, "bfloat16", device)

    output_bf16 = f.getvalue()

    # 从输出中提取o_proj的值
    print("\n从模型内部debug输出中提取o_proj输出:")
    print("(查找 'After o_proj (attention final output):' 后的tensor值)")

    lines = output_bf16.split('\n')
    oproj_output_bf16_1 = None
    oproj_output_bf16_3 = None

    for i, line in enumerate(lines):
        if "After o_proj (attention final output):" in line:
            # 下一行包含tensor
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if "config1" not in locals() and "tensor([" in next_line:
                    import re
                    match = re.search(r'tensor\(\[(.*?)\]', next_line)
                    if match and oproj_output_bf16_1 is None:
                        vals = [float(x.strip()) for x in match.group(1).split(',')]
                        oproj_output_bf16_1 = torch.tensor(vals)
                        print(f"  配置1 (bf16): {oproj_output_bf16_1}")
                    elif oproj_output_bf16_3 is None:
                        vals = [float(x.strip()) for x in match.group(1).split(',')]
                        oproj_output_bf16_3 = torch.tensor(vals)
                        print(f"  配置3 (bf16): {oproj_output_bf16_3}")

    if oproj_output_bf16_1 is not None and oproj_output_bf16_3 is not None:
        diff_bf16 = torch.norm(oproj_output_bf16_1 - oproj_output_bf16_3).item()
        max_diff_bf16 = torch.max(torch.abs(oproj_output_bf16_1 - oproj_output_bf16_3)).item()
        results_summary['bfloat16'] = {
            'l2_diff': diff_bf16,
            'max_diff': max_diff_bf16,
            'aligned': diff_bf16 < 1e-4
        }
        print(f"\nbfloat16 o_proj输出差异:")
        print(f"  L2差异: {diff_bf16:.6e}")
        print(f"  最大绝对差异: {max_diff_bf16:.6e}")
        print(f"  是否对齐(< 1e-4): {'✅ 是' if diff_bf16 < 1e-4 else '❌ 否'}")

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

    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        outputs_1_fp32, outputs_3_fp32 = test_with_dtype(model_fp32, sample, "float32", device)

    output_fp32 = f.getvalue()

    print("\n从模型内部debug输出中提取o_proj输出:")

    lines = output_fp32.split('\n')
    oproj_output_fp32_1 = None
    oproj_output_fp32_3 = None

    for i, line in enumerate(lines):
        if "After o_proj (attention final output):" in line:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if "tensor([" in next_line:
                    import re
                    match = re.search(r'tensor\(\[(.*?)\]', next_line)
                    if match and oproj_output_fp32_1 is None:
                        vals = [float(x.strip()) for x in match.group(1).split(',')]
                        oproj_output_fp32_1 = torch.tensor(vals)
                        print(f"  配置1 (fp32): {oproj_output_fp32_1}")
                    elif oproj_output_fp32_3 is None:
                        vals = [float(x.strip()) for x in match.group(1).split(',')]
                        oproj_output_fp32_3 = torch.tensor(vals)
                        print(f"  配置3 (fp32): {oproj_output_fp32_3}")

    if oproj_output_fp32_1 is not None and oproj_output_fp32_3 is not None:
        diff_fp32 = torch.norm(oproj_output_fp32_1 - oproj_output_fp32_3).item()
        max_diff_fp32 = torch.max(torch.abs(oproj_output_fp32_1 - oproj_output_fp32_3)).item()
        results_summary['float32'] = {
            'l2_diff': diff_fp32,
            'max_diff': max_diff_fp32,
            'aligned': diff_fp32 < 1e-4
        }
        print(f"\nfloat32 o_proj输出差异:")
        print(f"  L2差异: {diff_fp32:.6e}")
        print(f"  最大绝对差异: {max_diff_fp32:.6e}")
        print(f"  是否对齐(< 1e-4): {'✅ 是' if diff_fp32 < 1e-4 else '❌ 否'}")

    del model_fp32
    torch.cuda.empty_cache()

    # 最终对比
    print("\n" + "="*80)
    print("最终对比: o_proj输出对齐情况")
    print("="*80)
    print()
    print(f"{'精度类型':<15} {'L2差异':<15} {'最大绝对差异':<15} {'是否对齐':<15}")
    print("-" * 65)

    for dtype_name, metrics in results_summary.items():
        l2 = metrics['l2_diff']
        max_diff = metrics['max_diff']
        aligned = "✅ 是" if metrics['aligned'] else "❌ 否"
        print(f"{dtype_name:<15} {l2:<15.6e} {max_diff:<15.6e} {aligned:<15}")

    print()
    print("="*80)
    print("结论")
    print("="*80)
    print()

    if 'bfloat16' in results_summary and 'float32' in results_summary:
        if results_summary['float32']['aligned']:
            print("✅ Float32下o_proj输出完全对齐!")
            print("   → 说明o_proj的差异确实来自bfloat16的数值精度")
            print("   → 在float32下,不同序列长度不影响o_proj在M₁位置的输出")
        else:
            print("⚠️  Float32下o_proj输出仍有差异")
            print("   → 可能还有其他因素影响")
    print()


if __name__ == "__main__":
    main()
