#!/usr/bin/env python3
"""
逐层验证修复后的模型：确保每一层的输出在不同配置下都相同

测试配置：
1. [P][M₁]
2. [P][M₁][R₁]
3. [P][M₁][R₁][M₂]
4. 完整序列

验证：每一层的 M₁ 位置输出在所有配置下都应该完全相同
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


def capture_all_layers(model, input_ids, position_ids, block_info, prompt_len, seq_len, device):
    """
    捕获所有层的 M₁ 位置输出

    Returns:
        Dict[int, torch.Tensor]: {layer_idx: hidden_state_at_m1}
    """
    m1_pos = prompt_len
    layer_outputs = {}

    # Hook每一层的输出
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is tuple: (hidden_states, ...)
            # hidden_states shape: [batch, seq_len, hidden_dim]
            hidden_states = output[0]
            layer_outputs[layer_idx] = hidden_states[0, m1_pos].detach().clone()
        return hook

    # 注册所有层的hooks
    hooks = []
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        hook = model.model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            block_info=[block_info],
            prompt_len=[prompt_len],
            seq_lens=[seq_len],
            use_cache=False
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return layer_outputs


def test_config(model, sample, num_blocks_to_keep, device):
    """
    测试特定配置下的所有层输出

    Args:
        model: 模型
        sample: 数据样本
        num_blocks_to_keep: 保留多少个block
        device: 设备

    Returns:
        Dict[int, torch.Tensor]: {layer_idx: hidden_state_at_m1}
    """
    input_ids = sample['input_ids'].to(device)
    position_ids = sample['position_ids'].to(device)
    block_info = sample['block_info']
    prompt_len = sample['prompt_len']

    # 截断序列
    current_pos = prompt_len
    blocks_seen = 0
    truncate_pos = None

    for seg_type, seg_idx, seg_len in block_info:
        blocks_seen += 1
        current_pos += seg_len
        if blocks_seen >= num_blocks_to_keep:
            truncate_pos = current_pos
            break

    if truncate_pos is None:
        truncate_pos = input_ids.size(0)

    # 截断序列和block_info
    input_ids_truncated = input_ids[:truncate_pos]
    position_ids_truncated = position_ids[:truncate_pos]

    block_info_truncated = []
    blocks_added = 0
    for seg_type, seg_idx, seg_len in block_info:
        if blocks_added >= num_blocks_to_keep:
            break
        block_info_truncated.append((seg_type, seg_idx, seg_len))
        blocks_added += 1

    # 捕获所有层输出
    layer_outputs = capture_all_layers(
        model=model,
        input_ids=input_ids_truncated,
        position_ids=position_ids_truncated,
        block_info=block_info_truncated,
        prompt_len=prompt_len,
        seq_len=truncate_pos,
        device=device,
    )

    return layer_outputs


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("逐层验证修复后的模型")
    print("="*80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("加载修复后的本地模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 从本地代码加载模型
    config = DLLMConfig.from_pretrained(MODEL_PATH)
    model = DLLMForCausalLM(config)

    # 加载预训练权重
    from transformers import AutoModelForCausalLM
    pretrained = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.load_state_dict(pretrained.state_dict())
    model = model.to(device).to(torch.bfloat16)

    # 使用训练模式
    model.train()

    num_layers = len(model.model.layers)
    print(f"模型层数: {num_layers}")

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
    print(f"  总序列长度: {sample['input_ids'].shape[0]}")
    print(f"  Prompt长度: {sample['prompt_len']}")
    print(f"  总Block数: {len(sample['block_info'])}")
    print()

    # 测试四个配置
    test_configs = [
        (1, "[P][M₁]"),
        (2, "[P][M₁][R₁]"),
        (3, "[P][M₁][R₁][M₂]"),
        (len(sample['block_info']), f"完整序列({len(sample['block_info'])}块)"),
    ]

    print("="*80)
    print("开始逐层捕获...")
    print("="*80)
    print()

    # 收集所有配置的layer outputs
    all_results = {}

    for num_blocks, config_name in test_configs:
        if num_blocks > len(sample['block_info']):
            continue

        print(f"配置: {config_name} (保留前 {num_blocks} 个blocks)")

        layer_outputs = test_config(
            model=model,
            sample=sample,
            num_blocks_to_keep=num_blocks,
            device=device,
        )

        all_results[config_name] = layer_outputs
        print(f"  ✓ 已捕获 {len(layer_outputs)} 层的输出")
        print()

    # 逐层对比
    print("="*80)
    print("逐层对比结果")
    print("="*80)
    print()

    config_names = list(all_results.keys())
    base_config = config_names[0]  # [P][M₁]

    print(f"基准配置: {base_config}")
    print(f"对比配置: {', '.join(config_names[1:])}")
    print()

    # 统计信息
    total_layers = num_layers
    all_same_count = 0
    diff_layers = []

    print(f"{'Layer':^8} | {'状态':^10} | {'最大L2差异':^20}")
    print("-" * 50)

    for layer_idx in range(num_layers):
        base_output = all_results[base_config][layer_idx]

        # 计算这一层与其他配置的最大差异
        max_diff = 0.0
        diff_config = None

        for config_name in config_names[1:]:
            other_output = all_results[config_name][layer_idx]
            diff = torch.norm(base_output - other_output).item()

            if diff > max_diff:
                max_diff = diff
                diff_config = config_name

        # 判断是否相同（阈值 1e-5）
        is_same = max_diff < 1e-5

        if is_same:
            all_same_count += 1
            status = "✅ 相同"
        else:
            status = "❌ 不同"
            diff_layers.append((layer_idx, max_diff, diff_config))

        print(f"{layer_idx:^8} | {status:^10} | {max_diff:>12.6e} ({diff_config if diff_config else 'N/A'})")

    # 总结
    print()
    print("="*80)
    print("总结")
    print("="*80)
    print()

    print(f"总层数: {total_layers}")
    print(f"完全相同的层: {all_same_count}/{total_layers} ({all_same_count/total_layers*100:.1f}%)")
    print(f"存在差异的层: {len(diff_layers)}/{total_layers}")

    if len(diff_layers) == 0:
        print()
        print("🎉 所有层的 M₁ 输出在不同配置下都完全相同！")
        print()
        print("这说明：")
        print("  ✅ BlockMask 在每一层都正确工作")
        print("  ✅ M₁ 在所有层都看不到后续 token")
        print("  ✅ 模型修复完全成功！")
    else:
        print()
        print("⚠️ 发现差异的层：")
        print()
        for layer_idx, max_diff, diff_config in diff_layers:
            print(f"  Layer {layer_idx}: 最大差异 {max_diff:.6e} (与 {diff_config} 对比)")

        print()
        print("可能的原因：")
        print("  - 数值精度问题（如果差异很小 <1e-4）")
        print("  - BlockMask 在某些层没有正确应用")
        print("  - 需要进一步调试")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
