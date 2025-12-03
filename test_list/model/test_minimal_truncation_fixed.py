#!/usr/bin/env python3
"""
使用修复后的模型测试最小truncation

测试配置：
1. [P][M₁] - 只有第一个mask block
2. [P][M₁][R₁] - 第一个mask + 第一个real
3. [P][M₁][R₁][M₂] - 前3个block
4. 完整序列

目标：验证 M₁ 的输出是否不受后面 block 的影响
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


def test_with_truncation(
    model,
    tokenizer,
    sample,
    num_blocks_to_keep: int,
    device: str = "cuda",
):
    """
    测试保留前N个block时的准确率

    Args:
        model: 模型
        tokenizer: tokenizer
        sample: 数据集样本
        num_blocks_to_keep: 保留多少个block
        device: 设备

    Returns:
        前3个Mask block的准确率列表
    """
    input_ids = sample['input_ids'].to(device)
    position_ids = sample['position_ids'].to(device)
    labels = sample['labels'].to(device)
    block_info = sample['block_info']
    prompt_len = sample['prompt_len']

    # 截断block_info和序列
    # 计算要保留的序列长度
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

    # 截断序列
    input_ids_truncated = input_ids[:truncate_pos]
    position_ids_truncated = position_ids[:truncate_pos]
    labels_truncated = labels[:truncate_pos]

    # 截断block_info - 保持完整的三元组 (seg_type, seg_idx, seg_len)
    block_info_truncated = []
    blocks_added = 0
    for seg_type, seg_idx, seg_len in block_info:
        if blocks_added >= num_blocks_to_keep:
            break
        block_info_truncated.append((seg_type, seg_idx, seg_len))
        blocks_added += 1

    # 前向传播 - 使用修复后的模型
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_truncated.unsqueeze(0),
            position_ids=position_ids_truncated.unsqueeze(0),
            block_info=[block_info_truncated],
            prompt_len=[prompt_len],
            seq_lens=[truncate_pos],
            use_cache=False
        )
        logits = outputs.logits

    predictions = logits[0].argmax(dim=-1)
    labels_1d = labels_truncated

    # 统计前3个Mask block的准确率
    mask_block_accs = []
    current_pos = prompt_len
    mask_block_count = 0

    for seg_type, seg_idx, seg_len in block_info_truncated:
        if seg_type == 'mask' and mask_block_count < 3:
            # 计算这个mask block的准确率
            mask_labels = labels_1d[current_pos:current_pos+seg_len]
            mask_preds = predictions[current_pos:current_pos+seg_len]
            valid = mask_labels != -100

            if valid.sum() > 0:
                correct = ((mask_preds == mask_labels) & valid).sum().item()
                total = valid.sum().item()
                acc = correct / total
                mask_block_accs.append(acc)
                mask_block_count += 1

        current_pos += seg_len

    return mask_block_accs


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("使用修复后的模型测试 M₁ 是否受后续block影响")
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

    print(f"样本信息:")
    print(f"  总序列长度: {sample['input_ids'].shape[0]}")
    print(f"  Prompt长度: {sample['prompt_len']}")
    print(f"  总Block数: {len(sample['block_info'])}")

    # 统计总共有多少个mask block
    total_mask_blocks = sum(1 for seg_type, _, _ in sample['block_info'] if seg_type == 'mask')
    print(f"  总Mask blocks: {total_mask_blocks}\n")

    print(f"{'='*80}")
    print(f"渐进式截断测试：理解哪个block影响M₁的预测")
    print(f"{'='*80}\n")

    # 测试四个配置
    test_configs = [
        1,    # [P][M₁]
        2,    # [P][M₁][R₁]
        3,    # [P][M₁][R₁][M₂]
        len(sample['block_info']),  # 完整序列
    ]

    results = []

    for num_blocks in test_configs:
        if num_blocks > len(sample['block_info']):
            continue

        print(f"测试配置: 保留前 {num_blocks} 个blocks...")

        mask_accs = test_with_truncation(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            num_blocks_to_keep=num_blocks,
            device=device,
        )

        # 计算前3个mask block的平均准确率
        if len(mask_accs) > 0:
            avg_acc = sum(mask_accs) / len(mask_accs)
        else:
            avg_acc = 0.0

        results.append({
            'num_blocks': num_blocks,
            'mask_accs': mask_accs,
            'avg_acc': avg_acc,
        })

        print(f"  前3个Mask block准确率: {mask_accs}")
        print(f"  平均: {avg_acc:.4f}")
        print()

    # 总结
    print(f"\n{'='*80}")
    print(f"渐进式对比：第一个Mask block（M₁）的准确率变化")
    print(f"{'='*80}\n")

    # 提取所有配置的M₁准确率
    config_names = [
        "[P][M₁]",
        "[P][M₁][R₁]",
        "[P][M₁][R₁][M₂]",
        f"完整序列({results[-1]['num_blocks']}块)" if len(results) >= 4 else "完整序列"
    ]

    print(f"{'配置':^30} | {'M₁准确率':^15}")
    print(f"{'-'*50}")

    m1_accs = []
    for i, r in enumerate(results):
        m1_acc = r['mask_accs'][0] if len(r['mask_accs']) > 0 else 0
        m1_accs.append(m1_acc)
        config_name = config_names[i] if i < len(config_names) else f"{r['num_blocks']}块"
        print(f"{config_name:^30} | {m1_acc:>8.4f} ({m1_acc*100:>6.2f}%)")

    # 分析
    print(f"\n{'='*80}")
    print(f"分析")
    print(f"{'='*80}\n")

    if len(m1_accs) >= 3:
        only_m1 = m1_accs[0]
        with_r1 = m1_accs[1]
        with_m2 = m1_accs[2]
        full = m1_accs[-1]

        print(f"1️⃣  [P][M₁] → [P][M₁][R₁]:")
        diff1 = with_r1 - only_m1
        print(f"   准确率变化: {diff1:+.4f} ({diff1/only_m1*100 if only_m1 > 0 else 0:+.2f}%)")
        if abs(diff1) < 0.01:
            print(f"   ✅ R₁对M₁基本无影响")
        elif diff1 > 0:
            print(f"   ⚠️ R₁提升了M₁的准确率")
        else:
            print(f"   ⚠️ R₁降低了M₁的准确率")
        print()

        print(f"2️⃣  [P][M₁][R₁] → [P][M₁][R₁][M₂]:")
        diff2 = with_m2 - with_r1
        print(f"   准确率变化: {diff2:+.4f} ({diff2/with_r1*100 if with_r1 > 0 else 0:+.2f}%)")
        if abs(diff2) < 0.01:
            print(f"   ✅ M₂对M₁基本无影响")
        elif diff2 > 0:
            print(f"   ⚠️ M₂提升了M₁的准确率")
        else:
            print(f"   ⚠️ M₂降低了M₁的准确率")
        print()

        print(f"3️⃣  [P][M₁][R₁][M₂] → 完整序列:")
        diff3 = full - with_m2
        print(f"   准确率变化: {diff3:+.4f} ({diff3/with_m2*100 if with_m2 > 0 else 0:+.2f}%)")
        if abs(diff3) < 0.01:
            print(f"   ✅ 后续block（R₂M₃...）对M₁基本无影响")
        elif diff3 > 0:
            print(f"   ⚠️ 后续block提升了M₁的准确率")
        else:
            print(f"   ⚠️ 后续block降低了M₁的准确率")
        print()

        print(f"{'='*80}")
        print(f"关键发现")
        print(f"{'='*80}\n")

        # 检查所有差异是否都很小
        max_diff = max(abs(diff1), abs(diff2), abs(diff3))

        if max_diff < 0.01:
            print(f"✅ 所有配置下M₁的准确率基本相同（最大差异: {max_diff:.4f}）")
            print(f"\n这说明：")
            print(f"  - BlockMask正确工作，M₁看不到后续的token")
            print(f"  - 后续block不影响M₁的预测")
            print(f"  - 模型修复成功！")
        else:
            # 找出影响最大的变化
            diffs = [
                ("添加R₁", diff1),
                ("添加M₂", diff2),
                ("添加后续blocks", diff3)
            ]
            max_diff_item = max(diffs, key=lambda x: abs(x[1]))

            print(f"⚠️ 发现显著差异！")
            print(f"\n对M₁影响最大的是：{max_diff_item[0]}")
            print(f"影响程度: {max_diff_item[1]:+.4f}")

            if abs(diff2) > 0.01 and diff2 < 0:
                print(f"\n🔍 重要：M₂的存在显著降低了M₁的准确率！")
                print(f"   这可能说明BlockMask仍有问题")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
