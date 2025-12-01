#!/usr/bin/env python3
"""
测试截断效应：验证后面的block是否会"神秘地"影响前面的预测

实验设计：
1. 使用训练数据集的完整交错格式
2. 逐步截断后面的block：保留前1个、前2个、前3个...前N个block
3. 统计前面几个block的Mask准确率
4. 如果后面的block影响前面，应该看到：截断后准确率下降

这可以验证是否存在"神秘影响"（如数值精度、softmax归一化等）
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
        num_blocks_to_keep: 保留多少个block（包括mask+real）
        device: 设备

    Returns:
        前3个Mask block的准确率列表
    """
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    position_ids = sample['position_ids'].unsqueeze(0).to(device)
    labels = sample['labels'].unsqueeze(0).to(device)
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
        truncate_pos = input_ids.size(1)

    # 截断序列
    input_ids_truncated = input_ids[:, :truncate_pos]
    position_ids_truncated = position_ids[:, :truncate_pos]
    labels_truncated = labels[:, :truncate_pos]

    # 截断block_info
    block_info_truncated = []
    blocks_added = 0
    for seg_type, seg_idx, seg_len in block_info:
        if blocks_added >= num_blocks_to_keep:
            break
        block_info_truncated.append((seg_type, seg_len))
        blocks_added += 1

    # 前向传播
    with torch.no_grad():
        outputs = model(
            input_ids_truncated,
            position_ids=position_ids_truncated,
            block_info=[block_info_truncated],
            prompt_len=[prompt_len],
            seq_lens=[truncate_pos],
            use_cache=False
        )
        logits = outputs.logits

    predictions = logits[0].argmax(dim=-1)
    labels_1d = labels_truncated[0]

    # 统计前3个Mask block的准确率
    mask_block_accs = []
    current_pos = prompt_len
    mask_block_count = 0

    for seg_type, seg_idx, seg_len in block_info:
        if blocks_added > 0:
            blocks_added -= 1
        else:
            break

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

    # 使用training mode
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
    print(f"截断效应测试")
    print(f"{'='*80}\n")

    # 测试不同的截断长度
    # 从保留2个block（1个mask+1个real）到保留所有block
    test_configs = [
        2,    # 最少：1个mask + 1个real
        4,    # 2个mask + 2个real
        6,    # 3个mask + 3个real
        10,   # 5个mask + 5个real
        20,   # 10个mask + 10个real
        50,   # 25个mask + 25个real
        100,  # 50个mask + 50个real
        len(sample['block_info']),  # 全部
    ]

    results = []

    for num_blocks in test_configs:
        if num_blocks > len(sample['block_info']):
            continue

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

        print(f"保留前 {num_blocks:3d} 个blocks:")
        print(f"  前3个Mask block准确率: {mask_accs}")
        print(f"  平均: {avg_acc:.4f}")
        print()

    # 总结
    print(f"\n{'='*80}")
    print(f"总结：前3个Mask block的准确率变化")
    print(f"{'='*80}\n")

    print(f"{'保留Blocks':^15} | {'Mask1准确率':^12} | {'Mask2准确率':^12} | {'Mask3准确率':^12} | {'平均准确率':^12}")
    print(f"{'-'*80}")

    for r in results:
        accs = r['mask_accs']
        acc1 = f"{accs[0]:.4f}" if len(accs) > 0 else "N/A"
        acc2 = f"{accs[1]:.4f}" if len(accs) > 1 else "N/A"
        acc3 = f"{accs[2]:.4f}" if len(accs) > 2 else "N/A"
        avg = f"{r['avg_acc']:.4f}"

        print(f"{r['num_blocks']:^15d} | {acc1:^12s} | {acc2:^12s} | {acc3:^12s} | {avg:^12s}")

    # 分析趋势
    print(f"\n{'='*80}")
    print(f"结论分析")
    print(f"{'='*80}\n")

    if len(results) >= 2:
        first_avg = results[0]['avg_acc']
        last_avg = results[-1]['avg_acc']
        diff = last_avg - first_avg

        print(f"最少blocks ({results[0]['num_blocks']}个) 时的平均准确率: {first_avg:.4f}")
        print(f"最多blocks ({results[-1]['num_blocks']}个) 时的平均准确率: {last_avg:.4f}")
        print(f"差异: {diff:+.4f} ({diff/first_avg*100:+.1f}%)\n")

        if diff > 0.05:
            print("⚠️ 发现显著差异！后面的block确实影响了前面的预测！")
            print("可能的原因：")
            print("  1. Softmax归一化效应：更多token导致概率分布变化")
            print("  2. 数值精度问题：虽然mask掉了，但浮点运算有累积误差")
            print("  3. FlexAttention实现的优化细节")
            print("  4. Position IDs的间接影响")
        elif diff < -0.05:
            print("⚠️ 后面的block减少了前面的准确率（负面影响）")
        else:
            print("✅ 没有显著差异，后面的block基本不影响前面的预测")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
