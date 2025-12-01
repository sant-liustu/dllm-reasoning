#!/usr/bin/env python3
"""
模拟训练时的前向传播，查看Mask位置的实际预测
"""

import sys
from pathlib import Path
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset
from dllm_reasoning.trainer.flex_attention_utils import create_block_mask_from_batch


def main():
    MODEL_PATH = "dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
    DATA_PATH = "data/openr1.parquet"

    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # ⚠️ 使用training mode以启用FlexAttention
    model.train()

    print("加载数据集...")
    dataset = InterleavedSFTDataset(
        parquet_files=DATA_PATH,
        tokenizer=tokenizer,
        prompt_key="prompt",
        response_key="target",
        block_size=4,
        max_length=4096,
        truncation="right",  # 使用截断
    )

    # 取第一个样本
    print("获取第一个样本...")
    sample = dataset[0]

    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    position_ids = sample['position_ids'].unsqueeze(0).to(device)
    labels = sample['labels'].unsqueeze(0).to(device)
    block_info = sample['block_info']
    prompt_len = sample['prompt_len']

    print(f"\n样本信息:")
    print(f"  序列长度: {input_ids.size(1)}")
    print(f"  Prompt长度: {prompt_len}")
    print(f"  Block数量: {len(block_info)}")

    # 构造参数（和训练时一样）
    # block_info格式需要转换：从(type, idx, len)转为(type, len)
    block_info_for_mask = [(seg_type, seg_len) for seg_type, seg_idx, seg_len in block_info]

    print("\n开始前向传播（使用训练时的调用方式）...")
    # ⚠️ 关键：和训练一样，直接传递block_info等参数给模型
    # 让模型内部创建BlockMask，而不是手动创建！
    with torch.no_grad():
        outputs = model(
            input_ids,
            position_ids=position_ids,
            # 传递 Interleaved Training 参数（和训练完全一样）
            block_info=[block_info_for_mask],
            prompt_len=[prompt_len],
            seq_lens=[input_ids.size(1)],
            use_cache=False
        )
        logits = outputs.logits  # [1, seq_len, vocab]

    print("前向传播完成")

    # 统计Mask和Real的准确率
    predictions = logits[0].argmax(dim=-1)  # [seq_len]
    labels_1d = labels[0]  # [seq_len]

    # 分类位置
    mask_positions = []
    real_positions = []

    current_pos = prompt_len
    for seg_type, seg_idx, seg_len in block_info:
        if seg_type == 'mask':
            mask_positions.extend(range(current_pos, current_pos + seg_len))
        elif seg_type == 'real':
            real_positions.extend(range(current_pos, current_pos + seg_len))
        current_pos += seg_len

    # 计算Mask准确率
    mask_pos_tensor = torch.tensor(mask_positions[:100], device=device)  # 只看前100个
    mask_labels = labels_1d[mask_pos_tensor]
    mask_preds = predictions[mask_pos_tensor]
    mask_valid = mask_labels != -100

    if mask_valid.sum() > 0:
        mask_correct = ((mask_preds == mask_labels) & mask_valid).sum()
        mask_acc = mask_correct.float() / mask_valid.sum().float()
        print(f"\nMask准确率: {mask_acc.item():.4f} ({mask_correct.item()}/{mask_valid.sum().item()})")

    # 打印前10个Mask位置的预测
    print(f"\n前10个Mask位置的预测详情:")
    count = 0
    for pos in mask_positions[:100]:
        if labels_1d[pos].item() == -100:
            continue

        pred_id = predictions[pos].item()
        true_id = labels_1d[pos].item()
        input_id = input_ids[0, pos].item()
        is_correct = "✅" if pred_id == true_id else "❌"

        pred_text = tokenizer.decode([pred_id])
        true_text = tokenizer.decode([true_id])
        input_text = tokenizer.decode([input_id])

        # 检查是否预测的是BOS
        is_bos = " [BOS!]" if pred_id == 151646 else ""

        print(f"  pos={pos:4d} {is_correct} input={input_id:6d} '{input_text}' | "
              f"pred={pred_id:6d} '{pred_text}'{is_bos} | true={true_id:6d} '{true_text}'")

        count += 1
        if count >= 10:
            break

    # 计算Real准确率
    real_pos_tensor = torch.tensor(real_positions[:100], device=device)
    real_labels = labels_1d[real_pos_tensor]
    real_preds = predictions[real_pos_tensor]
    real_valid = real_labels != -100

    if real_valid.sum() > 0:
        real_correct = ((real_preds == real_labels) & real_valid).sum()
        real_acc = real_correct.float() / real_valid.sum().float()
        print(f"\nReal准确率: {real_acc.item():.4f} ({real_correct.item()}/{real_valid.sum().item()})")


if __name__ == "__main__":
    main()
