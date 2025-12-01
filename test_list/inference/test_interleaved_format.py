#!/usr/bin/env python3
"""
测试完整交错格式（和训练时完全一致）

验证假设：虽然逻辑上等价，但完整交错格式可能因为矩阵运算的工程实现细节，
导致预测结果和逐步推理不同。
"""

import sys
from pathlib import Path

# 完全禁用torch compile（必须在import torch之前）
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# 导入FlexAttention工具
from dllm_reasoning.trainer.flex_attention_utils import create_block_mask_from_batch


def test_full_interleaved_format(
    model,
    tokenizer,
    prompt_messages: list,
    ground_truth_content: str,
    block_size: int = 4,
    max_blocks: int = 50,
    device: str = "cuda",
):
    """
    使用完整交错格式测试（和训练时一致）

    输入格式: [Prompt][Mask₁][Real₁][Mask₂][Real₂]...[Maskₙ][Realₙ]
    """
    # 处理<think>标签（和训练对齐）
    response_content = ground_truth_content
    if response_content.strip().startswith('<think>'):
        think_start = response_content.find('<think>')
        response_content = response_content[think_start + 7:].lstrip()

    # Tokenize
    prompt_only_str = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )
    full_conversation_str = prompt_only_str + response_content + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    full_ids = tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

    prompt_len = prompt_ids.size(0)
    response_ids = full_ids[prompt_len:]
    response_len = response_ids.size(0)

    eos_token_id = tokenizer.eos_token_id
    num_masks = block_size - 1

    # 构造完整的交错序列（和训练数据处理一致）
    interleaved_ids = [prompt_ids]
    interleaved_pos = [torch.arange(prompt_len, device=device)]

    block_info = []  # 记录每个block的位置信息
    current_pos = prompt_len

    num_blocks = min((response_len + block_size - 1) // block_size, max_blocks)

    print(f"{'='*80}")
    print(f"完整交错格式测试")
    print(f"{'='*80}")
    print(f"Prompt长度: {prompt_len} tokens")
    print(f"Response长度: {response_len} tokens")
    print(f"Block size: {block_size}")
    print(f"Number of blocks: {num_blocks}")
    print(f"{'='*80}\n")

    # 构造交错序列（和训练时的逻辑一致）
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, response_len)
        block_tokens = response_ids[block_start:block_end]
        actual_block_size = block_tokens.size(0)

        if actual_block_size == 0:
            break

        # 添加Mask tokens（如果block足够大）
        if actual_block_size > num_masks:
            mask_tokens = torch.full((num_masks,), eos_token_id, dtype=prompt_ids.dtype, device=device)
            mask_positions = torch.arange(current_pos, current_pos + num_masks, device=device)

            interleaved_ids.append(mask_tokens)
            interleaved_pos.append(mask_positions)

            # 记录这个Mask block的信息
            mask_start_in_seq = sum(t.size(0) for t in interleaved_ids[:-1])
            block_info.append({
                'type': 'mask',
                'block_idx': block_idx,
                'start_in_seq': mask_start_in_seq,
                'length': num_masks,
                'positions': mask_positions.tolist(),
            })

        # 添加Real tokens
        block_positions = torch.arange(current_pos, current_pos + actual_block_size, device=device)

        interleaved_ids.append(block_tokens)
        interleaved_pos.append(block_positions)

        # 记录这个Real block的信息
        real_start_in_seq = sum(t.size(0) for t in interleaved_ids[:-1])
        block_info.append({
            'type': 'real',
            'block_idx': block_idx,
            'start_in_seq': real_start_in_seq,
            'length': actual_block_size,
            'positions': block_positions.tolist(),
            'true_tokens': block_tokens.tolist(),
        })

        current_pos += actual_block_size

    # 拼接完整序列
    interleaved_input_ids = torch.cat(interleaved_ids, dim=0).unsqueeze(0).to(device)
    interleaved_position_ids = torch.cat(interleaved_pos, dim=0).unsqueeze(0).to(device)

    print(f"交错序列长度: {interleaved_input_ids.size(1)} tokens")
    print(f"Position IDs范围: [{interleaved_position_ids.min().item()}, {interleaved_position_ids.max().item()}]")

    # Debug: 打印前2个block的详细信息
    print(f"\n=== Debug: 前2个block的详细信息 ===")
    for idx, info in enumerate(block_info[:4]):  # 前2个block (mask+real)
        print(f"\nBlock {idx}: type={info['type']}, start={info['start_in_seq']}, length={info['length']}")
        start = info['start_in_seq']
        length = info['length']
        seq_ids = interleaved_input_ids[0, start:start+length]
        seq_pos = interleaved_position_ids[0, start:start+length]
        print(f"  Input IDs: {seq_ids.tolist()}")
        print(f"  Position IDs: {seq_pos.tolist()}")
        print(f"  Tokens: {tokenizer.convert_ids_to_tokens(seq_ids.tolist())}")
        if info['type'] == 'real':
            print(f"  True tokens: {info['true_tokens']}")

    print(f"\n构造BlockMask...\n")

    # 构造batch（和训练时一样的格式）
    batch = {
        'input_ids': interleaved_input_ids,
        'block_info': [[(info['type'], info['length']) for info in block_info]],
        'prompt_len': [prompt_len],
        'seq_lens': [interleaved_input_ids.size(1)],
    }

    # 创建BlockMask（和训练时一样）
    block_mask = create_block_mask_from_batch(batch, device)

    print(f"BlockMask created successfully")
    print(f"\n开始前向传播...\n")

    # 前向传播（和训练时一样，传入完整的交错序列+BlockMask）
    with torch.no_grad():
        outputs = model(
            interleaved_input_ids,
            attention_mask=block_mask,  # ⚠️ 传入BlockMask
            position_ids=interleaved_position_ids,
            use_cache=False
        )
        logits = outputs.logits  # [1, seq_len, vocab]

    # 统计Mask和Real block的准确率
    print(f"{'='*80}")
    print(f"统计Mask和Real block的准确率")
    print(f"{'='*80}\n")

    mask_block_accuracies = []
    real_block_accuracies = []
    total_mask_correct = 0
    total_mask_tokens = 0
    total_real_correct = 0
    total_real_tokens = 0

    # 先构造labels（参考训练代码的逻辑）
    seq_len = interleaved_input_ids.size(1)
    labels = torch.full((seq_len,), -100, dtype=torch.long, device=device)

    # 为每个block设置labels（和训练代码对齐）
    for info in block_info:
        if info['type'] == 'mask':
            # Mask预测Real block中的对应token
            # 找到对应的Real block
            block_idx = info['block_idx']
            real_block = None
            for r_info in block_info:
                if r_info['type'] == 'real' and r_info['block_idx'] == block_idx:
                    real_block = r_info
                    break

            if real_block is None:
                continue

            # Mask[i]预测Real[i+1]（参考训练代码第268行）
            start_in_seq = info['start_in_seq']
            length = info['length']
            real_tokens = real_block['true_tokens']

            for i in range(length):
                mask_pos = start_in_seq + i
                target_idx = i + 1  # Mask[i]预测Real[i+1]
                if target_idx < len(real_tokens):
                    labels[mask_pos] = real_tokens[target_idx]

        elif info['type'] == 'real':
            # Real[i]预测Real[i+1]（参考训练代码第279行）
            start_in_seq = info['start_in_seq']
            length = info['length']
            real_tokens = info['true_tokens']

            for i in range(length):
                real_pos = start_in_seq + i
                target_idx = i + 1
                if target_idx < len(real_tokens):
                    labels[real_pos] = real_tokens[target_idx]

    # Prompt[-1]预测response[0]（参考训练代码第290行）
    if prompt_len > 0 and response_len > 0:
        labels[prompt_len - 1] = response_ids[0].item()

    # Debug: 打印前2个block的labels
    print(f"\n=== Debug: 前2个block的Labels ===")
    for idx, info in enumerate(block_info[:4]):  # 前2个block
        print(f"\nBlock {idx}: type={info['type']}")
        start = info['start_in_seq']
        length = info['length']
        block_labels = labels[start:start+length]
        print(f"  Labels: {block_labels.tolist()}")
        valid_labels = [lbl for lbl in block_labels.tolist() if lbl != -100]
        if len(valid_labels) > 0:
            print(f"  Valid label tokens: {tokenizer.convert_ids_to_tokens(valid_labels)}")

    # 现在用logits和labels对比
    predictions = logits[0].argmax(dim=-1)  # [seq_len]

    # 统计Mask准确率
    for info in block_info:
        if info['type'] != 'mask':
            continue

        block_idx = info['block_idx']
        start_in_seq = info['start_in_seq']
        length = info['length']

        # 提取这个Mask block的预测和labels
        mask_preds = predictions[start_in_seq : start_in_seq+length]
        mask_labels = labels[start_in_seq : start_in_seq+length]

        # 只统计有效的位置（label != -100）
        valid_mask = mask_labels != -100
        if valid_mask.sum() == 0:
            continue

        mask_correct = ((mask_preds == mask_labels) & valid_mask).sum().item()
        mask_total = valid_mask.sum().item()
        acc = mask_correct / mask_total

        mask_block_accuracies.append(acc)
        total_mask_correct += mask_correct
        total_mask_tokens += mask_total

        print(f"Block {block_idx}: Mask准确率 = {acc:.4f} ({mask_correct}/{mask_total})")

        # 显示前几个token的预测详情
        if block_idx < 5:
            print(f"  预测详情:")
            for i in range(min(3, length)):
                if mask_labels[i].item() == -100:
                    continue
                pred_id = mask_preds[i].item()
                true_id = mask_labels[i].item()
                is_correct = "✅" if pred_id == true_id else "❌"
                pred_text = tokenizer.decode([pred_id])
                true_text = tokenizer.decode([true_id])

                # 显示top-5 logits来debug
                pos_in_seq = start_in_seq + i
                token_logits = logits[0, pos_in_seq]  # [vocab_size]
                top5_values, top5_indices = token_logits.topk(5)
                top5_texts = [tokenizer.decode([idx.item()]) for idx in top5_indices]

                print(f"    [{i}] {is_correct} pred={pred_id:6d} '{pred_text}' | true={true_id:6d} '{true_text}'")
                print(f"         Top-5: {list(zip(top5_indices.tolist(), top5_texts, top5_values.tolist()))}")

    # 统计Real准确率
    print(f"\n统计Real block准确率:\n")
    for info in block_info:
        if info['type'] != 'real':
            continue

        block_idx = info['block_idx']
        start_in_seq = info['start_in_seq']
        length = info['length']

        # 提取这个Real block的预测和labels
        real_preds = predictions[start_in_seq : start_in_seq+length]
        real_labels = labels[start_in_seq : start_in_seq+length]

        # 只统计有效的位置
        valid_mask = real_labels != -100
        if valid_mask.sum() == 0:
            continue

        real_correct = ((real_preds == real_labels) & valid_mask).sum().item()
        real_total = valid_mask.sum().item()
        acc = real_correct / real_total

        real_block_accuracies.append(acc)
        total_real_correct += real_correct
        total_real_tokens += real_total

        if block_idx < 5:
            print(f"Block {block_idx}: Real准确率 = {acc:.4f} ({real_correct}/{real_total})")

    # 总结
    print(f"\n{'='*80}")
    print(f"测试总结")
    print(f"{'='*80}")

    print(f"\nMask统计:")
    print(f"  总Mask blocks: {len(mask_block_accuracies)}")
    print(f"  总Mask tokens: {total_mask_tokens}")
    print(f"  总正确: {total_mask_correct}")
    print(f"  总体Mask准确率: {total_mask_correct / total_mask_tokens if total_mask_tokens > 0 else 0:.4f}")
    if len(mask_block_accuracies) > 0:
        print(f"  平均准确率: {sum(mask_block_accuracies) / len(mask_block_accuracies):.4f}")
        print(f"  最高准确率: {max(mask_block_accuracies):.4f}")
        print(f"  最低准确率: {min(mask_block_accuracies):.4f}")

    print(f"\nReal统计:")
    print(f"  总Real blocks: {len(real_block_accuracies)}")
    print(f"  总Real tokens: {total_real_tokens}")
    print(f"  总正确: {total_real_correct}")
    print(f"  总体Real准确率: {total_real_correct / total_real_tokens if total_real_tokens > 0 else 0:.4f}")
    if len(real_block_accuracies) > 0:
        print(f"  平均准确率: {sum(real_block_accuracies) / len(real_block_accuracies):.4f}")
        print(f"  最高准确率: {max(real_block_accuracies):.4f}")
        print(f"  最低准确率: {min(real_block_accuracies):.4f}")

    print(f"\n总体统计:")
    total_correct = total_mask_correct + total_real_correct
    total_tokens = total_mask_tokens + total_real_tokens
    print(f"  总tokens: {total_tokens}")
    print(f"  总正确: {total_correct}")
    print(f"  总体准确率: {total_correct / total_tokens if total_tokens > 0 else 0:.4f}")

    return mask_block_accuracies, real_block_accuracies


def main():
    import pandas as pd
    import numpy as np

    MODEL_PATH = "dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
    DATA_PATH = "data/openr1.parquet"

    print("加载数据...")
    df = pd.read_parquet(DATA_PATH)
    sample = df.iloc[0]

    prompt_messages = sample['prompt']
    target_messages = sample['target']

    # 处理numpy arrays
    if isinstance(prompt_messages, np.ndarray):
        prompt_messages = prompt_messages.tolist() if prompt_messages.ndim == 0 else list(prompt_messages)
    if isinstance(target_messages, np.ndarray):
        target_messages = target_messages.tolist() if target_messages.ndim == 0 else list(target_messages)

    # 提取content
    if isinstance(target_messages, (list, tuple)) and len(target_messages) > 0 and isinstance(target_messages[0], dict):
        ground_truth_content = target_messages[0].get("content", "")
    else:
        ground_truth_content = target_messages

    print(f"✅ 数据加载完成\n")

    print("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # ⚠️ 关键：设置为training mode以使用FlexAttention
    # 但不计算梯度（用torch.no_grad()）
    model.train()

    print(f"✅ 模型加载完成（training mode以使用FlexAttention）\n")

    # 运行测试
    test_full_interleaved_format(
        model=model,
        tokenizer=tokenizer,
        prompt_messages=prompt_messages,
        ground_truth_content=ground_truth_content,
        block_size=4,
        max_blocks=50,
        device=device,
    )


if __name__ == "__main__":
    main()
