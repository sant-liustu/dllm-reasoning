#!/usr/bin/env python3
"""
Teacher Forcing 测试：逐block解码并统计准确率

测试逻辑：
1. 每次添加3个Mask token（EOS）
2. 前向传播预测4个token
3. 和真实答案对比，统计这个block的准确率
4. 用真实答案替换（teacher forcing）
5. 继续下一个block

这样可以避免累积误差，验证模型是否真的学会了block预测
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def teacher_forcing_test(
    model,
    tokenizer,
    prompt_messages: list,
    ground_truth_content: str,
    block_size: int = 4,
    max_blocks: int = 100,
    device: str = "cuda",
):
    """
    Teacher forcing测试

    Args:
        model: 语言模型
        tokenizer: tokenizer
        prompt_messages: prompt消息列表（和训练时一样的格式）
        ground_truth_content: 真实回答文本（用于teacher forcing）
        block_size: block大小（默认4）
        max_blocks: 最多测试多少个block
        device: 设备
    """
    # 和训练时对齐：处理<think>标签
    response_content = ground_truth_content
    if response_content.strip().startswith('<think>'):
        think_start = response_content.find('<think>')
        response_content = response_content[think_start + 7:].lstrip()

    # Tokenize prompt（和训练时一样）
    prompt_only_str = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )

    # 完整的conversation（和训练时一样）
    full_conversation_str = prompt_only_str + response_content + tokenizer.eos_token

    # Tokenize
    prompt_ids = tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].unsqueeze(0).to(device)
    full_ids = tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].unsqueeze(0).to(device)

    prompt_len = prompt_ids.size(1)
    full_len = full_ids.size(1)
    response_len = full_len - prompt_len

    print(f"{'='*80}")
    print(f"Teacher Forcing 测试")
    print(f"{'='*80}")
    print(f"Prompt长度: {prompt_len} tokens")
    print(f"Response长度: {response_len} tokens")
    print(f"Block size: {block_size}")
    print(f"Number of masks per block: {block_size - 1}")
    print(f"Expected blocks: {(response_len + block_size - 1) // block_size}")
    print(f"{'='*80}\n")

    # 提取真实response的token IDs
    ground_truth_response = full_ids[0, prompt_len:]

    eos_token_id = tokenizer.eos_token_id
    num_masks = block_size - 1  # 3个mask

    # 初始化
    current_ids = prompt_ids.clone()

    # 统计信息
    block_accuracies = []
    total_correct = 0
    total_tokens = 0

    num_blocks = min((response_len + block_size - 1) // block_size, max_blocks)

    for block_idx in range(num_blocks):
        block_start_in_response = block_idx * block_size
        block_end_in_response = min((block_idx + 1) * block_size, response_len)
        actual_block_size = block_end_in_response - block_start_in_response

        if actual_block_size == 0:
            break

        # 获取这个block的真实答案
        true_block = ground_truth_response[block_start_in_response:block_end_in_response]

        print(f"\n{'='*80}")
        print(f"Block {block_idx}: 预测位置 [{block_start_in_response}:{block_end_in_response}]")
        print(f"{'='*80}")

        # Step 1: 添加 num_masks 个 EOS token
        pre_length = current_ids.size(1)
        mask_block = torch.full((1, num_masks), eos_token_id, device=device, dtype=current_ids.dtype)
        input_with_masks = torch.cat([current_ids, mask_block], dim=1)

        print(f"输入长度: {input_with_masks.size(1)} (pre={pre_length} + masks={num_masks})")

        # Step 2: 前向传播
        with torch.no_grad():
            outputs = model(input_with_masks, use_cache=False)
            logits = outputs.logits  # [1, pre_length + num_masks, vocab]

        # Step 3: 提取新block的logits
        # 想预测位置: [pre_length, pre_length+1, ..., pre_length+block_size-1]
        # 共 block_size 个位置
        # 需要 logits[:, pre_length-1 : pre_length+num_masks, :]
        new_block_logits = logits[:, pre_length-1 : pre_length+num_masks, :]
        # new_block_logits 形状: [1, block_size, vocab]

        # Step 4: 解码（贪婪）
        predicted_block = new_block_logits.argmax(dim=-1)[0]  # [block_size]

        # Step 5: 只取实际需要的长度
        predicted_block = predicted_block[:actual_block_size]

        # Step 6: 对比准确率
        correct_mask = (predicted_block == true_block)
        num_correct = correct_mask.sum().item()
        block_acc = num_correct / actual_block_size

        block_accuracies.append(block_acc)
        total_correct += num_correct
        total_tokens += actual_block_size

        print(f"Block准确率: {block_acc:.4f} ({num_correct}/{actual_block_size})")

        # 显示前10个token的预测详情
        print(f"\n预测详情 (前{min(10, actual_block_size)}个token):")
        for i in range(min(10, actual_block_size)):
            pred_id = predicted_block[i].item()
            true_id = true_block[i].item()
            is_correct = "✅" if pred_id == true_id else "❌"
            pred_text = tokenizer.decode([pred_id])
            true_text = tokenizer.decode([true_id])
            print(f"  [{i}] {is_correct} pred={pred_id:6d} '{pred_text}' | true={true_id:6d} '{true_text}'")

        # Step 7: Teacher forcing - 用真实答案替换
        current_ids = torch.cat([current_ids, true_block.unsqueeze(0)], dim=1)

        print(f"Teacher forcing: 添加{actual_block_size}个真实token，当前长度={current_ids.size(1)}")

    # 统计总结
    print(f"\n{'='*80}")
    print(f"测试总结")
    print(f"{'='*80}")
    print(f"总blocks: {len(block_accuracies)}")
    print(f"总tokens: {total_tokens}")
    print(f"总正确: {total_correct}")
    print(f"总体准确率: {total_correct / total_tokens:.4f}")
    print(f"\n每个block的准确率:")
    for i, acc in enumerate(block_accuracies):
        print(f"  Block {i:3d}: {acc:.4f}")

    if len(block_accuracies) > 0:
        print(f"\n统计:")
        print(f"  平均准确率: {sum(block_accuracies) / len(block_accuracies):.4f}")
        print(f"  最高准确率: {max(block_accuracies):.4f}")
        print(f"  最低准确率: {min(block_accuracies):.4f}")

    return block_accuracies


def main():
    import pandas as pd
    import numpy as np

    # 配置
    MODEL_PATH = "dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
    DATA_PATH = "data/openr1.parquet"

    print("加载数据...")
    df = pd.read_parquet(DATA_PATH)
    sample = df.iloc[0]  # 使用第一个训练样本

    # 提取prompt和target（和训练时对齐）
    prompt_messages = sample['prompt']
    target_messages = sample['target']

    # 处理numpy arrays（和训练代码对齐）
    if isinstance(prompt_messages, np.ndarray):
        prompt_messages = prompt_messages.tolist() if prompt_messages.ndim == 0 else list(prompt_messages)
    if isinstance(target_messages, np.ndarray):
        target_messages = target_messages.tolist() if target_messages.ndim == 0 else list(target_messages)

    # 提取assistant的content作为ground truth（和训练代码对齐）
    ground_truth_content = None
    if isinstance(target_messages, (list, tuple)) and len(target_messages) > 0 and isinstance(target_messages[0], dict):
        ground_truth_content = target_messages[0].get("content", "")
    else:
        ground_truth_content = target_messages

    if ground_truth_content is None or ground_truth_content == "":
        print("错误：无法从数据中提取ground truth")
        return

    print(f"Prompt messages数量: {len(prompt_messages)}")
    print(f"Ground truth长度: {len(ground_truth_content)} 字符\n")

    print("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    print(f"✅ 模型加载完成\n")

    # 运行测试 - 限制在前50个blocks以节省时间
    teacher_forcing_test(
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
