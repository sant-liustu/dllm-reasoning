#!/usr/bin/env python3
"""
测试逐步生成模式下的Mask预测准确率

模式：
1. 已生成的token: [Prompt][R1][R2]...
2. 拼接3个Mask: [Prompt][R1][R2]...[M][M][M]
3. 预测这3个Mask位置 -> 得到预测的[R_next]
4. Teacher forcing: 用正确的token替换 -> [Prompt][R1][R2]...[R_correct]
5. 继续下一个block

这模拟了真实的生成场景，而不是训练时的交错格式。
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def test_sequential_generation(
    model,
    tokenizer,
    prompt_messages: list,
    ground_truth_content: str,
    block_size: int = 4,
    max_blocks: int = 50,
    device: str = "cuda",
):
    """
    测试逐步生成模式下的Mask预测准确率

    Args:
        model: 模型
        tokenizer: tokenizer
        prompt_messages: prompt消息列表
        ground_truth_content: 正确答案
        block_size: block大小（默认4）
        max_blocks: 最多测试多少个block
        device: 设备
    """
    # 处理<think>标签
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
    num_masks = block_size - 1  # 3个mask

    print(f"{'='*80}")
    print(f"逐步生成模式测试（Sequential Generation with Teacher Forcing）")
    print(f"{'='*80}")
    print(f"Prompt长度: {prompt_len} tokens")
    print(f"Response长度: {response_len} tokens")
    print(f"Block size: {block_size}")
    print(f"Masks per block: {num_masks}")
    print(f"Max blocks: {max_blocks}")
    print(f"{'='*80}\n")

    # 统计信息
    all_mask_predictions = []  # 每个mask位置的预测结果
    total_mask_correct = 0
    total_mask_tokens = 0

    # 初始化：从prompt开始
    current_generated = prompt_ids.clone()  # [prompt_len]

    num_blocks = min((response_len + block_size - 1) // block_size, max_blocks)

    print(f"开始逐步生成测试，共{num_blocks}个block\n")

    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, response_len)
        true_block_tokens = response_ids[block_start:block_end]  # 正确答案
        actual_block_size = true_block_tokens.size(0)

        if actual_block_size == 0:
            break

        # Step 1: 拼接3个Mask token
        mask_tokens = torch.full((num_masks,), eos_token_id, dtype=torch.long, device=device)
        input_with_masks = torch.cat([current_generated, mask_tokens], dim=0)

        # Step 2: 构造position_ids（连续递增）
        position_ids = torch.arange(input_with_masks.size(0), device=device)

        # Step 3: 准备输入
        input_ids = input_with_masks.unsqueeze(0)  # [1, seq_len]
        position_ids_input = position_ids.unsqueeze(0)  # [1, seq_len]

        # Step 4: 前向传播（使用标准causal mask，不是FlexAttention）
        # 因为这是真实生成场景，不是训练时的交错格式
        with torch.no_grad():
            outputs = model(
                input_ids,
                position_ids=position_ids_input,
                use_cache=False
            )
            logits = outputs.logits  # [1, seq_len, vocab]

        # Step 5: 获取Mask位置的预测
        # Mask位置是最后num_masks个位置
        mask_start_pos = current_generated.size(0)
        mask_predictions = []
        mask_correct_count = 0

        for i in range(num_masks):
            mask_pos = mask_start_pos + i
            # logits[0, mask_pos]预测的是下一个token
            # 所以logits[0, mask_pos]对应的是true_block_tokens[i+1]
            pred_id = logits[0, mask_pos].argmax(dim=-1).item()

            # 目标token: Mask[i]预测Real[i+1]
            target_idx = i + 1
            if target_idx < len(true_block_tokens):
                true_id = true_block_tokens[target_idx].item()
                is_correct = (pred_id == true_id)

                mask_predictions.append({
                    'block_idx': block_idx,
                    'mask_idx': i,
                    'pred_id': pred_id,
                    'true_id': true_id,
                    'correct': is_correct,
                })

                if is_correct:
                    mask_correct_count += 1
                    total_mask_correct += 1
                total_mask_tokens += 1

        # 统计这个block的准确率
        block_acc = mask_correct_count / num_masks if num_masks > 0 else 0.0

        # Step 6: Teacher forcing - 用正确的token替换
        # 添加真实的block tokens到已生成序列
        current_generated = torch.cat([current_generated, true_block_tokens], dim=0)

        # 打印前几个block的详情
        if block_idx < 5:
            print(f"Block {block_idx}: 准确率 = {block_acc:.4f} ({mask_correct_count}/{num_masks})")
            for mp in mask_predictions:
                is_correct = "✅" if mp['correct'] else "❌"
                pred_text = tokenizer.decode([mp['pred_id']])
                true_text = tokenizer.decode([mp['true_id']])
                print(f"  Mask[{mp['mask_idx']}]: {is_correct} pred={mp['pred_id']:6d} '{pred_text}' | "
                      f"true={mp['true_id']:6d} '{true_text}'")

        all_mask_predictions.extend(mask_predictions)

    # 总结
    print(f"\n{'='*80}")
    print(f"测试总结")
    print(f"{'='*80}")
    print(f"总blocks: {num_blocks}")
    print(f"总Mask tokens: {total_mask_tokens}")
    print(f"总正确: {total_mask_correct}")
    print(f"总体Mask准确率: {total_mask_correct / total_mask_tokens if total_mask_tokens > 0 else 0:.4f}")
    print(f"{'='*80}")

    return all_mask_predictions


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

    # ⚠️ 使用eval模式（这是真实的生成场景）
    model.eval()

    print(f"✅ 模型加载完成（eval mode - 标准生成模式）\n")

    # 运行测试
    test_sequential_generation(
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
