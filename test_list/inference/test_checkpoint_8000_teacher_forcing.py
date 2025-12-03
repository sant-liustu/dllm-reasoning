#!/usr/bin/env python3
"""
测试checkpoint 8000的三种Teacher Forcing场景:
1. 纯Causal (无mask): 标准自回归teacher forcing
2. Interleaved teacher forcing: 使用完整的interleaved mask,所有ground truth可见
3. Block-by-block teacher forcing: 逐block预测,预测后用正确答案替换再预测下一个block
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def test_pure_causal_teacher_forcing(model, tokenizer, prompt_ids, full_ids, prompt_len, device, num_test_tokens=100):
    """
    测试1: 纯Causal Teacher Forcing (无mask)
    使用真实tokens计算logits并统计准确率
    """
    print(f"\n{'='*80}")
    print(f"测试1: 纯Causal Teacher Forcing (无mask)")
    print(f"{'='*80}")

    response_ids = full_ids[prompt_len:]
    num_test = min(num_test_tokens, response_ids.size(0))

    print(f"Prompt长度: {prompt_len}")
    print(f"Response长度: {response_ids.size(0)}")
    print(f"测试前{num_test}个response tokens")

    # 前向传播 - 使用eval模式,不使用FlexAttention
    model.eval()
    with torch.no_grad():
        outputs = model(full_ids.unsqueeze(0), use_cache=False)
        logits = outputs.logits[0]  # [seq_len, vocab]

    predictions = logits.argmax(dim=-1)

    # 统计准确率
    num_correct = 0
    print(f"\n前20个token的预测详情:")
    for i in range(min(20, num_test)):
        pred_pos = prompt_len - 1 + i
        true_pos = prompt_len + i

        pred_id = predictions[pred_pos].item()
        true_id = full_ids[true_pos].item()

        is_correct = pred_id == true_id
        if is_correct:
            num_correct += 1

        if i < 20:  # 只显示前20个
            status = "✅" if is_correct else "❌"
            pred_text = tokenizer.decode([pred_id])
            true_text = tokenizer.decode([true_id])
            print(f"  [{i:2d}] {status} pred={pred_id:6d} '{pred_text}' | true={true_id:6d} '{true_text}'")

    # 计算完整准确率
    for i in range(20, num_test):
        pred_pos = prompt_len - 1 + i
        true_pos = prompt_len + i
        if predictions[pred_pos].item() == full_ids[true_pos].item():
            num_correct += 1

    accuracy = num_correct / num_test
    print(f"\n纯Causal Teacher Forcing准确率: {accuracy:.4f} ({num_correct}/{num_test})")

    return accuracy


def test_interleaved_teacher_forcing(model, tokenizer, prompt_ids, full_ids, prompt_len, device, block_size=4, max_blocks=50):
    """
    测试2: Interleaved Teacher Forcing
    使用完整的interleaved mask,所有ground truth可见
    """
    print(f"\n{'='*80}")
    print(f"测试2: Interleaved Teacher Forcing (完整mask)")
    print(f"{'='*80}")

    response_ids = full_ids[prompt_len:]
    response_len = response_ids.size(0)

    # 计算block数量
    num_blocks = min((response_len + block_size - 1) // block_size, max_blocks)
    actual_test_len = min(num_blocks * block_size, response_len)

    print(f"Prompt长度: {prompt_len}")
    print(f"Response长度: {response_len}")
    print(f"Block size: {block_size}")
    print(f"测试blocks: {num_blocks}")
    print(f"实际测试tokens: {actual_test_len}")

    # 构造block_info - 完整的interleaved格式
    block_info = []
    for i in range(num_blocks):
        block_info.append(('real', 1))   # Real token
        block_info.append(('mask', block_size - 1))  # Mask tokens

    # 准备input: prompt + response (实际的response tokens)
    test_ids = torch.cat([prompt_ids, response_ids[:actual_test_len]], dim=0)
    position_ids = torch.arange(test_ids.size(0), device=device)

    # 前向传播 - 使用train模式以启用FlexAttention
    model.train()
    with torch.no_grad():
        outputs = model(
            test_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            block_info=[block_info],
            prompt_len=[prompt_len],
            seq_lens=[test_ids.size(0)],
            use_cache=False
        )
        logits = outputs.logits[0]  # [seq_len, vocab]

    predictions = logits.argmax(dim=-1)

    # 统计mask位置的准确率
    num_correct = 0
    num_total = 0

    print(f"\n前5个block的预测详情:")
    for block_idx in range(min(5, num_blocks)):
        block_start = prompt_len + block_idx * block_size
        real_pos = block_start  # Real token位置

        print(f"\n  Block {block_idx}:")
        # Mask预测: mask[0]预测real[1], mask[1]预测real[2], mask[2]预测real[3]
        for j in range(block_size - 1):
            mask_pos = real_pos + j  # 当前mask的位置
            pred_target_offset = j + 1  # 预测目标相对于real的偏移

            if block_start + pred_target_offset < prompt_len + actual_test_len:
                pred_id = predictions[mask_pos].item()
                true_id = response_ids[block_idx * block_size + pred_target_offset].item()

                is_correct = pred_id == true_id
                if is_correct:
                    num_correct += 1
                num_total += 1

                if block_idx < 5:  # 只显示前5个block
                    status = "✅" if is_correct else "❌"
                    pred_text = tokenizer.decode([pred_id])
                    true_text = tokenizer.decode([true_id])
                    print(f"    Mask[{j}]→Real[{pred_target_offset}]: {status} pred={pred_id:6d} '{pred_text}' | true={true_id:6d} '{true_text}'")

    # 计算剩余blocks的准确率
    for block_idx in range(5, num_blocks):
        block_start = prompt_len + block_idx * block_size
        real_pos = block_start

        for j in range(block_size - 1):
            mask_pos = real_pos + j
            pred_target_offset = j + 1

            if block_start + pred_target_offset < prompt_len + actual_test_len:
                pred_id = predictions[mask_pos].item()
                true_id = response_ids[block_idx * block_size + pred_target_offset].item()

                if pred_id == true_id:
                    num_correct += 1
                num_total += 1

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    print(f"\nInterleaved Teacher Forcing准确率: {accuracy:.4f} ({num_correct}/{num_total})")

    return accuracy


def test_block_by_block_teacher_forcing(model, tokenizer, prompt_ids, full_ids, prompt_len, device, block_size=4, max_blocks=50):
    """
    测试3: Block-by-block Teacher Forcing
    逐block预测,预测后用正确答案替换再预测下一个block
    """
    print(f"\n{'='*80}")
    print(f"测试3: Block-by-block Teacher Forcing (逐块预测)")
    print(f"{'='*80}")

    response_ids = full_ids[prompt_len:]
    response_len = response_ids.size(0)
    num_blocks = min((response_len + block_size - 1) // block_size, max_blocks)

    print(f"Prompt长度: {prompt_len}")
    print(f"Response长度: {response_len}")
    print(f"Block size: {block_size}")
    print(f"测试blocks: {num_blocks}")

    eos_token_id = tokenizer.eos_token_id
    num_masks = block_size - 1

    current_ids = prompt_ids.clone()

    num_correct = 0
    num_total = 0
    block_accuracies = []

    model.train()  # 使用train模式以启用FlexAttention

    print(f"\n前5个block的预测详情:")
    for block_idx in range(num_blocks):
        block_start_in_response = block_idx * block_size
        block_end_in_response = min((block_idx + 1) * block_size, response_len)
        actual_block_size = block_end_in_response - block_start_in_response

        if actual_block_size == 0:
            break

        true_block = response_ids[block_start_in_response:block_end_in_response]

        # 添加mask tokens
        mask_block = torch.full((num_masks,), eos_token_id, device=device, dtype=current_ids.dtype)
        input_with_masks = torch.cat([current_ids, mask_block], dim=0)

        # 构造block_info
        response_so_far = current_ids.size(0) - prompt_len
        if response_so_far > 0:
            # 已经有生成的response,把它们当作prompt延伸
            block_info = [('mask', num_masks)]
            effective_prompt_len = current_ids.size(0)
        else:
            # 第一个block
            block_info = [('mask', num_masks)]
            effective_prompt_len = prompt_len

        position_ids = torch.arange(input_with_masks.size(0), device=device)

        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_with_masks.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
                block_info=[block_info],
                prompt_len=[effective_prompt_len],
                seq_lens=[input_with_masks.size(0)],
                use_cache=False
            )
            logits = outputs.logits[0]

        # 预测: mask位置预测后续tokens
        mask_start = current_ids.size(0)
        mask_logits = logits[mask_start:mask_start + num_masks, :]
        predicted_tokens = mask_logits.argmax(dim=-1)

        # 对比准确率 - mask预测的是接下来的tokens (真实block的[1:])
        compare_len = min(num_masks, actual_block_size - 1)
        if compare_len > 0:
            true_targets = true_block[1:1+compare_len]
            pred_targets = predicted_tokens[:compare_len]
            correct_mask = (pred_targets == true_targets)
            block_correct = correct_mask.sum().item()

            num_correct += block_correct
            num_total += compare_len
            block_acc = block_correct / compare_len
            block_accuracies.append(block_acc)

            if block_idx < 5:
                print(f"\n  Block {block_idx}: 准确率 {block_acc:.4f} ({block_correct}/{compare_len})")
                for j in range(compare_len):
                    pred_id = pred_targets[j].item()
                    true_id = true_targets[j].item()
                    is_correct = pred_id == true_id
                    status = "✅" if is_correct else "❌"
                    pred_text = tokenizer.decode([pred_id])
                    true_text = tokenizer.decode([true_id])
                    print(f"    Mask[{j}]: {status} pred={pred_id:6d} '{pred_text}' | true={true_id:6d} '{true_text}'")

        # Teacher forcing: 用真实答案替换
        current_ids = torch.cat([current_ids, true_block], dim=0)

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    print(f"\nBlock-by-block Teacher Forcing准确率: {accuracy:.4f} ({num_correct}/{num_total})")

    if len(block_accuracies) > 0:
        print(f"  平均block准确率: {sum(block_accuracies) / len(block_accuracies):.4f}")
        print(f"  最高block准确率: {max(block_accuracies):.4f}")
        print(f"  最低block准确率: {min(block_accuracies):.4f}")

    return accuracy


def main():
    # 配置
    MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft_1202/global_step_8000/huggingface"
    DATA_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"

    print("="*80)
    print("测试Checkpoint 8000 - 三种Teacher Forcing场景")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
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

    # 去掉<think>标签
    if ground_truth_content.strip().startswith('<think>'):
        think_start = ground_truth_content.find('<think>')
        ground_truth_content = ground_truth_content[think_start + 7:].lstrip()

    print("✅ 数据加载完成")

    # 加载模型
    print("\n加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    print("✅ 模型加载完成")

    # Tokenize
    prompt_only_str = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )
    full_conversation_str = prompt_only_str + ground_truth_content + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    full_ids = tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

    prompt_len = prompt_ids.size(0)
    response_len = full_ids.size(0) - prompt_len

    print(f"\n序列信息:")
    print(f"  Prompt长度: {prompt_len}")
    print(f"  Response长度: {response_len}")
    print(f"  总长度: {full_ids.size(0)}")

    # 运行三个测试
    acc1 = test_pure_causal_teacher_forcing(model, tokenizer, prompt_ids, full_ids, prompt_len, device, num_test_tokens=100)
    acc2 = test_interleaved_teacher_forcing(model, tokenizer, prompt_ids, full_ids, prompt_len, device, block_size=4, max_blocks=50)
    acc3 = test_block_by_block_teacher_forcing(model, tokenizer, prompt_ids, full_ids, prompt_len, device, block_size=4, max_blocks=50)

    # 总结
    print(f"\n{'='*80}")
    print(f"测试总结")
    print(f"{'='*80}")
    print(f"1. 纯Causal Teacher Forcing:         {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"2. Interleaved Teacher Forcing:      {acc2:.4f} ({acc2*100:.2f}%)")
    print(f"3. Block-by-block Teacher Forcing:   {acc3:.4f} ({acc3*100:.2f}%)")

    print(f"\n解读:")
    print(f"- 如果acc1很高(>0.9): 说明模型基础能力正常,纯causal预测准确")
    print(f"- 如果acc2很高(>0.9): 说明模型interleaved格式学习正常")
    print(f"- 如果acc3很高(>0.9): 说明模型逐块生成能力正常")
    print(f"- 如果acc1高但acc2/acc3低: 可能是mask机制的问题")
    print(f"- 如果所有都低(<0.5): 说明checkpoint训练不充分")


if __name__ == "__main__":
    main()
