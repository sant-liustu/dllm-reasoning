"""
测试两种不同的block_info构造方式对单block逐步生成的影响

方式1: block_info包含已生成的response
    block_info = [('real', len(已生成response)), ('mask', 3)]
    prompt_len = 真实的prompt长度

方式2: 把已生成的response当作prompt的延伸
    block_info = [('mask', 3)]
    prompt_len = len(prompt + 已生成response)

对比两种方式的准确率差异
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # 禁用torch.compile以支持FlexAttention

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json

# 导入数据集
sys.path.append('/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning')
from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset


def test_method_1(model, tokenizer, prompt_ids, full_ids, prompt_len, device):
    """
    方式1: block_info包含已生成的response
    block_info = [('real', len(已生成response)), ('mask', 3)]
    prompt_len = 真实的prompt长度
    """
    num_masks = 3
    mask_token_id = tokenizer.eos_token_id  # 使用EOS作为mask token

    # 开始：只有prompt
    current_generated = prompt_ids.clone()
    response_ids = full_ids[prompt_len:]

    # 统计信息
    total_mask_tokens = 0
    correct_mask_predictions = 0

    block_idx = 0
    block_size = 4
    total_blocks = min((response_ids.size(0) + block_size - 1) // block_size, 50)  # 最多测试50个block

    while block_idx < total_blocks:
        # Step 1: 添加mask tokens
        masks = torch.full((num_masks,), mask_token_id, dtype=torch.long, device=device)
        input_with_masks = torch.cat([current_generated, masks], dim=0)

        # Step 2: 构造position_ids (和训练时一样overlap)
        current_max_pos = current_generated.size(0) - 1
        mask_positions = torch.arange(current_max_pos, current_max_pos + num_masks, device=device)
        position_ids = torch.cat([
            torch.arange(current_generated.size(0), device=device),
            mask_positions
        ], dim=0)

        # Step 3: 构造block_info - 方式1
        # ⚠️ 包含已生成的response部分
        response_len = current_generated.size(0) - prompt_len
        if response_len > 0:
            block_info_for_model = [
                ('real', response_len),  # 已生成的response
                ('mask', num_masks),     # 新的mask block
            ]
        else:
            # 第一次生成，没有已生成的response
            block_info_for_model = [
                ('mask', num_masks),
            ]

        # Step 4: 准备输入
        input_ids_tensor = input_with_masks.unsqueeze(0)
        position_ids_tensor = position_ids.unsqueeze(0)

        # Step 5: 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids_tensor,
                position_ids=position_ids_tensor,
                block_info=[block_info_for_model],
                prompt_len=[prompt_len],  # ⚠️ 真实的prompt长度
                seq_lens=[input_with_masks.size(0)],
                use_cache=False
            )
            logits = outputs.logits

        # Step 6: 预测mask位置的token
        mask_start_idx = current_generated.size(0)
        mask_logits = logits[0, mask_start_idx:mask_start_idx + num_masks, :]
        predicted_tokens = torch.argmax(mask_logits, dim=-1)

        # Step 7: 获取ground truth并计算准确率
        gt_start = prompt_len + block_idx * 4 + 1  # Mask[i] predicts Real[i+1]
        ground_truth = sample['input_ids'][gt_start:gt_start + num_masks]

        correct = (predicted_tokens == ground_truth).sum().item()
        total_mask_tokens += num_masks
        correct_mask_predictions += correct

        # Step 8: Teacher forcing - 使用正确的label继续生成
        # 获取完整的real block (4 tokens)
        real_block_start = prompt_len + block_idx * 4
        real_block = sample['input_ids'][real_block_start:real_block_start + 4]
        current_generated = torch.cat([current_generated, real_block], dim=0)

        block_idx += 1

    accuracy = correct_mask_predictions / total_mask_tokens if total_mask_tokens > 0 else 0.0
    return accuracy


def test_method_2(model, tokenizer, sample, prompt_len, device):
    """
    方式2: 把已生成的response当作prompt的延伸
    block_info = [('mask', 3)]
    prompt_len = len(prompt + 已生成response)
    """
    model.train()  # 使用training mode以启用FlexAttention

    num_masks = 3
    mask_token_id = tokenizer.encode('<|mask|>', add_special_tokens=False)[0]

    # 开始：只有prompt
    input_ids = sample['input_ids'][:prompt_len]
    current_generated = input_ids.clone()

    # 统计信息
    total_mask_tokens = 0
    correct_mask_predictions = 0

    block_idx = 0
    total_blocks = (len(sample['input_ids']) - prompt_len) // 4

    while block_idx < total_blocks:
        # Step 1: 添加mask tokens
        masks = torch.full((num_masks,), mask_token_id, dtype=torch.long, device=device)
        input_with_masks = torch.cat([current_generated, masks], dim=0)

        # Step 2: 构造position_ids (和训练时一样overlap)
        current_max_pos = current_generated.size(0) - 1
        mask_positions = torch.arange(current_max_pos, current_max_pos + num_masks, device=device)
        position_ids = torch.cat([
            torch.arange(current_generated.size(0), device=device),
            mask_positions
        ], dim=0)

        # Step 3: 构造block_info - 方式2
        # ⚠️ 只有mask block，把已生成的都当作prompt
        block_info_for_model = [
            ('mask', num_masks),
        ]

        # Step 4: 准备输入
        input_ids_tensor = input_with_masks.unsqueeze(0)
        position_ids_tensor = position_ids.unsqueeze(0)

        # Step 5: 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids_tensor,
                position_ids=position_ids_tensor,
                block_info=[block_info_for_model],
                prompt_len=[current_generated.size(0)],  # ⚠️ 动态的prompt长度（包含已生成）
                seq_lens=[input_with_masks.size(0)],
                use_cache=False
            )
            logits = outputs.logits

        # Step 6: 预测mask位置的token
        mask_start_idx = current_generated.size(0)
        mask_logits = logits[0, mask_start_idx:mask_start_idx + num_masks, :]
        predicted_tokens = torch.argmax(mask_logits, dim=-1)

        # Step 7: 获取ground truth并计算准确率
        gt_start = prompt_len + block_idx * 4 + 1
        ground_truth = sample['input_ids'][gt_start:gt_start + num_masks]

        correct = (predicted_tokens == ground_truth).sum().item()
        total_mask_tokens += num_masks
        correct_mask_predictions += correct

        # Step 8: Teacher forcing - 使用正确的label继续生成
        real_block_start = prompt_len + block_idx * 4
        real_block = sample['input_ids'][real_block_start:real_block_start + 4]
        current_generated = torch.cat([current_generated, real_block], dim=0)

        block_idx += 1

    accuracy = correct_mask_predictions / total_mask_tokens if total_mask_tokens > 0 else 0.0
    return accuracy


def main():
    import pandas as pd
    import numpy as np

    # 加载数据（只加载第一个样本，避免内存问题）
    print("加载数据...")
    data_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"
    df = pd.read_parquet(data_path)
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

    print("✅ 数据加载完成\n")

    # 加载模型和tokenizer
    print("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # ⚠️ 关键：使用training mode以启用FlexAttention
    model.train()

    # Tokenize数据
    prompt_only_str = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )
    full_conversation_str = prompt_only_str + ground_truth_content + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    full_ids = tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

    prompt_len = prompt_ids.size(0)
    response_ids = full_ids[prompt_len:]

    print(f"\n样本信息:")
    print(f"  Prompt长度: {prompt_len}")
    print(f"  Response长度: {response_ids.size(0)}")
    print(f"  总序列长度: {full_ids.size(0)}")

    print("\n" + "="*80)
    print("测试方式1: block_info包含已生成的response")
    print("="*80)
    accuracy_method1 = test_method_1(model, tokenizer, sample, prompt_len, device)
    print(f"Mask准确率: {accuracy_method1:.4f} ({accuracy_method1*100:.2f}%)")

    print("\n" + "="*80)
    print("测试方式2: 把已生成的response当作prompt延伸")
    print("="*80)
    accuracy_method2 = test_method_2(model, tokenizer, sample, prompt_len, device)
    print(f"Mask准确率: {accuracy_method2:.4f} ({accuracy_method2*100:.2f}%)")

    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    print(f"方式1准确率: {accuracy_method1*100:.2f}%")
    print(f"方式2准确率: {accuracy_method2*100:.2f}%")
    print(f"差异: {(accuracy_method2 - accuracy_method1)*100:.2f}%")

    if accuracy_method1 > accuracy_method2:
        print("\n✅ 方式1更好 - block_info应该包含已生成的response")
    elif accuracy_method2 > accuracy_method1:
        print("\n✅ 方式2更好 - 应该把已生成的当作prompt延伸")
    else:
        print("\n两种方式效果相同")


if __name__ == "__main__":
    main()
