#!/usr/bin/env python3
"""
测试多个样本的准确率分布
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# 禁用torch.compile
import torch._dynamo
torch._dynamo.config.disable = True

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import create_interleaved_training_data, create_interleaved_labels
from dllm_reasoning.trainer.flex_attention_utils import create_block_mask_from_batch

MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
DATA_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"

print("="*100)
print("测试多个样本的准确率")
print("="*100)

# ========== 加载模型 ==========
print("\n📦 加载模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).to(device).train()  # 使用train模式，和训练时一致

print(f"   ✅ 模型加载完成 (train mode)")

# ========== 加载数据 ==========
print(f"\n📂 加载训练数据: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"   总样本数: {len(df)}")

# ========== 测试前20个样本 ==========
NUM_SAMPLES = 20
print(f"\n🔍 测试前{NUM_SAMPLES}个样本:")

results = []

for sample_idx in range(NUM_SAMPLES):
    sample = df.iloc[sample_idx]

    # 处理数据 - 对齐训练逻辑
    prompt = sample['prompt']
    target = sample['target']

    messages = list(prompt) if isinstance(prompt, (list, tuple)) else prompt
    messages = messages.tolist() if hasattr(messages, 'tolist') else messages

    target_list = target.tolist() if hasattr(target, 'tolist') else target
    if isinstance(target_list, list) and len(target_list) > 0:
        target_content = target_list[0].get('content', '') if isinstance(target_list[0], dict) else str(target_list[0])
    else:
        target_content = str(target_list)

    # 去掉<think>
    if target_content.strip().startswith('<think>'):
        think_start = target_content.find('<think>')
        target_content = target_content[think_start + 7:].lstrip()

    # Tokenize - 对齐训练逻辑
    prompt_only_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full_conversation_str = prompt_only_str + target_content + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    input_ids = tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

    prompt_len = prompt_ids.size(0)

    # ========== 重要：对齐训练时的max_length truncation ==========
    MAX_LENGTH = 2048
    if input_ids.size(0) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]  # right truncation
        # 重新调整prompt_len如果被截断
        if prompt_len > MAX_LENGTH:
            prompt_len = MAX_LENGTH

    loss_mask = torch.ones(input_ids.size(0), dtype=torch.long, device=device)
    loss_mask[:prompt_len] = 0

    # 创建interleaved格式
    interleaved_ids, position_ids, interleaved_loss_mask, block_info = create_interleaved_training_data(
        input_ids=input_ids,
        loss_mask=loss_mask,
        block_size=4,
        mask_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 创建labels
    labels = create_interleaved_labels(
        original_input_ids=input_ids,
        interleaved_input_ids=interleaved_ids,
        interleaved_loss_mask=interleaved_loss_mask,
        block_size=4,
        prompt_len=prompt_len,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 转换block_info格式
    block_info_converted = [(seg_type, seg_len) for seg_type, _, seg_len in block_info]

    # Forward pass
    batch = {
        "input_ids": interleaved_ids.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
        "block_info": [block_info_converted],
        "prompt_len": [prompt_len],
        "seq_lens": [interleaved_ids.size(0)],
    }

    block_mask = create_block_mask_from_batch(batch, device)

    with torch.no_grad():
        outputs = model(
            interleaved_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            attention_mask=block_mask,
            use_cache=False
        )
        logits = outputs.logits[0]

    # 计算准确率
    predictions = logits.argmax(dim=-1)
    valid_mask = (labels != -100)

    # 分类统计
    mask_positions = []
    real_positions = []

    current_pos = prompt_len
    for seg_type, _, seg_len in block_info:
        if seg_type == 'mask':
            mask_positions.extend(range(current_pos, current_pos + seg_len))
        elif seg_type == 'real':
            real_positions.extend(range(current_pos, current_pos + seg_len))
        current_pos += seg_len

    # Prompt[-1]
    prompt_last_correct = 0
    prompt_last_total = 0
    if labels[prompt_len - 1].item() != -100:
        prompt_last_total = 1
        if predictions[prompt_len - 1].item() == labels[prompt_len - 1].item():
            prompt_last_correct = 1

    # Mask positions
    mask_positions_tensor = torch.tensor(mask_positions, device=device)
    mask_labels = labels[mask_positions_tensor]
    mask_valid = mask_labels != -100
    mask_total = mask_valid.sum().item()
    mask_correct = ((predictions[mask_positions_tensor] == mask_labels) & mask_valid).sum().item()

    # Real positions
    real_positions_tensor = torch.tensor(real_positions, device=device)
    real_labels = labels[real_positions_tensor]
    real_valid = real_labels != -100
    real_total = real_valid.sum().item()
    real_correct = ((predictions[real_positions_tensor] == real_labels) & real_valid).sum().item()

    # 总体
    total_valid = valid_mask.sum().item()
    total_correct = ((predictions == labels) & valid_mask).sum().item()

    overall_acc = total_correct / total_valid if total_valid > 0 else 0
    mask_acc = mask_correct / mask_total if mask_total > 0 else 0
    real_acc = real_correct / real_total if real_total > 0 else 0
    prompt_acc = prompt_last_correct / prompt_last_total if prompt_last_total > 0 else 0

    results.append({
        'sample_idx': sample_idx,
        'overall_acc': overall_acc,
        'mask_acc': mask_acc,
        'real_acc': real_acc,
        'prompt_acc': prompt_acc,
        'mask_total': mask_total,
        'real_total': real_total,
        'total_valid': total_valid,
    })

    print(f"\n样本 {sample_idx:2d}:")
    print(f"  总体准确率: {overall_acc:.4f} ({total_correct}/{total_valid})")
    print(f"  Prompt[-1]: {prompt_acc:.4f} ({prompt_last_correct}/{prompt_last_total})")
    print(f"  Mask:       {mask_acc:.4f} ({mask_correct}/{mask_total})")
    print(f"  Real:       {real_acc:.4f} ({real_correct}/{real_total})")

# ========== 统计汇总 ==========
print(f"\n" + "="*100)
print("汇总统计")
print("="*100)

import numpy as np

overall_accs = [r['overall_acc'] for r in results]
mask_accs = [r['mask_acc'] for r in results]
real_accs = [r['real_acc'] for r in results]
prompt_accs = [r['prompt_acc'] for r in results]

print(f"\n总体准确率:")
print(f"  平均: {np.mean(overall_accs):.4f}")
print(f"  中位数: {np.median(overall_accs):.4f}")
print(f"  最小: {np.min(overall_accs):.4f}")
print(f"  最大: {np.max(overall_accs):.4f}")

print(f"\nMask准确率:")
print(f"  平均: {np.mean(mask_accs):.4f}")
print(f"  中位数: {np.median(mask_accs):.4f}")
print(f"  最小: {np.min(mask_accs):.4f}")
print(f"  最大: {np.max(mask_accs):.4f}")

print(f"\nReal准确率:")
print(f"  平均: {np.mean(real_accs):.4f}")
print(f"  中位数: {np.median(real_accs):.4f}")
print(f"  最小: {np.min(real_accs):.4f}")
print(f"  最大: {np.max(real_accs):.4f}")

print(f"\nPrompt[-1]准确率:")
print(f"  平均: {np.mean(prompt_accs):.4f}")

# 加权平均（按token数量）
total_mask_tokens = sum(r['mask_total'] for r in results)
total_real_tokens = sum(r['real_total'] for r in results)
total_tokens = sum(r['total_valid'] for r in results)

weighted_mask_acc = sum(r['mask_acc'] * r['mask_total'] for r in results) / total_mask_tokens if total_mask_tokens > 0 else 0
weighted_real_acc = sum(r['real_acc'] * r['real_total'] for r in results) / total_real_tokens if total_real_tokens > 0 else 0
weighted_overall_acc = sum(r['overall_acc'] * r['total_valid'] for r in results) / total_tokens if total_tokens > 0 else 0

print(f"\n加权平均准确率（按token数量）:")
print(f"  总体: {weighted_overall_acc:.4f}")
print(f"  Mask: {weighted_mask_acc:.4f}")
print(f"  Real: {weighted_real_acc:.4f}")
