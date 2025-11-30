#!/usr/bin/env python3
"""
测试FlexAttention是否真正隔离了Mask和Real tokens
通过对比两个forward pass的hidden states来验证
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import create_interleaved_training_data
from dllm_reasoning.trainer.flex_attention_utils import create_block_mask_from_batch

MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
DATA_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*100)
print("测试FlexAttention隔离性")
print("="*100)

# ========== 加载模型 ==========
print("\n📦 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).to(device).train()  # Training mode for FlexAttention

print(f"   ✅ 模型加载完成")

# ========== 准备数据 ==========
print("\n📂 准备测试数据...")
df = pd.read_parquet(DATA_PATH)
sample = df.iloc[0]

prompt = sample['prompt']
target = sample['target']

# 处理
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

# 完整对话
full_messages = messages + [{"role": "assistant", "content": target_content}]

# Tokenize
input_ids = tokenizer.apply_chat_template(
    full_messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)[0].to(device)

prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)[0].to(device)

prompt_len = prompt_ids.size(0)
loss_mask = torch.ones(input_ids.size(0), dtype=torch.long, device=device)

# 创建interleaved格式
interleaved_ids, position_ids, interleaved_loss_mask, block_info = create_interleaved_training_data(
    input_ids=input_ids,
    loss_mask=loss_mask,
    block_size=4,
    mask_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

print(f"   Prompt长度: {prompt_len}")
print(f"   Interleaved序列长度: {interleaved_ids.size(0)}")
print(f"   Block info (前3个): {block_info[:3]}")

# 转换block_info格式: (seg_type, list_idx, seg_len) -> (seg_type, seg_len)
block_info_converted = [(seg_type, seg_len) for seg_type, _, seg_len in block_info]
print(f"   Block info converted (前3个): {block_info_converted[:3]}")

# 为了避免序列过长,只测试第一个block: Prompt + Mask×3 + Real×4
first_block_end = prompt_len + 3 + 4  # 277 + 3 + 4 = 284
interleaved_ids_short = interleaved_ids[:first_block_end]
position_ids_short = position_ids[:first_block_end]
block_info_short = block_info_converted[:2]  # 只有第一个mask和第一个real

print(f"\n仅测试第一个block以避免内存问题:")
print(f"   截断长度: {first_block_end}")
print(f"   Block info: {block_info_short}")

# ========== 测试1: 完整序列 (训练时的情况) ==========
print("\n" + "="*100)
print("测试1: 完整序列 [Prompt + Mask×3 + Real×4]")
print("="*100)

# 构造batch格式
batch = {
    "input_ids": interleaved_ids_short.unsqueeze(0),
    "position_ids": position_ids_short.unsqueeze(0),
    "block_info": [block_info_short],
    "prompt_len": [prompt_len],
    "seq_lens": [interleaved_ids_short.size(0)],
}

# 创建FlexAttention mask
block_mask_full = create_block_mask_from_batch(batch, device)

# Forward pass
with torch.no_grad():
    outputs_full = model(
        interleaved_ids_short.unsqueeze(0),
        position_ids=position_ids_short.unsqueeze(0),
        attention_mask=block_mask_full,
        use_cache=False,
        output_hidden_states=True
    )

logits_full = outputs_full.logits[0]  # [seq_len, vocab]
hidden_states_full = outputs_full.hidden_states[-1][0]  # 最后一层的hidden states [seq_len, hidden_dim]

print(f"   Logits shape: {logits_full.shape}")
print(f"   Hidden states shape: {hidden_states_full.shape}")

# 提取Mask positions的hidden states和logits
mask_start = prompt_len
mask_end = mask_start + 3

mask_hidden_full = hidden_states_full[mask_start:mask_end]  # [3, hidden_dim]
mask_logits_full = logits_full[mask_start:mask_end]  # [3, vocab]

print(f"\n   Mask positions: [{mask_start}:{mask_end}]")
print(f"   Mask hidden states shape: {mask_hidden_full.shape}")
print(f"   Mask logits shape: {mask_logits_full.shape}")

# Mask的预测
mask_predictions_full = mask_logits_full.argmax(dim=-1)
print(f"   Mask predictions: {mask_predictions_full.tolist()}")
print(f"   Mask predictions (text): {[tokenizer.decode([t]) for t in mask_predictions_full]}")

# ========== 测试2: 截断序列 (只有Prompt + Mask×3,推理时的情况) ==========
print("\n" + "="*100)
print("测试2: 截断序列 [Prompt + Mask×3] (模拟推理)")
print("="*100)

# 只保留Prompt和第一组Mask
truncated_ids = interleaved_ids[:mask_end].clone()
truncated_position_ids = position_ids[:mask_end].clone()

print(f"   截断序列长度: {truncated_ids.size(0)}")
print(f"   = Prompt({prompt_len}) + Mask(3)")

# 为截断序列创建FlexAttention mask
# 注意: 这里的block_info只包含prompt和第一个mask group
truncated_block_info = [('mask', 0, 3)]  # 只有一个mask segment

batch_truncated = {
    "input_ids": truncated_ids.unsqueeze(0),
    "position_ids": truncated_position_ids.unsqueeze(0),
    "block_info": [truncated_block_info],
    "prompt_len": [prompt_len],
    "seq_lens": [truncated_ids.size(0)],
}

block_mask_truncated = create_block_mask_from_batch(batch_truncated, device)

# Forward pass
with torch.no_grad():
    outputs_truncated = model(
        truncated_ids.unsqueeze(0),
        position_ids=truncated_position_ids.unsqueeze(0),
        attention_mask=block_mask_truncated,
        use_cache=False,
        output_hidden_states=True
    )

logits_truncated = outputs_truncated.logits[0]  # [mask_end, vocab]
hidden_states_truncated = outputs_truncated.hidden_states[-1][0]  # [mask_end, hidden_dim]

# Mask positions的hidden states和logits
mask_hidden_truncated = hidden_states_truncated[mask_start:mask_end]
mask_logits_truncated = logits_truncated[mask_start:mask_end]

mask_predictions_truncated = mask_logits_truncated.argmax(dim=-1)
print(f"   Mask predictions: {mask_predictions_truncated.tolist()}")
print(f"   Mask predictions (text): {[tokenizer.decode([t]) for t in mask_predictions_truncated]}")

# ========== 对比结果 ==========
print("\n" + "="*100)
print("对比结果")
print("="*100)

# 计算hidden states的差异
hidden_diff = torch.abs(mask_hidden_full - mask_hidden_truncated)
hidden_max_diff = hidden_diff.max().item()
hidden_mean_diff = hidden_diff.mean().item()

print(f"\nMask positions的Hidden States差异:")
print(f"   最大差异: {hidden_max_diff:.6f}")
print(f"   平均差异: {hidden_mean_diff:.6f}")

# 计算logits的差异
logits_diff = torch.abs(mask_logits_full - mask_logits_truncated)
logits_max_diff = logits_diff.max().item()
logits_mean_diff = logits_diff.mean().item()

print(f"\nMask positions的Logits差异:")
print(f"   最大差异: {logits_max_diff:.6f}")
print(f"   平均差异: {logits_mean_diff:.6f}")

# 预测是否相同
predictions_match = torch.equal(mask_predictions_full, mask_predictions_truncated)
print(f"\n预测结果是否完全相同: {predictions_match}")
if not predictions_match:
    print(f"   完整序列预测: {mask_predictions_full.tolist()}")
    print(f"   截断序列预测: {mask_predictions_truncated.tolist()}")

# ========== 结论 ==========
print("\n" + "="*100)
print("结论")
print("="*100)

if hidden_max_diff < 1e-4 and predictions_match:
    print("✅ FlexAttention完美隔离了Mask和Real tokens!")
    print("   训练时有Real tokens和推理时没有Real tokens的结果完全一致")
    print("   问题不在隔离性上,需要寻找其他原因")
elif hidden_max_diff < 1e-2:
    print("⚠️  有轻微差异,但基本隔离")
    print(f"   Hidden states最大差异: {hidden_max_diff:.6f}")
    print("   这可能是数值精度问题")
else:
    print("❌ 隔离失败! FlexAttention mask没有正确隔离Mask和Real tokens")
    print(f"   Hidden states最大差异: {hidden_max_diff:.6f}")
    print("   这说明训练时的mask实现有问题!")
