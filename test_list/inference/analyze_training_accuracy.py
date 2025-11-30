#!/usr/bin/env python3
"""
分析训练时Mask positions vs Real positions的准确率
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
print("分析训练准确率组成: Mask vs Real positions")
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
).to(device).train()

print(f"   ✅ 模型加载完成")

# ========== 加载训练数据 ==========
print(f"\n📂 加载训练数据: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
sample = df.iloc[0]

# ========== 处理数据 - 对齐训练逻辑 ==========
# 完全复现 interleaved_sft_dataset.py:479-516 的处理逻辑
prompt = sample['prompt']
target = sample['target']

messages = list(prompt) if isinstance(prompt, (list, tuple)) else prompt
messages = messages.tolist() if hasattr(messages, 'tolist') else messages

target_list = target.tolist() if hasattr(target, 'tolist') else target
if isinstance(target_list, list) and len(target_list) > 0:
    target_content = target_list[0].get('content', '') if isinstance(target_list[0], dict) else str(target_list[0])
else:
    target_content = str(target_list)

# 去掉<think> - 对齐训练数据处理
if target_content.strip().startswith('<think>'):
    think_start = target_content.find('<think>')
    target_content = target_content[think_start + 7:].lstrip()

# ========== 关键：使用和训练完全一致的tokenization方式 ==========
# 见 interleaved_sft_dataset.py:502-516
prompt_only_str = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)

full_conversation_str = prompt_only_str + target_content + tokenizer.eos_token

prompt_ids = tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
input_ids = tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

prompt_len = prompt_ids.size(0)
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

print(f"\n📝 数据信息:")
print(f"   Prompt长度: {prompt_len}")
print(f"   Interleaved序列长度: {interleaved_ids.size(0)}")
print(f"   Block数量: {len(block_info)}")

# 转换block_info格式
block_info_converted = [(seg_type, seg_len) for seg_type, _, seg_len in block_info]

# ========== Forward pass 计算完整序列的logits ==========
print(f"\n🚀 Forward pass...")

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
    logits = outputs.logits[0]  # [seq_len, vocab]

print(f"   Logits shape: {logits.shape}")

# ========== 分析各个位置的准确率 ==========
print(f"\n📊 分析各位置类型的准确率:")

predictions = logits.argmax(dim=-1)
valid_mask = (labels != -100)

# 统计
prompt_last_positions = []
mask_positions = []
real_positions = []

# ========== 重要修正：block_info的位置是从prompt_len开始的！ ==========
# 根据block_info确定每个位置的类型
current_pos = prompt_len  # 从prompt结束位置开始！
for seg_type, _, seg_len in block_info:
    if seg_type == 'mask':
        mask_positions.extend(range(current_pos, current_pos + seg_len))
    elif seg_type == 'real':
        real_positions.extend(range(current_pos, current_pos + seg_len))
    current_pos += seg_len

# Prompt最后一个位置
if labels[prompt_len - 1].item() != -100:
    prompt_last_positions.append(prompt_len - 1)

print(f"\n位置统计:")
print(f"   Prompt[-1] 有label的位置数: {len(prompt_last_positions)}")
print(f"   Mask positions: {len(mask_positions)}")
print(f"   Real positions: {len(real_positions)}")
print(f"   总计有label的位置: {valid_mask.sum().item()}")

# 计算各部分准确率
def calc_accuracy(positions, name):
    if len(positions) == 0:
        print(f"\n{name}: 无数据")
        return

    positions_tensor = torch.tensor(positions, device=device)
    preds = predictions[positions_tensor]
    labels_subset = labels[positions_tensor]
    valid = labels_subset != -100

    if valid.sum() == 0:
        print(f"\n{name}: 无有效label")
        return

    correct = (preds == labels_subset) & valid
    accuracy = correct.sum().float() / valid.sum().float()

    print(f"\n{name}:")
    print(f"   有效位置数: {valid.sum().item()}")
    print(f"   准确数: {correct.sum().item()}")
    print(f"   准确率: {accuracy.item():.4f}")

    # 显示前5个位置的预测详情
    print(f"   前5个位置详情:")
    for i in range(min(5, len(positions))):
        pos = positions[i]
        pred_id = predictions[pos].item()
        true_id = labels[pos].item()
        if true_id != -100:
            is_correct = "✅" if pred_id == true_id else "❌"
            print(f"     [{i}] pos={pos}: {is_correct} pred={pred_id} '{tokenizer.decode([pred_id])}', true={true_id} '{tokenizer.decode([true_id])}'")

calc_accuracy(prompt_last_positions, "Prompt[-1]")
calc_accuracy(mask_positions, "Mask positions")
calc_accuracy(real_positions, "Real positions")

# 总体准确率
total_correct = (predictions == labels) & valid_mask
total_accuracy = total_correct.sum().float() / valid_mask.sum().float()
print(f"\n总体准确率: {total_accuracy.item():.4f}")

# ========== 分析不同block的Mask准确率 ==========
print(f"\n" + "="*100)
print("分析各个Block的Mask准确率")
print("="*100)

block_idx = 0
current_pos = prompt_len

for seg_type, _, seg_len in block_info:
    if seg_type == 'mask':
        block_mask_positions = list(range(current_pos, current_pos + seg_len))

        preds = predictions[current_pos:current_pos + seg_len]
        labels_subset = labels[current_pos:current_pos + seg_len]
        valid = labels_subset != -100

        if valid.sum() > 0:
            correct = (preds == labels_subset) & valid
            accuracy = correct.sum().float() / valid.sum().float()

            print(f"\nBlock {block_idx} Mask positions [{current_pos}:{current_pos+seg_len}]:")
            print(f"   准确率: {accuracy.item():.4f} ({correct.sum().item()}/{valid.sum().item()})")

            # 显示每个mask的预测
            for i in range(seg_len):
                pos = current_pos + i
                pred_id = preds[i].item()
                true_id = labels_subset[i].item()
                if true_id != -100:
                    is_correct = "✅" if pred_id == true_id else "❌"
                    print(f"     Mask[{i}]: {is_correct} pred={pred_id} '{tokenizer.decode([pred_id])}', true={true_id} '{tokenizer.decode([true_id])}'")

        block_idx += 1

    current_pos += seg_len
