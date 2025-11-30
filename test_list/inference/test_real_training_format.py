#!/usr/bin/env python3
"""
使用真实训练数据格式测试模型
"""

import sys
from pathlib import Path
import torch

# 禁用所有torch.compile
import torch._dynamo
torch._dynamo.config.disable = True

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import create_interleaved_training_data, create_interleaved_labels
from dllm_reasoning.trainer.flex_attention_utils import create_block_mask_from_batch

MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
DATA_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"

print("="*100)
print("使用真实训练数据格式测试模型")
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
).to(device).train()  # 使用 train() 模式以启用 FlexAttention

print(f"   ✅ 模型加载完成 (training mode for FlexAttention)")

# ========== 加载训练数据 ==========
print(f"\n📂 加载训练数据: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"   总样本数: {df.shape[0]}")

# 取第一个样本
sample = df.iloc[0]

# 处理数据
prompt = sample['prompt']
target = sample['target']

messages = list(prompt) if isinstance(prompt, (list, tuple)) else prompt
messages = messages.tolist() if hasattr(messages, 'tolist') else messages

target_list = target.tolist() if hasattr(target, 'tolist') else target
if isinstance(target_list, list) and len(target_list) > 0:
    target_content = target_list[0].get('content', '') if isinstance(target_list[0], dict) else str(target_list[0])
else:
    target_content = str(target_list)

# 去掉<think>标签 (和训练时保持一致)
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

print(f"\n📝 数据信息:")
print(f"   Prompt长度: {prompt_len}")
print(f"   完整序列长度: {input_ids.size(0)}")
print(f"   Response长度: {input_ids.size(0) - prompt_len}")

# 创建loss_mask (训练时用的)
loss_mask = torch.ones(input_ids.size(0), dtype=torch.long, device=device)
loss_mask[:prompt_len] = 0  # Prompt部分mask=0

# ========== 使用训练时的函数创建interleaved格式 ==========
print(f"\n🔧 创建Interleaved训练格式...")
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

print(f"   Interleaved序列长度: {interleaved_ids.size(0)}")
print(f"   Block info数量: {len(block_info)}")
print(f"   前3个blocks: {block_info[:3]}")

# 转换block_info格式
block_info_converted = [(seg_type, seg_len) for seg_type, _, seg_len in block_info]

# ========== 只测试第一个block ==========
# 找到第一个mask和第一个real
first_mask_idx = None
first_real_idx = None

for idx, (seg_type, _, seg_len) in enumerate(block_info):
    if seg_type == 'mask' and first_mask_idx is None:
        first_mask_idx = idx
    if seg_type == 'real' and first_real_idx is None:
        first_real_idx = idx
        break

print(f"\n第一个mask block index: {first_mask_idx}")
print(f"第一个real block index: {first_real_idx}")

# 计算实际序列范围
# 从prompt到第一个real block结束
# block_info格式: (seg_type, list_idx, seg_len)
# interleaved_ids是按list顺序拼接的

# 计算第一个real block在interleaved_ids中的结束位置
seq_pos = prompt_len
for idx, (seg_type, _, seg_len) in enumerate(block_info):
    if idx <= first_real_idx:
        seq_pos += seg_len
    else:
        break

test_seq_len = seq_pos
print(f"测试序列长度: {test_seq_len} (Prompt + 第一个Mask + 第一个Real)")

# 截取
test_ids = interleaved_ids[:test_seq_len]
test_position_ids = position_ids[:test_seq_len]
test_labels = labels[:test_seq_len]
test_block_info = block_info_converted[:first_real_idx+1]

print(f"\n测试序列信息:")
print(f"  IDs shape: {test_ids.shape}")
print(f"  Position IDs shape: {test_position_ids.shape}")
print(f"  Labels shape: {test_labels.shape}")
print(f"  Block info: {test_block_info}")

# 打印关键部分内容
print(f"\n关键部分内容:")
# 第一个mask的位置
mask_start = prompt_len
mask_end = prompt_len + 3
real_start = mask_end
real_end = real_start + 4

print(f"  Mask部分 [{mask_start}:{mask_end}]:")
print(f"    IDs: {test_ids[mask_start:mask_end].tolist()}")
print(f"    Text: {tokenizer.decode(test_ids[mask_start:mask_end])}")

print(f"  Real部分 [{real_start}:{real_end}]:")
print(f"    IDs: {test_ids[real_start:real_end].tolist()}")
print(f"    Text: '{tokenizer.decode(test_ids[real_start:real_end])}'")

print(f"  Labels (Mask位置) [{mask_start}:{mask_end}]:")
print(f"    {test_labels[mask_start:mask_end].tolist()}")
for i in range(3):
    label_id = test_labels[mask_start + i].item()
    if label_id != -100:
        print(f"      Mask[{i}]应该预测: ID={label_id}, '{tokenizer.decode([label_id])}'")

# ========== Forward pass ==========
print(f"\n🚀 Forward pass...")

# 构造batch格式
batch = {
    "input_ids": test_ids.unsqueeze(0),
    "position_ids": test_position_ids.unsqueeze(0),
    "block_info": [test_block_info],
    "prompt_len": [prompt_len],
    "seq_lens": [test_ids.size(0)],
}

# 创建FlexAttention mask
block_mask = create_block_mask_from_batch(batch, device)

with torch.no_grad():
    outputs = model(
        test_ids.unsqueeze(0),
        position_ids=test_position_ids.unsqueeze(0),
        attention_mask=block_mask,
        use_cache=False
    )
    logits = outputs.logits[0]  # [seq_len, vocab]

print(f"  Logits shape: {logits.shape}")

# ========== 验证预测 ==========
print(f"\n📊 验证Mask位置的预测:")

for i in range(3):
    pos = mask_start + i
    pred_logit = logits[pos]  # 这个logit预测labels[pos]
    true_label = test_labels[pos].item()

    if true_label == -100:
        print(f"  Mask[{i}] (位置{pos}): 没有label")
        continue

    # Top 5预测
    top5_logits, top5_indices = pred_logit.topk(5)
    print(f"\n  Mask[{i}] (位置{pos}) 应该预测 ID={true_label} '{tokenizer.decode([true_label])}':")
    print(f"    Top 5 predictions:")
    for j, (logit_val, token_id) in enumerate(zip(top5_logits, top5_indices)):
        is_correct = "✅" if token_id.item() == true_label else "  "
        print(f"      {is_correct} #{j+1}: ID={token_id.item():6d}, logit={logit_val.item():8.4f}, text={repr(tokenizer.decode([token_id.item()]))}")

    # 真实label的logit和排名
    true_logit = pred_logit[true_label].item()
    all_logits_sorted = pred_logit.sort(descending=True)
    true_rank = (all_logits_sorted.values > true_logit).sum().item() + 1
    print(f"    真实token排名: {true_rank}/{pred_logit.size(0)}")

    # 判断是否正确
    predicted_id = pred_logit.argmax().item()
    if predicted_id == true_label:
        print(f"    ✅ 预测正确!")
    else:
        print(f"    ❌ 预测错误! 预测了 ID={predicted_id} '{tokenizer.decode([predicted_id])}'")
