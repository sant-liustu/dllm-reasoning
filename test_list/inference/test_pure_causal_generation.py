#!/usr/bin/env python3
"""
测试纯causal解码（不使用Mask tokens）
验证模型基础生成能力是否正常
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

MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
DATA_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"

print("="*100)
print("测试纯Causal解码 (不使用Mask)")
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
).to(device).eval()  # 使用eval模式

print(f"   ✅ 模型加载完成 (eval mode)")

# ========== 加载训练数据 ==========
print(f"\n📂 加载训练数据: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
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

# 去掉<think> - 对齐训练数据处理逻辑
if target_content.strip().startswith('<think>'):
    think_start = target_content.find('<think>')
    target_content = target_content[think_start + 7:].lstrip()

print(f"\n📝 Prompt最后部分: ...{messages[-1]['content'][-100:]}")
print(f"\n📝 Target开头: {target_content[:200]}")

# ========== 重要：对齐训练数据处理逻辑 ==========
# 训练时是这样处理的（见 interleaved_sft_dataset.py:502-514）：
# 1. 先将prompt转成字符串（add_generation_prompt=True）
# 2. 然后直接字符串拼接response
# 3. 最后一起tokenize

# 步骤1: Prompt转字符串
prompt_only_str = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)

# 步骤2: 字符串拼接（response + eos）
full_conversation_str = prompt_only_str + target_content + tokenizer.eos_token

# 步骤3: 分别tokenize
prompt_ids = tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
full_input_ids = tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

prompt_len = prompt_ids.size(0)
response_ids = full_input_ids[prompt_len:]

print(f"\n📏 序列长度:")
print(f"   Prompt: {prompt_len}")
print(f"   Response: {response_ids.size(0)}")
print(f"   Total: {full_input_ids.size(0)}")

# ========== 测试1: Teacher Forcing - 使用真实tokens计算logits ==========
print(f"\n" + "="*100)
print("测试1: Teacher Forcing (使用真实response tokens)")
print("="*100)

with torch.no_grad():
    outputs = model(
        full_input_ids.unsqueeze(0),
        use_cache=False
    )
    logits = outputs.logits[0]  # [seq_len, vocab]

# 分析前20个response positions的预测准确性
print(f"\n📊 Response前20个位置的预测:")
predictions = logits.argmax(dim=-1)

num_correct = 0
num_total = 0

for i in range(min(20, response_ids.size(0))):
    # Position prompt_len-1+i 预测 position prompt_len+i
    pred_pos = prompt_len - 1 + i
    true_pos = prompt_len + i

    pred_id = predictions[pred_pos].item()
    true_id = full_input_ids[true_pos].item()

    is_correct = pred_id == true_id
    if is_correct:
        num_correct += 1
    num_total += 1

    status = "✅" if is_correct else "❌"
    pred_text = tokenizer.decode([pred_id])
    true_text = tokenizer.decode([true_id])

    print(f"   [{i:2d}] pos {pred_pos:4d}→{true_pos:4d}: {status} pred={pred_id:6d} '{pred_text}', true={true_id:6d} '{true_text}'")

accuracy_tf = num_correct / num_total if num_total > 0 else 0
print(f"\n   Teacher Forcing 准确率: {accuracy_tf:.4f} ({num_correct}/{num_total})")

# ========== 测试2: Auto-regressive Generation - 逐个生成 ==========
print(f"\n" + "="*100)
print("测试2: Auto-regressive Generation (逐个生成)")
print("="*100)

# 从prompt开始，逐个生成20个tokens
generated_ids = prompt_ids.clone()
num_generate = 20

print(f"\n🚀 生成{num_generate}个tokens:")

for step in range(num_generate):
    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            use_cache=False
        )
        next_token_logits = outputs.logits[0, -1, :]  # Last position
        next_token_id = next_token_logits.argmax().item()

    # 添加生成的token
    generated_ids = torch.cat([generated_ids, torch.tensor([next_token_id], device=device)])

    # 对比真实token
    true_token_id = response_ids[step].item() if step < response_ids.size(0) else -1
    is_correct = next_token_id == true_token_id

    status = "✅" if is_correct else "❌"
    gen_text = tokenizer.decode([next_token_id])
    true_text = tokenizer.decode([true_token_id]) if true_token_id != -1 else "N/A"

    print(f"   Step {step:2d}: {status} gen={next_token_id:6d} '{gen_text}', true={true_token_id:6d} '{true_text}'")

# 计算AR生成准确率
generated_response = generated_ids[prompt_len:]
num_correct_ar = 0
for i in range(min(num_generate, response_ids.size(0))):
    if generated_response[i].item() == response_ids[i].item():
        num_correct_ar += 1

accuracy_ar = num_correct_ar / num_generate
print(f"\n   Auto-regressive 准确率: {accuracy_ar:.4f} ({num_correct_ar}/{num_generate})")

# ========== 总结 ==========
print(f"\n" + "="*100)
print("总结")
print("="*100)
print(f"Teacher Forcing准确率: {accuracy_tf:.4f} - 说明模型在看到真实tokens时的预测能力")
print(f"Auto-regressive准确率: {accuracy_ar:.4f} - 说明模型逐步生成时的准确性")
print(f"\n如果两个准确率都很高(>0.9)，说明：")
print(f"  ✅ 模型基础训练正常")
print(f"  ✅ Labels设置正确")
print(f"  ✅ 纯causal解码功能正常")
print(f"  ⚠️  问题可能出在Mask机制的实现上")
print(f"\n如果准确率都很低(<0.5)，说明：")
print(f"  ❌ 模型训练可能不充分")
print(f"  ❌ 或者使用了错误的checkpoint")
