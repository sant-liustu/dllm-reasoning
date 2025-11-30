#!/usr/bin/env python3
"""
快速测试 position_ids 是否正确
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.inference.interleaved_generator import interleaved_generate

MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).to(device).eval()

# 获取特殊 token
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else eos_token_id
mask_token_id = eos_token_id  # 使用 eos 作为 mask

# 测试
prompt = "What is 2+2?"
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

print(f"测试 prompt: {prompt}")
print(f"输入长度: {input_ids.size(1)} tokens\n")

# 测试 num_masks=1，verbose=True
print("="*80)
print("测试 num_masks=1 (应该每次预测2个token)")
print("="*80)

output_ids = interleaved_generate(
    model=model,
    input_ids=input_ids,
    eos_token_id=eos_token_id,
    mask_token_id=mask_token_id,
    pad_token_id=pad_token_id,
    max_new_tokens=10,  # 只生成10个token，快速测试
    num_masks=1,
    verbose=True,  # 开启详细输出
    tokenizer=tokenizer,
)

generated_text = tokenizer.decode(
    output_ids[0, input_ids.size(1):],
    skip_special_tokens=True
)

print(f"\n最终生成: {generated_text}")
