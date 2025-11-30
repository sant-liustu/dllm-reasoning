#!/usr/bin/env python3
"""
测试修复后的数据处理逻辑
验证 <think> 内容是否被正确保留
"""

import sys
sys.path.insert(0, '/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning')

from transformers import AutoTokenizer
import pandas as pd

def main():
    # 加载 tokenizer
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/checkpoints/iterative_refine/global_step_17172/huggingface"
    print(f"🔧 加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    # 加载一条真实训练数据
    data_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"
    print(f"\n📂 加载数据: {data_path}")
    df = pd.read_parquet(data_path)
    sample = df.iloc[0]

    prompt = sample['prompt']
    target = sample['target']
    response_content = target[0]['content']

    print(f"\n原始 response 长度: {len(response_content)} 字符")
    print(f"是否包含 <think>: {'<think>' in response_content}")
    print(f"是否包含 </think>: {'</think>' in response_content}")

    # 分析 response 结构
    if '<think>' in response_content and '</think>' in response_content:
        think_start = response_content.find('<think>')
        think_end = response_content.find('</think>') + 8
        think_content = response_content[think_start:think_end]
        solution_content = response_content[think_end:].strip()

        print(f"\n<think> 推理部分: {len(think_content)} 字符")
        print(f"Solution 部分: {len(solution_content)} 字符")

    print("\n" + "="*80)
    print("🔧 方法 1: 原始方式 (apply_chat_template - 会过滤 <think>)")
    print("="*80)

    prompt_only_str_old = tokenizer.apply_chat_template(
        list(prompt), add_generation_prompt=True, tokenize=False
    )
    full_messages_old = list(prompt) + [{"role": "assistant", "content": response_content}]
    full_str_old = tokenizer.apply_chat_template(
        full_messages_old, add_generation_prompt=False, tokenize=False
    )

    print(f"Prompt 长度: {len(prompt_only_str_old)} 字符")
    print(f"完整对话长度: {len(full_str_old)} 字符")
    print(f"Response 部分长度: {len(full_str_old) - len(prompt_only_str_old)} 字符")

    tokens_old = tokenizer(full_str_old, return_tensors="pt", add_special_tokens=False)
    print(f"总 token 数: {tokens_old['input_ids'].shape[1]}")

    print("\n" + "="*80)
    print("✅ 方法 2: 修复后的方式 (手动拼接 - 保留 <think>)")
    print("="*80)

    # 使用新的逻辑
    prompt_only_str_new = tokenizer.apply_chat_template(
        list(prompt), add_generation_prompt=True, tokenize=False
    )

    response_content_new = response_content
    if response_content_new.strip().startswith('<think>'):
        think_start = response_content_new.find('<think>')
        response_content_new = response_content_new[think_start + 7:].lstrip()

    full_str_new = prompt_only_str_new + response_content_new + tokenizer.eos_token

    print(f"Prompt 长度: {len(prompt_only_str_new)} 字符")
    print(f"完整对话长度: {len(full_str_new)} 字符")
    print(f"Response 部分长度: {len(full_str_new) - len(prompt_only_str_new)} 字符")

    tokens_new = tokenizer(full_str_new, return_tensors="pt", add_special_tokens=False)
    print(f"总 token 数: {tokens_new['input_ids'].shape[1]}")

    print("\n" + "="*80)
    print("📊 对比结果")
    print("="*80)
    print(f"Token 数量增加: {tokens_new['input_ids'].shape[1]} - {tokens_old['input_ids'].shape[1]} = {tokens_new['input_ids'].shape[1] - tokens_old['input_ids'].shape[1]}")
    print(f"Response 长度增加: {len(full_str_new) - len(prompt_only_str_new)} - {len(full_str_old) - len(prompt_only_str_old)} = {(len(full_str_new) - len(prompt_only_str_new)) - (len(full_str_old) - len(prompt_only_str_old))} 字符")

    # 检查是否包含 <think> 内容
    response_part_old = full_str_old[len(prompt_only_str_old):]
    response_part_new = full_str_new[len(prompt_only_str_new):]

    print(f"\n原始方式 response 包含 <think>: {'<think>' in response_part_old}")
    print(f"原始方式 response 包含 </think>: {'</think>' in response_part_old}")
    print(f"新方式 response 包含 <think>: {'<think>' in response_part_new}")
    print(f"新方式 response 包含 </think>: {'</think>' in response_part_new}")

    print("\n" + "="*80)
    print("🔍 验证 EOS token")
    print("="*80)
    print(f"原始方式最后 50 字符: {repr(full_str_old[-50:])}")
    print(f"新方式最后 50 字符: {repr(full_str_new[-50:])}")
    print(f"\n原始方式包含 EOS: {tokenizer.eos_token in full_str_old}")
    print(f"新方式包含 EOS: {tokenizer.eos_token in full_str_new}")

    print("\n" + "="*80)
    print("✅ 测试完成")
    print("="*80)

    if '</think>' in response_part_new and tokenizer.eos_token in full_str_new:
        print("✅ 成功! 新方式保留了完整的 <think> 推理过程并正确添加了 EOS token")
    else:
        print("❌ 失败! 请检查逻辑")

if __name__ == "__main__":
    main()
