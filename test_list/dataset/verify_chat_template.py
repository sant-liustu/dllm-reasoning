#!/usr/bin/env python3
"""
验证 chat_template 对 assistant response 的完整处理流程
确保手动拼接不会遗漏任何规范操作
"""

import sys
sys.path.insert(0, '/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning')

from transformers import AutoTokenizer
import json

def main():
    # 加载 tokenizer
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/checkpoints/iterative_refine/global_step_17172/huggingface"
    print(f"🔧 加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    # 构造测试数据
    prompt = [
        {"role": "system", "content": "Please reason step by step..."},
        {"role": "user", "content": "Solve x^2 = 0"}
    ]

    # 测试用的 response (包含 <think> 标签)
    response_with_think = """<think>
Okay, so I need to solve the equation x² = 0. Let me think about this step by step.
</think>

Solution: x = 0"""

    # 测试用的 response (不包含 <think> 标签)
    response_no_think = "Solution: x = 0"

    print("\n" + "="*80)
    print("📋 测试 1: Prompt only (add_generation_prompt=True)")
    print("="*80)
    prompt_only_str = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=False
    )
    print(f"结果长度: {len(prompt_only_str)} 字符")
    print(f"最后 200 字符:\n{repr(prompt_only_str[-200:])}")

    print("\n" + "="*80)
    print("📋 测试 2: Full conversation with <think> response")
    print("="*80)
    full_messages_with_think = prompt + [{"role": "assistant", "content": response_with_think}]
    full_str_with_think = tokenizer.apply_chat_template(
        full_messages_with_think, add_generation_prompt=False, tokenize=False
    )
    print(f"结果长度: {len(full_str_with_think)} 字符")
    print(f"最后 200 字符:\n{repr(full_str_with_think[-200:])}")

    print("\n" + "="*80)
    print("📋 测试 3: Full conversation without <think> response")
    print("="*80)
    full_messages_no_think = prompt + [{"role": "assistant", "content": response_no_think}]
    full_str_no_think = tokenizer.apply_chat_template(
        full_messages_no_think, add_generation_prompt=False, tokenize=False
    )
    print(f"结果长度: {len(full_str_no_think)} 字符")
    print(f"最后 200 字符:\n{repr(full_str_no_think[-200:])}")

    print("\n" + "="*80)
    print("🔍 关键分析: Response 部分的处理")
    print("="*80)

    # 提取 response 部分 (full - prompt_only)
    response_part_with_think = full_str_with_think[len(prompt_only_str):]
    response_part_no_think = full_str_no_think[len(prompt_only_str):]

    print(f"\n原始 response (with think): {len(response_with_think)} 字符")
    print(f"处理后 response 部分: {len(response_part_with_think)} 字符")
    print(f"完整内容:\n{repr(response_part_with_think)}")

    print(f"\n原始 response (no think): {len(response_no_think)} 字符")
    print(f"处理后 response 部分: {len(response_part_no_think)} 字符")
    print(f"完整内容:\n{repr(response_part_no_think)}")

    print("\n" + "="*80)
    print("🔍 检查 EOS token")
    print("="*80)
    print(f"tokenizer.eos_token = {repr(tokenizer.eos_token)}")
    print(f"tokenizer.eos_token_id = {tokenizer.eos_token_id}")

    # 检查是否包含 EOS token
    has_eos_with_think = tokenizer.eos_token in response_part_with_think if tokenizer.eos_token else False
    has_eos_no_think = tokenizer.eos_token in response_part_no_think if tokenizer.eos_token else False

    print(f"\nResponse (with think) 包含 EOS token: {has_eos_with_think}")
    print(f"Response (no think) 包含 EOS token: {has_eos_no_think}")

    print("\n" + "="*80)
    print("🔍 检查特殊标记")
    print("="*80)

    # 查找所有可能的特殊标记
    special_tokens = ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>", "</s>"]

    for token in special_tokens:
        in_with_think = token in response_part_with_think
        in_no_think = token in response_part_no_think
        print(f"  {token}:")
        print(f"    - Response (with think): {in_with_think}")
        print(f"    - Response (no think): {in_no_think}")

    print("\n" + "="*80)
    print("💡 解决方案验证: 手动拼接方式")
    print("="*80)

    # 方案 2: 手动拼接
    response_content = response_with_think
    if response_content.strip().startswith('<think>'):
        # 去掉开头的 <think>\n,因为 prompt 已经有了
        response_content = response_content[response_content.find('<think>') + 7:].lstrip()

    # 手动构造完整对话
    manual_full_str = prompt_only_str + response_content + tokenizer.eos_token

    print(f"原始方式 (apply_chat_template) 长度: {len(full_str_with_think)}")
    print(f"手动拼接方式长度: {len(manual_full_str)}")
    print(f"\n原始方式最后 150 字符:\n{repr(full_str_with_think[-150:])}")
    print(f"\n手动拼接最后 150 字符:\n{repr(manual_full_str[-150:])}")

    # 对比差异
    print("\n" + "="*80)
    print("🔍 差异对比")
    print("="*80)

    if full_str_with_think == manual_full_str:
        print("✅ 完全一致!")
    else:
        print("❌ 存在差异")

        # 找到第一个不同的位置
        min_len = min(len(full_str_with_think), len(manual_full_str))
        first_diff = None
        for i in range(min_len):
            if full_str_with_think[i] != manual_full_str[i]:
                first_diff = i
                break

        if first_diff is not None:
            print(f"\n第一个差异位置: {first_diff}")
            print(f"原始方式: {repr(full_str_with_think[first_diff:first_diff+100])}")
            print(f"手动方式: {repr(manual_full_str[first_diff:first_diff+100])}")
        elif len(full_str_with_think) != len(manual_full_str):
            print(f"\n长度不同:")
            print(f"  原始方式: {len(full_str_with_think)}")
            print(f"  手动方式: {len(manual_full_str)}")

    print("\n" + "="*80)
    print("📊 Tokenization 验证")
    print("="*80)

    # Tokenize 两种方式的结果
    tokens_original = tokenizer(full_str_with_think, return_tensors="pt", add_special_tokens=False)
    tokens_manual = tokenizer(manual_full_str, return_tensors="pt", add_special_tokens=False)

    print(f"原始方式 token 数量: {tokens_original['input_ids'].shape[1]}")
    print(f"手动方式 token 数量: {tokens_manual['input_ids'].shape[1]}")
    print(f"原始方式最后 10 个 token IDs: {tokens_original['input_ids'][0, -10:].tolist()}")
    print(f"手动方式最后 10 个 token IDs: {tokens_manual['input_ids'][0, -10:].tolist()}")

    # 解码最后几个 token
    print(f"\n原始方式最后 10 个 token 解码:")
    for i, tid in enumerate(tokens_original['input_ids'][0, -10:].tolist()):
        decoded = tokenizer.decode([tid])
        print(f"  {i}: {tid} -> {repr(decoded)}")

    print(f"\n手动方式最后 10 个 token 解码:")
    for i, tid in enumerate(tokens_manual['input_ids'][0, -10:].tolist()):
        decoded = tokenizer.decode([tid])
        print(f"  {i}: {tid} -> {repr(decoded)}")

    print("\n" + "="*80)
    print("✅ 验证完成")
    print("="*80)

if __name__ == "__main__":
    main()
