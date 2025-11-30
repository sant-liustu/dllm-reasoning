#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试DLLM模型的thinking生成能力。

DeepSeek-R1-Distill模型使用<think>标签进行链式思维推理。
参考: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def test_generation(model_dir: str, attn_impl: str = "sdpa", device: str = "cuda"):
    """测试模型生成能力"""

    print(f"\n{'='*60}")
    print(f"Testing DLLM Thinking Generation")
    print(f"Attention implementation: {attn_impl}")
    print(f"{'='*60}\n")

    # 加载模型
    print("Loading model...")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.inference_attn_implementation = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    # 测试prompts - 使用DeepSeek-R1推荐的格式
    test_cases = [
        {
            "name": "AIME-style Math Problem",
            "prompt": r"""Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
\[\log_2\left({x \over yz}\right) = {1 \over 2}\]
\[\log_2\left({y \over xz}\right) = {1 \over 3}\]
\[\log_2\left({z \over xy}\right) = {1 \over 4}\]
Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.

Please reason step by step, and put your final answer within \boxed{}.""",
        },
    ]

    # 生成参数 - 按DeepSeek-R1推荐设置
    gen_kwargs = {
        "max_new_tokens": 2048,  # 困难问题需要更长的推理
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for i, case in enumerate(test_cases):
        print(f"\n{'─'*60}")
        print(f"Test {i+1}: {case['name']}")
        print(f"{'─'*60}")

        # 使用chat template构建输入
        messages = [{"role": "user", "content": case["prompt"]}]

        # 尝试使用chat template，如果不支持则直接使用prompt
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            input_text = case["prompt"]

        print(f"\nInput:\n{case['prompt']}")

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # 解码输出
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 只显示新生成的部分
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"\nGenerated ({len(new_tokens)} tokens):")
        print(generated_text[:3000])  # 增加输出长度
        if len(generated_text) > 3000:
            print("... [truncated]")

        # 检查是否包含thinking标记
        if "<think>" in generated or "</think>" in generated:
            print("\n✓ Contains thinking tags")
        else:
            print("\n○ No explicit thinking tags (may still be valid)")

    print(f"\n{'='*60}")
    print("Generation test complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test DLLM thinking generation")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B",
        help="Path to DLLM model"
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Attention implementation (default: sdpa)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)"
    )

    args = parser.parse_args()
    test_generation(args.model_dir, args.attn_impl, args.device)


if __name__ == "__main__":
    main()
