#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试FlexAttention推理：使用标准causal mask进行长文本生成。

在train模式下（走FlexAttention分支），传入标准causal mask，
手动逐token生成，验证输出是连贯有逻辑的。
"""

import argparse
import warnings
import torch
import torch.nn.functional as F

# 禁用 dynamo 中的 warnings.warn 问题
import torch._dynamo.config
torch._dynamo.config.reorderable_logging_functions.add(warnings.warn)

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def create_causal_block_mask(batch_size, seq_len, device):
    """创建标准causal mask的BlockMask格式"""
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal_mask,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
    return block_mask


def test_flex_generation(model_dir: str, device: str = "cuda", max_new_tokens: int = 1000):
    """测试FlexAttention的长文本生成（train模式 + causal mask）"""

    print(f"\n{'='*60}")
    print("Testing FlexAttention Generation (train mode + causal mask)")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"{'='*60}\n")

    # 加载模型
    print("Loading model...")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 设置为train模式，走FlexAttention分支
    model.train()
    print("Model set to TRAIN mode (using FlexAttention)")

    # 测试prompt
    test_prompt = r"""Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
\[\log_2\left({x \over yz}\right) = {1 \over 2}\]
\[\log_2\left({y \over xz}\right) = {1 \over 3}\]
\[\log_2\left({z \over xy}\right) = {1 \over 4}\]
Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.

Please reason step by step, and put your final answer within \boxed{}."""

    print(f"Test prompt:\n{test_prompt[:200]}...")

    # Tokenize
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    print(f"\nInput length: {input_ids.shape[1]} tokens")

    print(f"\nGenerating {max_new_tokens} tokens with FlexAttention...")
    print("-" * 60)

    generated_ids = input_ids.clone()

    # 手动逐token生成
    with torch.no_grad():
        for i in range(max_new_tokens):
            seq_len = generated_ids.shape[1]

            # 创建causal BlockMask
            causal_mask = create_causal_block_mask(
                batch_size=1,
                seq_len=seq_len,
                device=device,
            )

            # 创建position_ids
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # Forward (train mode会走FlexAttention)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model.model(
                    input_ids=generated_ids,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )

            # 获取最后一个token的hidden state，通过lm_head得到logits
            last_hidden = outputs.last_hidden_state[:, -1, :]
            logits = model.lm_head(last_hidden)

            # Sample next token
            probs = F.softmax(logits / 0.6, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # 检查是否生成了EOS
            if next_token.item() == tokenizer.eos_token_id:
                print(f"\n[EOS reached at token {i+1}]")
                break

            # 进度提示
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1} tokens...")

    # 解码
    new_tokens = generated_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Generated ({len(new_tokens)} tokens):\n")
    print(generated_text)
    print(f"\n{'='*60}")

    # 质量检查
    print("\n[Quality Check]")

    words = generated_text.split()
    if len(words) > 10:
        repeat_count = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        repeat_ratio = repeat_count / len(words)
        print(f"  Word repeat ratio: {repeat_ratio:.2%}")
        if repeat_ratio > 0.3:
            print("  ⚠ Warning: High repetition detected")
        else:
            print("  ✓ Low repetition - good!")

    math_keywords = ['log', 'equation', 'solve', 'therefore', 'thus', 'let', '=', '+']
    found = sum(1 for kw in math_keywords if kw.lower() in generated_text.lower())
    print(f"  Math keywords found: {found}/{len(math_keywords)}")

    if r'\boxed' in generated_text or 'boxed' in generated_text:
        print("  ✓ Contains boxed answer")

    print(f"\n✓ FlexAttention generation test complete!")
    print(f"  Please review the output above for coherence and logic.\n")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=1000)

    args = parser.parse_args()
    test_flex_generation(args.model_dir, args.device, args.max_new_tokens)


if __name__ == "__main__":
    main()
