#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证FlexAttention在block_size=1时与eager attention输出一致。

当block_size=1时，FlexAttention的mask就是标准causal mask，
输出应该与eager attention完全一致。
"""

import warnings
import torch

# 禁用 dynamo 中的 warnings.warn 问题
import torch._dynamo.config
torch._dynamo.config.reorderable_logging_functions.add(warnings.warn)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def test_flex_vs_eager(model_dir: str, device: str = "cuda"):
    """比较FlexAttention(training mode, block_size=1)与eager attention的输出"""

    print(f"\n{'='*60}")
    print("Testing FlexAttention vs Eager Attention")
    print("(block_size=1 should produce identical results)")
    print(f"{'='*60}\n")

    # 加载模型 - eager模式
    print("Loading model with eager attention...")
    config_eager = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config_eager.inference_attn_implementation = "eager"

    model_eager = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config_eager,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model_eager.eval()

    # 加载模型 - flex attention模式 (training)
    print("Loading model for FlexAttention test...")
    config_flex = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    model_flex = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config_flex,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model_flex.train()  # 训练模式使用FlexAttention

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 测试输入
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]

    print(f"\nTest prompt: '{test_prompt}'")
    print(f"Sequence length: {seq_len}")

    # Eager模式forward
    print("\n[1] Running eager attention (eval mode)...")
    with torch.no_grad():
        outputs_eager = model_eager(
            **inputs,
            output_hidden_states=True,
        )
    print(f"  Eager logits shape: {outputs_eager.logits.shape}")

    # FlexAttention模式forward (block_size=1)
    print("\n[2] Running FlexAttention (train mode, block_size=1)...")
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    labels = inputs.input_ids.clone()

    with torch.no_grad():  # 虽然是train mode，但不需要梯度
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs_flex = model_flex(
                input_ids=inputs.input_ids,
                position_ids=position_ids,
                labels=labels,
                block_size=1,  # block_size=1 退化为标准causal
                output_hidden_states=True,
            )
    print(f"  Flex logits shape: {outputs_flex.logits.shape}")

    # 比较输出
    print("\n[3] Comparing outputs...")

    # 注意：训练模式的logits可能只返回部分（根据logits_to_keep）
    # 需要处理形状不同的情况
    eager_logits = outputs_eager.logits
    flex_logits = outputs_flex.logits

    print(f"  Eager logits shape: {eager_logits.shape}")
    print(f"  Flex logits shape: {flex_logits.shape}")

    # 比较最后一个token的logits（预测下一个token）
    eager_last = eager_logits[0, -1, :]

    # Flex模式可能返回不同形状，取最后一个
    if flex_logits.dim() == 3:
        flex_last = flex_logits[0, -1, :]
    else:
        flex_last = flex_logits[-1, :]

    diff = (eager_last - flex_last).abs().max().item()
    print(f"\n  Max diff (last token logits): {diff:.6f}")

    # 比较预测的token
    eager_pred = eager_last.argmax().item()
    flex_pred = flex_last.argmax().item()

    eager_token = tokenizer.decode([eager_pred])
    flex_token = tokenizer.decode([flex_pred])

    print(f"\n  Eager predicts: '{eager_token}' (id={eager_pred})")
    print(f"  Flex predicts: '{flex_token}' (id={flex_pred})")

    if eager_pred == flex_pred:
        print("\n  ✓ Predictions match!")
    else:
        print("\n  ✗ Predictions differ!")

    # 判断是否通过
    # 由于FlexAttention和eager的数值实现不同，允许一定误差
    threshold = 1.0  # bfloat16精度下的合理阈值
    if diff < threshold and eager_pred == flex_pred:
        print(f"\n{'='*60}")
        print("✓ TEST PASSED: FlexAttention (block_size=1) ~ Eager")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"\n{'='*60}")
        print("✗ TEST FAILED: Outputs differ significantly")
        print(f"{'='*60}\n")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    success = test_flex_vs_eager(args.model_dir, args.device)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
