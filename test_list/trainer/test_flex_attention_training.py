#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试DLLM模型训练模式下的FlexAttention。

训练模式使用FlexAttention进行block diffusion的attention mask。
"""

import argparse
import warnings
import torch

# 禁用 dynamo 中的 warnings.warn 问题
import torch._dynamo.config
torch._dynamo.config.reorderable_logging_functions.add(warnings.warn)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def test_flex_attention(model_dir: str, device: str = "cuda"):
    """测试FlexAttention在训练模式下的行为"""

    print(f"\n{'='*60}")
    print(f"Testing DLLM FlexAttention (Training Mode)")
    print(f"{'='*60}\n")

    # 检查PyTorch版本
    print(f"PyTorch version: {torch.__version__}")

    # 检查FlexAttention可用性
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        print("✓ FlexAttention is available")
    except ImportError as e:
        print(f"✗ FlexAttention not available: {e}")
        return False

    # 加载模型
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 设置为训练模式
    model.train()
    print("Model set to training mode")

    # 测试输入
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    print(f"\nTest prompt: '{test_prompt}'")
    print(f"Input shape: {inputs.input_ids.shape}")

    # 创建block diffusion所需的输入
    # 在实际训练中，会有 xt (noisy) 和 x0 (clean) 的拼接
    # 这里我们简单测试forward pass

    # 创建一个简单的attention mask用于测试
    batch_size, seq_len = inputs.input_ids.shape

    # 模拟block diffusion的输入：拼接 xt 和 x0
    # xt: [batch, block_size] - noisy tokens
    # x0: [batch, block_size] - clean tokens
    # 实际输入: [batch, 2*block_size]
    block_size = seq_len  # 用当前序列长度作为block_size

    # 训练模式需要的参数
    block_size = seq_len  # block_size 等于原始序列长度

    # 创建position_ids - 训练模式需要
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 创建labels - 训练模式需要
    labels = inputs.input_ids.clone()

    print(f"Input shape: {inputs.input_ids.shape}")
    print(f"Block size: {block_size}")
    print(f"Position IDs shape: {position_ids.shape}")

    # 尝试forward pass
    print("\n[1] Testing forward pass in training mode...")
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=inputs.input_ids,
                position_ids=position_ids,
                labels=labels,
                block_size=block_size,
                output_hidden_states=True,
            )
        print(f"  ✓ Forward pass successful")
        print(f"  Output logits shape: {outputs.logits.shape}")
        print(f"  Number of hidden states: {len(outputs.hidden_states)}")

        # 检查输出是否有效（非NaN/Inf）
        if torch.isnan(outputs.logits).any():
            print("  ✗ WARNING: Output contains NaN!")
            return False
        if torch.isinf(outputs.logits).any():
            print("  ✗ WARNING: Output contains Inf!")
            return False
        print("  ✓ Output values are valid (no NaN/Inf)")

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试梯度计算
    print("\n[2] Testing backward pass...")
    try:
        # 简单的loss计算
        loss = outputs.logits.mean()
        loss.backward()
        print(f"  ✓ Backward pass successful")
        print(f"  Loss value: {loss.item():.6f}")

        # 检查梯度是否有效
        grad_ok = True
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  ✗ WARNING: Gradient for {name} contains NaN!")
                    grad_ok = False
                    break
        if grad_ok:
            print("  ✓ Gradients are valid (no NaN)")

    except Exception as e:
        print(f"  ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 切换回eval模式，确保推理仍然工作
    print("\n[3] Verifying inference mode still works...")
    model.eval()
    try:
        with torch.no_grad():
            outputs_eval = model(**inputs)
        print(f"  ✓ Eval mode forward pass successful")
        print(f"  Logits shape: {outputs_eval.logits.shape}")
    except Exception as e:
        print(f"  ✗ Eval mode failed: {e}")
        return False

    print(f"\n{'='*60}")
    print("✓ FlexAttention training mode test PASSED!")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test DLLM FlexAttention training mode")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B",
        help="Path to DLLM model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)"
    )

    args = parser.parse_args()
    success = test_flex_attention(args.model_dir, args.device)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
