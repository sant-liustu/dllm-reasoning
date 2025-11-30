#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证DLLM模型转换正确性：比较原始Qwen模型与转换后DLLM模型的推理输出。

确保两边都使用相同的attention实现以进行公平比较。
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def verify_conversion(original_dir: str, converted_dir: str, device: str = "cuda"):
    """比较原始模型和转换后模型的输出"""

    print("Loading models...")

    # 加载原始Qwen模型 - 使用eager attention进行验证
    original_model = AutoModelForCausalLM.from_pretrained(
        original_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",  # 使用eager attention进行验证
    )
    original_tokenizer = AutoTokenizer.from_pretrained(original_dir)

    # 加载转换后的DLLM模型 - 使用eager attention进行验证
    from transformers import AutoConfig
    dllm_config = AutoConfig.from_pretrained(converted_dir, trust_remote_code=True)
    dllm_config.inference_attn_implementation = "eager"  # 使用eager进行验证
    converted_model = AutoModelForCausalLM.from_pretrained(
        converted_dir,
        config=dllm_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    converted_tokenizer = AutoTokenizer.from_pretrained(converted_dir)

    # 设置为eval模式
    original_model.eval()
    converted_model.eval()

    # 测试输入
    test_prompt = "The capital of France is"

    print(f"\nTest prompt: '{test_prompt}'")
    print("-" * 60)

    # Tokenize
    orig_inputs = original_tokenizer(test_prompt, return_tensors="pt").to(device)
    conv_inputs = converted_tokenizer(test_prompt, return_tensors="pt").to(device)

    # 验证tokenizer输出一致
    print("\n[1] Checking tokenizer consistency...")
    if torch.equal(orig_inputs.input_ids, conv_inputs.input_ids):
        print("  ✓ Tokenizer outputs are identical")
    else:
        print("  ✗ Tokenizer outputs differ!")
        print(f"    Original: {orig_inputs.input_ids}")
        print(f"    Converted: {conv_inputs.input_ids}")
        return False

    # Forward pass
    print("\n[2] Comparing forward pass...")
    with torch.no_grad():
        orig_outputs = original_model(**orig_inputs, output_hidden_states=True)
        conv_outputs = converted_model(**conv_inputs, output_hidden_states=True)

    # 比较hidden states
    print("\n[3] Comparing hidden states layer by layer...")
    orig_hidden = orig_outputs.hidden_states
    conv_hidden = conv_outputs.hidden_states

    all_match = True
    for i, (oh, ch) in enumerate(zip(orig_hidden, conv_hidden)):
        diff = (oh - ch).abs().max().item()
        if diff > 1e-4:
            print(f"  Layer {i}: max diff = {diff:.6f} ✗")
            all_match = False
        else:
            print(f"  Layer {i}: max diff = {diff:.6f} ✓")

    # 比较logits
    print("\n[4] Comparing logits...")
    logits_diff = (orig_outputs.logits - conv_outputs.logits).abs().max().item()
    print(f"  Max logits diff: {logits_diff:.6f}")

    if logits_diff < 1e-4:
        print("  ✓ Logits match!")
    else:
        print("  ✗ Logits differ significantly!")
        all_match = False

    # 比较预测token
    print("\n[5] Comparing predicted tokens...")
    orig_pred = orig_outputs.logits[0, -1].argmax().item()
    conv_pred = conv_outputs.logits[0, -1].argmax().item()

    orig_token = original_tokenizer.decode([orig_pred])
    conv_token = converted_tokenizer.decode([conv_pred])

    print(f"  Original predicts: '{orig_token}'")
    print(f"  Converted predicts: '{conv_token}'")

    if orig_pred == conv_pred:
        print("  ✓ Predictions match!")
    else:
        print("  ✗ Predictions differ!")
        all_match = False

    # 总结
    print("\n" + "=" * 60)
    if all_match:
        print("✓ VERIFICATION PASSED - Models are equivalent")
    else:
        print("✗ VERIFICATION FAILED - Models differ")
    print("=" * 60)

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Verify DLLM model conversion")
    parser.add_argument(
        "--original_dir",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Path to original Qwen model (HuggingFace model ID or local path)"
    )
    parser.add_argument(
        "--converted_dir",
        type=str,
        default="/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B",
        help="Path to converted DLLM model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )

    args = parser.parse_args()
    success = verify_conversion(args.original_dir, args.converted_dir, args.device)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
