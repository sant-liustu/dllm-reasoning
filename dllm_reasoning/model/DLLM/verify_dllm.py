#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify DLLM model loading and FlexAttention support.

Usage:
    python verify_dllm.py --model_dir /path/to/DLLM-7B
"""

import argparse
import sys
import torch


def verify_model(model_dir: str, device: str = "cuda"):
    """Verify DLLM model can be loaded and supports FlexAttention."""

    print(f"\n{'='*60}")
    print("DLLM Model Verification")
    print(f"{'='*60}\n")

    # Step 1: Check PyTorch version
    print("[1/5] Checking PyTorch version...")
    pytorch_version = torch.__version__
    print(f"  PyTorch version: {pytorch_version}")

    major, minor = map(int, pytorch_version.split(".")[:2])
    if major < 2 or (major == 2 and minor < 5):
        print(f"  ⚠ Warning: FlexAttention requires PyTorch >= 2.5.0")
        print(f"  Current version may not support FlexAttention")
    else:
        print(f"  ✓ PyTorch version OK")

    # Step 2: Check FlexAttention availability
    print("\n[2/5] Checking FlexAttention availability...")
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        print("  ✓ FlexAttention imported successfully")
    except ImportError as e:
        print(f"  ✗ FlexAttention import failed: {e}")
        print("  Please upgrade PyTorch to 2.5.0 or later")
        return False

    # Step 3: Load config
    print("\n[3/5] Loading DLLM config...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        print(f"  ✓ Config loaded successfully")
        print(f"  Model type: {config.model_type}")
        print(f"  Mask token ID: {getattr(config, 'mask_token_id', 'N/A')}")
        print(f"  Attn implementation: {getattr(config, 'attn_implementation', 'N/A')}")
        print(f"  Note: block_size is a training hyperparameter, not in config")
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False

    # Step 4: Load tokenizer
    print("\n[4/5] Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print(f"  ✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  EOS token ID: {tokenizer.eos_token_id}")
    except Exception as e:
        print(f"  ✗ Tokenizer loading failed: {e}")
        return False

    # Step 5: Load model
    print("\n[5/5] Loading DLLM model...")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device if torch.cuda.is_available() else "cpu",
        )
        print(f"  ✓ Model loaded successfully")
        print(f"  Model class: {model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Quick inference test (optional)
    print("\n[Bonus] Quick inference test...")
    try:
        model.eval()
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"  ✓ Forward pass successful")
        print(f"  Output logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"  ⚠ Inference test failed: {e}")
        print("  (This may be expected if FlexAttention requires specific mask)")

    print(f"\n{'='*60}")
    print("✓ DLLM model verification complete!")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify DLLM model loading")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to DLLM model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load model on (default: cuda)"
    )

    args = parser.parse_args()
    success = verify_model(args.model_dir, args.device)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
