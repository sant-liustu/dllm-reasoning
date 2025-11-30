#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Qwen2.5 model to DLLM format.

This script converts a standard Qwen2.5 model to DLLM format by:
1. Copying all weight files (no conversion needed - weights are compatible)
2. Copying and modifying config.json to use DLLM model type
3. Copying tokenizer files
4. Copying DLLM modeling files

Note: block_size is a training hyperparameter, not part of model conversion.
      It should be specified in your training configuration.

Usage:
    python convert_qwen_to_dllm.py \
        --input_dir /path/to/Qwen2.5-7B \
        --output_dir /path/to/DLLM-7B

The converted model can then be loaded with:
    model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def get_dllm_model_dir():
    """Get the directory containing DLLM model files."""
    return Path(__file__).parent


def convert_config(input_dir: str, output_dir: str, eos_token_id: int):
    """Convert Qwen config.json to DLLM format."""
    config_path = os.path.join(input_dir, "config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Modify config for DLLM
    config["model_type"] = "dllm"
    config["architectures"] = ["DLLMForCausalLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_dllm.DLLMConfig",
        "AutoModel": "modeling_dllm.DLLMModel",
        "AutoModelForCausalLM": "modeling_dllm.DLLMForCausalLM"
    }

    # Add DLLM-specific fields
    # Note: block_size is NOT set here - it's a training hyperparameter
    config["attn_implementation"] = "flex_attention"
    config["mask_token_id"] = eos_token_id  # Use EOS as mask token
    config["fuse_cross_entropy"] = True
    config["use_cache"] = False  # Must be False for training
    config["debug"] = False
    config["micro_forward"] = False
    config["skip_checkpoint"] = False

    output_config_path = os.path.join(output_dir, "config.json")
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✓ Config saved to {output_config_path}")
    return config


def copy_weight_files(input_dir: str, output_dir: str):
    """Copy weight files (safetensors or bin)."""
    weight_patterns = [
        "*.safetensors",
        "*.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]

    copied = 0
    for pattern in weight_patterns:
        for file_path in Path(input_dir).glob(pattern):
            dest = os.path.join(output_dir, file_path.name)
            if not os.path.exists(dest):
                shutil.copy2(file_path, dest)
                copied += 1
                print(f"  Copied: {file_path.name}")

    print(f"✓ Copied {copied} weight files")
    return copied


def copy_tokenizer_files(input_dir: str, output_dir: str):
    """Copy tokenizer files."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "tokenizer.model",
        "generation_config.json",
    ]

    copied = 0
    for filename in tokenizer_files:
        src = os.path.join(input_dir, filename)
        if os.path.exists(src):
            dest = os.path.join(output_dir, filename)
            shutil.copy2(src, dest)
            copied += 1
            print(f"  Copied: {filename}")

    print(f"✓ Copied {copied} tokenizer files")
    return copied


def copy_dllm_model_files(output_dir: str):
    """Copy DLLM modeling files."""
    dllm_dir = get_dllm_model_dir()

    model_files = [
        "configuration_dllm.py",
        "modeling_dllm.py",
        "fused_linear_diffusion_cross_entropy.py",
    ]

    for filename in model_files:
        src = os.path.join(dllm_dir, filename)
        if os.path.exists(src):
            dest = os.path.join(output_dir, filename)
            shutil.copy2(src, dest)
            print(f"  Copied: {filename}")
        else:
            print(f"  ⚠ Warning: {filename} not found in {dllm_dir}")

    print(f"✓ Copied DLLM model files")


def get_eos_token_id(input_dir: str) -> int:
    """Get EOS token ID from config or tokenizer."""
    # Try config.json first
    config_path = os.path.join(input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            if "eos_token_id" in config:
                eos_id = config["eos_token_id"]
                # Handle list format
                if isinstance(eos_id, list):
                    eos_id = eos_id[0]
                return eos_id

    # Fallback to Qwen2.5 default
    return 151643


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen2.5 model to DLLM format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input Qwen2.5 model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output DLLM model directory"
    )
    parser.add_argument(
        "--eos_token_id",
        type=int,
        default=None,
        help="EOS token ID to use as mask token (default: auto-detect from config)"
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Converting Qwen2.5 to DLLM format")
    print(f"{'='*60}")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Get EOS token ID
    eos_token_id = args.eos_token_id
    if eos_token_id is None:
        eos_token_id = get_eos_token_id(args.input_dir)
    print(f"Using EOS token ID as mask: {eos_token_id}\n")

    # Step 1: Copy weight files
    print("[1/4] Copying weight files...")
    copy_weight_files(args.input_dir, args.output_dir)

    # Step 2: Copy tokenizer files
    print("\n[2/4] Copying tokenizer files...")
    copy_tokenizer_files(args.input_dir, args.output_dir)

    # Step 3: Convert config
    print("\n[3/4] Converting config.json...")
    convert_config(args.input_dir, args.output_dir, eos_token_id)

    # Step 4: Copy DLLM model files
    print("\n[4/4] Copying DLLM model files...")
    copy_dllm_model_files(args.output_dir)

    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"{'='*60}")
    print(f"\nTo load the model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained(")
    print(f"      '{args.output_dir}',")
    print(f"      trust_remote_code=True")
    print(f"  )")


if __name__ == "__main__":
    main()
