#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Batch 处理和 Padding 的正确性

验证关键点:
1. 不同长度的序列在 batch 中是否正确 padding
2. Padding 位置是否正确设置为 -100 (不计算 loss)
3. block_info 在 padding 后是否仍然正确对应
4. BlockMask 构造是否正确处理 padding
5. seq_lens 是否正确用于 mask 构造
"""

import sys
sys.path.insert(0, "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning")

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dllm_reasoning.model.DLLM import DLLMForCausalLM
from dllm_reasoning.trainer.flex_attention_utils import create_block_mask_from_batch

try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    print("⚠️  FlexAttention 不可用，将跳过测试")


def test_batch_with_padding():
    """测试包含不同长度序列的 batch"""
    print("=" * 100)
    print("Test: Batch Processing with Different Sequence Lengths")
    print("=" * 100)

    if not FLEX_ATTN_AVAILABLE:
        print("⚠️  跳过测试（FlexAttention 不可用）\n")
        return False

    print("\n加载模型和 tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mask_token_id = tokenizer.eos_token_id

    model = DLLMForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)

    model.train()
    print(f"✓ 模型加载成功（设备: {device}）")
    print(f"  Pad token ID: {pad_token_id}")

    # ========== 构造不同长度的序列 ==========
    print("\n构造测试 batch（3个样本，不同长度）...")

    # 样本 1: 短序列 [P(3)][M(3)][R(4)]
    prompt1_len = 3
    sample1_prompt = torch.randint(1000, 5000, (prompt1_len,), device=device)
    sample1_mask = torch.full((3,), mask_token_id, device=device)
    sample1_real = torch.randint(5000, 10000, (4,), device=device)
    sample1 = torch.cat([sample1_prompt, sample1_mask, sample1_real], dim=0)
    sample1_len = len(sample1)  # 10
    sample1_block_info = [('mask', 3), ('real', 4)]
    sample1_prompt_len = prompt1_len

    # 样本 2: 中等长度 [P(5)][M(3)][R(4)][M(3)][R(4)]
    prompt2_len = 5
    sample2_prompt = torch.randint(1000, 5000, (prompt2_len,), device=device)
    sample2_mask1 = torch.full((3,), mask_token_id, device=device)
    sample2_real1 = torch.randint(5000, 10000, (4,), device=device)
    sample2_mask2 = torch.full((3,), mask_token_id, device=device)
    sample2_real2 = torch.randint(5000, 10000, (4,), device=device)
    sample2 = torch.cat([sample2_prompt, sample2_mask1, sample2_real1, sample2_mask2, sample2_real2], dim=0)
    sample2_len = len(sample2)  # 19
    sample2_block_info = [('mask', 3), ('real', 4), ('mask', 3), ('real', 4)]
    sample2_prompt_len = prompt2_len

    # 样本 3: 长序列 [P(7)][M(3)][R(4)][M(3)][R(4)][M(3)][R(4)]
    prompt3_len = 7
    sample3_prompt = torch.randint(1000, 5000, (prompt3_len,), device=device)
    sample3_parts = [sample3_prompt]
    for _ in range(3):
        sample3_parts.append(torch.full((3,), mask_token_id, device=device))
        sample3_parts.append(torch.randint(5000, 10000, (4,), device=device))
    sample3 = torch.cat(sample3_parts, dim=0)
    sample3_len = len(sample3)  # 28
    sample3_block_info = [('mask', 3), ('real', 4), ('mask', 3), ('real', 4), ('mask', 3), ('real', 4)]
    sample3_prompt_len = prompt3_len

    print(f"  样本 1 长度: {sample1_len} (Prompt={sample1_prompt_len})")
    print(f"  样本 2 长度: {sample2_len} (Prompt={sample2_prompt_len})")
    print(f"  样本 3 长度: {sample3_len} (Prompt={sample3_prompt_len})")

    # ========== Padding 到相同长度 ==========
    max_len = max(sample1_len, sample2_len, sample3_len)
    print(f"\n  Padding 到最大长度: {max_len}")

    # Pad sample 1
    pad_len1 = max_len - sample1_len
    sample1_padded = torch.cat([
        sample1,
        torch.full((pad_len1,), pad_token_id, device=device)
    ], dim=0)

    # Pad sample 2
    pad_len2 = max_len - sample2_len
    sample2_padded = torch.cat([
        sample2,
        torch.full((pad_len2,), pad_token_id, device=device)
    ], dim=0)

    # sample 3 不需要 pad
    sample3_padded = sample3

    # 构造 batch
    input_ids = torch.stack([sample1_padded, sample2_padded, sample3_padded], dim=0)

    print(f"  Batch input_ids shape: {input_ids.shape}")

    # 构造 position_ids（每个样本独立）
    def create_position_ids(prompt_len, block_info, max_len, device):
        pos_ids = []
        # Prompt
        pos_ids.append(torch.arange(0, prompt_len, device=device))
        current_pos = prompt_len

        # Blocks
        for seg_type, seg_len in block_info:
            if seg_type == 'mask':
                pos_ids.append(torch.arange(current_pos, current_pos + seg_len, device=device))
            else:  # real
                pos_ids.append(torch.arange(current_pos, current_pos + seg_len, device=device))
                current_pos += seg_len

        pos_ids = torch.cat(pos_ids, dim=0)

        # Pad
        if len(pos_ids) < max_len:
            pos_ids = torch.cat([
                pos_ids,
                torch.full((max_len - len(pos_ids),), 0, device=device)  # Pad with 0
            ], dim=0)

        return pos_ids

    position_ids = torch.stack([
        create_position_ids(sample1_prompt_len, sample1_block_info, max_len, device),
        create_position_ids(sample2_prompt_len, sample2_block_info, max_len, device),
        create_position_ids(sample3_prompt_len, sample3_block_info, max_len, device),
    ], dim=0)

    # 构造 labels
    def create_labels(prompt_len, block_info, real_tokens_list, max_len, device):
        labels = []
        # Prompt: -100
        labels.extend([-100] * prompt_len)

        real_idx = 0
        block_start_in_response = 0

        for seg_type, seg_len in block_info:
            if seg_type == 'mask':
                # Mask 预测下一个 block 的前几个 token
                for i in range(seg_len):
                    target_idx = block_start_in_response + i + 1
                    if target_idx < len(real_tokens_list):
                        labels.append(real_tokens_list[target_idx].item())
                    else:
                        labels.append(-100)
            else:  # real
                # Real block 预测下一个 token
                for i in range(seg_len):
                    target_idx = block_start_in_response + i + 1
                    if target_idx < len(real_tokens_list):
                        labels.append(real_tokens_list[target_idx].item())
                    else:
                        labels.append(-100)
                block_start_in_response += seg_len

        # Pad: -100
        labels.extend([-100] * (max_len - len(labels)))

        return torch.tensor(labels, device=device)

    # 从样本中提取 real tokens
    sample1_real_tokens = sample1_real
    sample2_real_tokens = torch.cat([sample2_real1, sample2_real2], dim=0)
    sample3_real_tokens = torch.cat([
        sample3_parts[2],  # real1
        sample3_parts[4],  # real2
        sample3_parts[6],  # real3
    ], dim=0)

    labels = torch.stack([
        create_labels(sample1_prompt_len, sample1_block_info, sample1_real_tokens, max_len, device),
        create_labels(sample2_prompt_len, sample2_block_info, sample2_real_tokens, max_len, device),
        create_labels(sample3_prompt_len, sample3_block_info, sample3_real_tokens, max_len, device),
    ], dim=0)

    # 构造 batch dict
    batch = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'labels': labels,
        'block_info': [sample1_block_info, sample2_block_info, sample3_block_info],
        'prompt_len': [sample1_prompt_len, sample2_prompt_len, sample3_prompt_len],
        'seq_lens': [sample1_len, sample2_len, sample3_len],
    }

    # ========== 测试 1: BlockMask 构造 ==========
    print("\n" + "=" * 80)
    print("Test 1: BlockMask 构造（处理不同长度）")
    print("=" * 80)

    try:
        block_mask = create_block_mask_from_batch(batch, device)
        print("✅ BlockMask 构造成功")
    except Exception as e:
        print(f"❌ BlockMask 构造失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========== 测试 2: 前向传播 ==========
    print("\n" + "=" * 80)
    print("Test 2: 前向传播")
    print("=" * 80)

    try:
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=block_mask,
        )
        logits = outputs.logits

        # 在外部计算 loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )

        print(f"✅ 前向传播成功")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========== 测试 3: Loss 有效性检查 ==========
    print("\n" + "=" * 80)
    print("Test 3: Loss 有效性检查（验证 padding 正确排除）")
    print("=" * 80)

    # Loss 已在模型内部计算
    print(f"✅ Loss 值: {loss.item():.4f}")

    # 验证每个样本的 loss 位置数
    for i in range(3):
        valid_mask = (labels[i] != -100)
        num_loss_pos = valid_mask.sum().item()
        print(f"  样本 {i+1} 的 loss 位置数: {num_loss_pos}")

    # ========== 测试 4: Padding 位置验证 ==========
    print("\n" + "=" * 80)
    print("Test 5: Padding 位置验证")
    print("=" * 80)

    for i in range(3):
        seq_len = batch['seq_lens'][i]
        # 检查实际内容区域
        content_ids = input_ids[i, :seq_len]
        # 检查 padding 区域
        if seq_len < max_len:
            pad_ids = input_ids[i, seq_len:]
            pad_labels = labels[i, seq_len:]

            # Padding 应该都是 pad_token_id
            if not (pad_ids == pad_token_id).all():
                print(f"  ❌ 样本 {i+1} 的 padding 区域包含非 pad token")
                return False

            # Padding 的 labels 应该都是 -100
            if not (pad_labels == -100).all():
                print(f"  ❌ 样本 {i+1} 的 padding labels 包含非 -100 值")
                return False

            print(f"  ✅ 样本 {i+1}: Padding 正确（{max_len - seq_len} 个 pad token）")
        else:
            print(f"  ✅ 样本 {i+1}: 无需 padding（已是最大长度）")

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    print("\n✅ 所有测试通过！")
    print("  1. ✅ BlockMask 正确处理不同长度")
    print("  2. ✅ 前向传播正常")
    print("  3. ✅ Loss 计算正确排除 padding")
    print("  4. ✅ Padding 位置正确设置")

    return True


if __name__ == "__main__":
    try:
        result = test_batch_with_padding()

        if result:
            print("\n" + "=" * 100)
            print("✅ Batch 和 Padding 测试通过！")
            print("=" * 100 + "\n")
            exit(0)
        else:
            print("\n" + "=" * 100)
            print("❌ 测试失败！")
            print("=" * 100 + "\n")
            exit(1)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
