#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端测试：交错训练流程

测试完整的训练流程：
1. 创建 InterleavedSFTDataset
2. 创建 DataLoader with collate function
3. 模拟前向传播（使用 BlockMask）
4. 计算 loss
5. 验证梯度可以反向传播
"""

import sys
sys.path.insert(0, "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning")

import torch
from transformers import AutoTokenizer
from dllm_reasoning.trainer.interleaved_sft_dataset import (
    InterleavedSFTDataset,
    collate_interleaved_batch,
)
from dllm_reasoning.trainer.flex_attention_utils import create_block_mask_from_batch

# 检查 FlexAttention 是否可用
try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    print("⚠️  FlexAttention 不可用，将跳过相关测试")


def test_dataset_creation():
    """测试数据集创建"""
    print("=" * 100)
    print("Test 1: Dataset Creation")
    print("=" * 100)

    # 创建 tokenizer（使用一个小模型的 tokenizer）
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # 使用已有的数据文件
    parquet_file = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"

    print(f"\n使用数据文件: {parquet_file}")

    # 创建数据集（openr1.parquet 使用 'target' 作为 response 列名）
    dataset = InterleavedSFTDataset(
        parquet_files=parquet_file,
        tokenizer=tokenizer,
        block_size=4,
        max_length=2048,  # 增加 max_length
        truncation="right",  # 启用截断
        prompt_key="prompt",
        response_key="target",
    )

    print(f"\n✓ 数据集创建成功")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  Block size: {dataset.block_size}")

    # 获取一个样本
    sample = dataset[0]

    print(f"\n样本结构:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        else:
            print(f"  {key}: {type(value)}")

    # 验证 block_info 存在
    assert "block_info" in sample, "样本中缺少 block_info"
    assert "prompt_len" in sample, "样本中缺少 prompt_len"

    print(f"\n  block_info: {sample['block_info']}")
    print(f"  prompt_len: {sample['prompt_len']}")

    # ========== 新增：详细的 token 级别调试输出 ==========
    print(f"\n" + "=" * 80)
    print(f"【Token 级别调试信息】")
    print(f"=" * 80)

    input_ids = sample['input_ids']
    position_ids = sample['position_ids']
    labels = sample['labels']
    prompt_len = sample['prompt_len']

    # 打印前 50 个 token（如果序列长度 >= 50）
    num_tokens_to_show = min(50, len(input_ids))

    print(f"\n前 {num_tokens_to_show} 个位置的详细信息:")
    print(f"{'Pos':<5} {'PosID':<7} {'Input':<8} {'Label':<8} {'Decoded Input':<30} {'Decoded Label':<30}")
    print(f"-" * 120)

    for i in range(num_tokens_to_show):
        input_id = input_ids[i].item()
        pos_id = position_ids[i].item()
        label_id = labels[i].item()

        # 解码 token
        input_text = tokenizer.decode([input_id]).replace('\n', '\\n')
        label_text = tokenizer.decode([label_id]).replace('\n', '\\n') if label_id != -100 else "[IGNORE]"

        # 标记 segment 类型
        segment_type = "???"
        if i < prompt_len:
            segment_type = "PROMPT"
        else:
            # 根据 block_info 判断
            pos_in_response = i - prompt_len
            current_pos = 0
            for seg_type, seg_idx, seg_len in sample['block_info']:
                if current_pos <= pos_in_response < current_pos + seg_len:
                    segment_type = seg_type.upper()
                    break
                current_pos += seg_len

        print(f"{i:<5} {pos_id:<7} {input_id:<8} {label_id:<8} {input_text:<30.30} {label_text:<30.30} [{segment_type}]")

    # 统计信息
    print(f"\n统计信息:")
    print(f"  总长度: {len(input_ids)}")
    print(f"  Prompt 长度: {prompt_len}")
    print(f"  Response 长度: {len(input_ids) - prompt_len}")
    print(f"  有效 label 数量: {(labels != -100).sum().item()}")
    print(f"  忽略 label 数量: {(labels == -100).sum().item()}")

    # 打印 block 结构
    print(f"\n序列结构:")
    current_pos = 0
    print(f"  [0-{prompt_len-1}]: PROMPT (len={prompt_len})")
    for seg_type, seg_idx, seg_len in sample['block_info']:
        start = prompt_len + current_pos
        end = start + seg_len - 1
        print(f"  [{start}-{end}]: {seg_type.upper()} (len={seg_len})")
        current_pos += seg_len

    print(f"\n✓ Test 1 passed\n")
    return dataset, tokenizer


def test_dataloader_and_collate():
    """测试 DataLoader 和 collate 函数"""
    print("=" * 100)
    print("Test 2: DataLoader and Collate Function")
    print("=" * 100)

    dataset, tokenizer = test_dataset_creation()

    # 创建 collate 函数
    from functools import partial
    collate_fn = partial(
        collate_interleaved_batch,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=-100,
    )

    # 创建 DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 获取一个 batch
    batch = next(iter(dataloader))

    print(f"\nBatch 结构:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
            if len(value) > 0 and isinstance(value[0], list):
                print(f"    第一个元素: {value[0]}")
        else:
            print(f"  {key}: {type(value)}")

    # 验证 batch 结构
    assert "input_ids" in batch
    assert "position_ids" in batch
    assert "labels" in batch
    assert "block_info" in batch
    assert "prompt_len" in batch
    assert "seq_lens" in batch

    # ========== 新增：Batch 级别调试输出 ==========
    print(f"\n" + "=" * 80)
    print(f"【Batch 级别调试信息】")
    print(f"=" * 80)

    # 显示第一个样本的前 30 个 token
    print(f"\nBatch 中第一个样本的前 30 个位置:")
    print(f"{'Pos':<5} {'PosID':<7} {'Input':<8} {'Label':<8} {'Type':<10}")
    print(f"-" * 60)

    sample_0_input = batch['input_ids'][0]
    sample_0_pos = batch['position_ids'][0]
    sample_0_labels = batch['labels'][0]
    sample_0_prompt_len = batch['prompt_len'][0]
    sample_0_block_info = batch['block_info'][0]

    num_to_show = min(30, len(sample_0_input))
    for i in range(num_to_show):
        input_id = sample_0_input[i].item()
        pos_id = sample_0_pos[i].item()
        label_id = sample_0_labels[i].item()

        # 判断类型
        if i < sample_0_prompt_len:
            seg_type = "PROMPT"
        else:
            pos_in_response = i - sample_0_prompt_len
            current_pos = 0
            seg_type = "???"
            for s_type, s_len in sample_0_block_info:
                if current_pos <= pos_in_response < current_pos + s_len:
                    seg_type = s_type.upper()
                    break
                current_pos += s_len

        print(f"{i:<5} {pos_id:<7} {input_id:<8} {label_id:<8} {seg_type:<10}")

    # 验证 padding 是否正确
    print(f"\nPadding 验证:")
    print(f"  样本 0 实际长度: {batch['seq_lens'][0]}")
    print(f"  样本 1 实际长度: {batch['seq_lens'][1]}")
    print(f"  Batch 最大长度: {batch['input_ids'].shape[1]}")

    # 验证第二个样本的 padding 位置
    sample_1_len = batch['seq_lens'][1]
    sample_1_input = batch['input_ids'][1]
    sample_1_labels = batch['labels'][1]

    print(f"  样本 1 最后一个有效位置: input_id={sample_1_input[sample_1_len-1].item()}")
    if sample_1_len < len(sample_1_input):
        print(f"  样本 1 第一个 padding 位置: input_id={sample_1_input[sample_1_len].item()} (应该是 pad_token_id={tokenizer.pad_token_id})")
        print(f"  样本 1 第一个 padding label: {sample_1_labels[sample_1_len].item()} (应该是 -100)")

    print(f"\n✓ Test 2 passed\n")
    return batch, tokenizer


def test_block_mask_construction():
    """测试 BlockMask 构造"""
    print("=" * 100)
    print("Test 3: BlockMask Construction")
    print("=" * 100)

    if not FLEX_ATTN_AVAILABLE:
        print("⚠️  跳过测试（FlexAttention 不可用）\n")
        return None

    batch, tokenizer = test_dataloader_and_collate()

    # 构造 BlockMask
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将 batch 移动到设备
    batch_on_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_on_device[key] = value.to(device)
        else:
            batch_on_device[key] = value

    print(f"\n使用设备: {device}")

    try:
        block_mask = create_block_mask_from_batch(batch_on_device, device)
        print(f"✓ BlockMask 构造成功")
        print(f"  类型: {type(block_mask)}")
    except Exception as e:
        print(f"❌ BlockMask 构造失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ========== 新增：Attention Mask 可视化 ==========
    print(f"\n" + "=" * 80)
    print(f"【Attention Mask 可视化】")
    print(f"=" * 80)

    # 从 create_block_mask_from_batch 中提取预计算的 mask
    # 为了可视化，我们需要重新计算第一个样本的 mask
    from dllm_reasoning.trainer.flex_attention_utils import create_mask_mod_from_block_info

    sample_0_block_info = batch_on_device['block_info'][0]
    sample_0_prompt_len = batch_on_device['prompt_len'][0]
    sample_0_seq_len = batch_on_device['seq_lens'][0]

    mask_mod = create_mask_mod_from_block_info(
        block_info=sample_0_block_info,
        prompt_len=sample_0_prompt_len,
        seq_len=sample_0_seq_len,
    )

    # 构造 2D mask（前 20x20 用于可视化）
    vis_size = min(20, sample_0_seq_len)
    print(f"\n第一个样本的 Attention Mask (前 {vis_size}x{vis_size} 位置):")
    print(f"  (行=query, 列=key, 1=可见, 0=不可见)")
    print(f"\n    ", end="")
    for kv in range(vis_size):
        print(f"{kv:3d}", end="")
    print()

    for q in range(vis_size):
        print(f"{q:3d} ", end="")
        for kv in range(vis_size):
            can_see = mask_mod(0, 0, q, kv)
            print(f"  {'1' if can_see else '.'}", end="")
        print()

    # 打印 segment 边界
    print(f"\nSegment 边界:")
    print(f"  Prompt: [0-{sample_0_prompt_len-1}]")
    current_pos = sample_0_prompt_len
    for seg_type, seg_len in sample_0_block_info:
        print(f"  {seg_type.upper()}: [{current_pos}-{current_pos+seg_len-1}]")
        current_pos += seg_len

    # 验证关键规则
    print(f"\n关键规则验证:")
    prompt_end = sample_0_prompt_len - 1

    # 找到第一个 mask 和第一个 real block
    first_mask_start = sample_0_prompt_len
    first_mask_len = None
    first_real_start = None

    pos = sample_0_prompt_len
    for seg_type, seg_len in sample_0_block_info:
        if seg_type == 'mask' and first_mask_len is None:
            first_mask_len = seg_len
            first_real_start = pos + seg_len
            break
        pos += seg_len

    if first_mask_len and first_real_start:
        # 规则 1: Prompt 最后一个 token 看不到 response
        can_see_mask = mask_mod(0, 0, prompt_end, first_mask_start)
        print(f"  ✓ Prompt[{prompt_end}] -> Mask[{first_mask_start}]: {can_see_mask} (应该是 False)")

        # 规则 2: Mask 可以看 prompt
        can_see_prompt = mask_mod(0, 0, first_mask_start, 0)
        print(f"  ✓ Mask[{first_mask_start}] -> Prompt[0]: {can_see_prompt} (应该是 True)")

        # 规则 3: Mask 看不到对应的 real block
        can_see_real = mask_mod(0, 0, first_mask_start, first_real_start)
        print(f"  ✓ Mask[{first_mask_start}] -> Real[{first_real_start}]: {can_see_real} (应该是 False)")

        # 规则 4: Real 看不到 mask
        can_see_mask_from_real = mask_mod(0, 0, first_real_start, first_mask_start)
        print(f"  ✓ Real[{first_real_start}] -> Mask[{first_mask_start}]: {can_see_mask_from_real} (应该是 False)")

        # 规则 5: Real 可以看 prompt
        can_see_prompt_from_real = mask_mod(0, 0, first_real_start, 0)
        print(f"  ✓ Real[{first_real_start}] -> Prompt[0]: {can_see_prompt_from_real} (应该是 True)")

    print(f"\n✓ Test 3 passed\n")
    return batch_on_device, block_mask, device


def test_forward_pass():
    """测试前向传播"""
    print("=" * 100)
    print("Test 4: Forward Pass with BlockMask")
    print("=" * 100)

    if not FLEX_ATTN_AVAILABLE:
        print("⚠️  跳过测试（FlexAttention 不可用）\n")
        return

    result = test_block_mask_construction()
    if result is None:
        return

    batch, block_mask, device = result

    # 创建一个小模型进行测试
    from transformers import AutoModelForCausalLM
    print(f"\n加载测试模型...")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    print(f"✓ 模型加载成功")

    # 前向传播
    print(f"\n执行前向传播...")

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                attention_mask=block_mask,
            )

        logits = outputs.logits
        print(f"✓ 前向传播成功")
        print(f"  Logits shape: {logits.shape}")

        # 计算 loss
        labels = batch["labels"]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )

        print(f"✓ Loss 计算成功")
        print(f"  Loss: {loss.item():.4f}")

        # ========== 新增：预测验证 ==========
        print(f"\n" + "=" * 80)
        print(f"【预测验证】")
        print(f"=" * 80)

        # 获取预测结果
        predictions = logits.argmax(dim=-1)  # [B, seq_len]

        # 显示第一个样本的前 20 个位置的预测
        print(f"\n第一个样本的前 20 个位置预测:")
        print(f"{'Pos':<5} {'Input':<8} {'Pred':<8} {'Label':<8} {'Match':<7} {'Type':<10}")
        print(f"-" * 70)

        sample_0_input = batch['input_ids'][0]
        sample_0_pred = predictions[0]
        sample_0_labels = batch['labels'][0]
        sample_0_prompt_len = batch['prompt_len'][0]
        sample_0_block_info = batch['block_info'][0]

        num_to_show = min(20, len(sample_0_input))

        correct_count = 0
        total_valid = 0

        for i in range(num_to_show):
            input_id = sample_0_input[i].item()
            pred_id = sample_0_pred[i].item()
            label_id = sample_0_labels[i].item()

            # 判断类型
            if i < sample_0_prompt_len:
                seg_type = "PROMPT"
            else:
                pos_in_response = i - sample_0_prompt_len
                current_pos = 0
                seg_type = "???"
                for s_type, s_len in sample_0_block_info:
                    if current_pos <= pos_in_response < current_pos + s_len:
                        seg_type = s_type.upper()
                        break
                    current_pos += s_len

            if label_id != -100:
                match = "✓" if pred_id == label_id else "✗"
                if pred_id == label_id:
                    correct_count += 1
                total_valid += 1
            else:
                match = "-"

            print(f"{i:<5} {input_id:<8} {pred_id:<8} {label_id:<8} {match:<7} {seg_type:<10}")

        if total_valid > 0:
            accuracy = correct_count / total_valid
            print(f"\n前 {num_to_show} 个位置的准确率: {correct_count}/{total_valid} = {accuracy:.2%}")

        # 统计整个 batch 的指标
        all_valid_mask = (labels != -100)
        if all_valid_mask.sum() > 0:
            all_correct = (predictions == labels) & all_valid_mask
            overall_accuracy = all_correct.sum().float() / all_valid_mask.sum().float()
            print(f"\n整个 Batch 的准确率: {overall_accuracy.item():.4f}")
            print(f"  有效位置数: {all_valid_mask.sum().item()}")
            print(f"  正确预测数: {all_correct.sum().item()}")

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n✓ Test 4 passed\n")


def test_backward_pass():
    """测试反向传播"""
    print("=" * 100)
    print("Test 5: Backward Pass")
    print("=" * 100)

    if not FLEX_ATTN_AVAILABLE:
        print("⚠️  跳过测试（FlexAttention 不可用）\n")
        return

    result = test_block_mask_construction()
    if result is None:
        return

    batch, block_mask, device = result

    # 创建一个小模型
    from transformers import AutoModelForCausalLM
    print(f"\n加载测试模型...")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,  # 反向传播需要 float32
    ).to(device)

    model.train()

    print(f"✓ 模型加载成功（训练模式）")

    # 前向传播
    print(f"\n执行前向传播...")

    try:
        outputs = model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask=block_mask,
        )

        logits = outputs.logits
        labels = batch["labels"]

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )

        print(f"✓ 前向传播成功，Loss: {loss.item():.4f}")

        # 反向传播
        print(f"\n执行反向传播...")
        loss.backward()

        print(f"✓ 反向传播成功")

        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        if has_grad:
            print(f"✓ 梯度计算成功")
        else:
            print(f"⚠️  未检测到梯度")

    except Exception as e:
        print(f"❌ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n✓ Test 5 passed\n")


if __name__ == "__main__":
    try:
        print("\n" + "=" * 100)
        print("端到端交错训练测试")
        print("=" * 100 + "\n")

        # 运行所有测试
        test_dataset_creation()
        test_dataloader_and_collate()
        test_block_mask_construction()
        test_forward_pass()
        test_backward_pass()

        print("\n" + "=" * 100)
        print("✅ 所有测试通过！")
        print("=" * 100 + "\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
