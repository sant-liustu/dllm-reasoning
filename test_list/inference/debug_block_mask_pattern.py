#!/usr/bin/env python3
"""
详细检查[P][M₁]和[P][M₁][R₁]两种配置下的BlockMask pattern

验证M₁位置的attention pattern是否真的相同
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.attention.flex_attention import create_block_mask

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset


def visualize_block_mask(block_info, seq_len, prompt_len, device):
    """
    可视化BlockMask pattern

    Args:
        block_info: list of (block_type, block_len)
        seq_len: 序列长度
        prompt_len: prompt长度
    """
    # 构造mask函数（复制自InterleavedSFTDataset）
    def block_mask_mod(b, h, q_idx, kv_idx):
        # 计算每个位置的block_id和block内的相对位置
        # prompt部分的block_id为-1
        q_is_prompt = q_idx < prompt_len
        kv_is_prompt = kv_idx < prompt_len

        # 计算block positions
        q_block_pos = q_idx - prompt_len
        kv_block_pos = kv_idx - prompt_len

        # 计算block_id（每个block 4个token）
        block_size = 4
        q_block_id = q_block_pos // block_size
        kv_block_id = kv_block_pos // block_size

        # Mask rules:
        # 1. Prompt can see all prompt
        # 2. Response can see all prompt + current and previous blocks
        # 3. Mask can only see prompt + current block's mask positions

        # Determine block types
        # 根据block_info确定每个block的类型
        cumsum = 0
        q_block_type = None
        kv_block_type = None

        # 简化实现：直接用block_info判断
        # 这里我们用一个简单的方法：根据block_info构造一个查找表

        # For now, let's use a simpler version:
        # Assume alternating [mask, real, mask, real, ...]
        # This is a placeholder - we'll need to match the actual dataset logic

        # Causal mask within blocks
        causal = q_idx >= kv_idx

        # Prompt sees all prompt
        prompt_to_prompt = q_is_prompt & kv_is_prompt

        # Response sees prompt
        resp_to_prompt = (~q_is_prompt) & kv_is_prompt & causal

        # Within response blocks
        same_block = q_block_id == kv_block_id
        prev_block = q_block_id > kv_block_id

        # Response sees current and previous blocks
        resp_to_resp = (~q_is_prompt) & (~kv_is_prompt) & (same_block | prev_block) & causal

        return prompt_to_prompt | resp_to_prompt | resp_to_resp

    # 创建BlockMask
    block_mask = create_block_mask(
        block_mask_mod,
        B=1, H=None,
        Q_LEN=seq_len, KV_LEN=seq_len,
    )

    # 提取mask矩阵（通过手动计算）
    mask_matrix = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for q in range(seq_len):
        for kv in range(seq_len):
            mask_matrix[q, kv] = block_mask_mod(None, None,
                                                torch.tensor(q, device=device),
                                                torch.tensor(kv, device=device))

    return mask_matrix


def compare_masks(model, tokenizer, sample, device):
    """
    对比[P][M₁]和[P][M₁][R₁]两种配置的mask pattern
    """
    print("="*80)
    print("BlockMask Pattern对比")
    print("="*80)

    prompt_len = sample['prompt_len']
    m1_pos = prompt_len

    # 配置1: [P][M₁]
    print("\n配置1: [P][M₁]")
    truncate_pos_1 = prompt_len + 3
    block_info_1 = [('mask', 3)]

    print(f"  序列长度: {truncate_pos_1}")
    print(f"  Block info: {block_info_1}")
    print(f"  M₁位置: {m1_pos}")

    # 配置2: [P][M₁][R₁]
    print("\n配置2: [P][M₁][R₁]")
    truncate_pos_2 = prompt_len + 7
    block_info_2 = [('mask', 3), ('real', 4)]

    print(f"  序列长度: {truncate_pos_2}")
    print(f"  Block info: {block_info_2}")
    print(f"  M₁位置: {m1_pos}")

    # 可视化M₁位置的attention pattern
    print("\n" + "="*80)
    print(f"M₁位置（pos={m1_pos}）的Attention Pattern")
    print("="*80)

    # 需要实际从dataset获取正确的mask
    # 让我们直接运行forward pass并捕获attention mask
    input_ids_full = sample['input_ids'].to(device)
    position_ids_full = sample['position_ids'].to(device)

    # 配置1
    input_ids_1 = input_ids_full[:truncate_pos_1]
    position_ids_1 = position_ids_full[:truncate_pos_1]

    # 配置2
    input_ids_2 = input_ids_full[:truncate_pos_2]
    position_ids_2 = position_ids_full[:truncate_pos_2]

    print(f"\n配置1的M₁可以attend到的位置数:")
    print(f"  理论上：M₁可以看到 Prompt(277) + M₁自己(1) = 278个位置")
    print(f"  （M₁不能看到M₂和M₃）")

    print(f"\n配置2的M₁可以attend到的位置数:")
    print(f"  理论上：M₁可以看到 Prompt(277) + M₁自己(1) = 278个位置")
    print(f"  （M₁不能看到M₂、M₃以及R₁的4个位置）")

    print(f"\n⚠️ 关键问题：")
    print(f"  如果两种配置下M₁的attention pattern完全相同，")
    print(f"  那为什么M₁的输出会不同？")

    # 打印position_ids
    print(f"\n配置1的position_ids (后10个):")
    print(f"  {position_ids_1[-10:].tolist()}")

    print(f"\n配置2的position_ids (后14个):")
    print(f"  {position_ids_2[-14:].tolist()}")

    print(f"\n⚠️ Position IDs分析：")
    print(f"  M₁的position_id在两种配置下都是: {position_ids_1[m1_pos].item()}")
    print(f"  但是后续token的position_ids可能不同！")
    print(f"\n  配置1 [P][M₁]:")
    print(f"    - Prompt: 0-276")
    print(f"    - M₁: {position_ids_1[m1_pos].item()}")
    print(f"    - M₂: {position_ids_1[m1_pos+1].item() if m1_pos+1 < len(position_ids_1) else 'N/A'}")
    print(f"    - M₃: {position_ids_1[m1_pos+2].item() if m1_pos+2 < len(position_ids_1) else 'N/A'}")

    print(f"\n  配置2 [P][M₁][R₁]:")
    print(f"    - Prompt: 0-276")
    print(f"    - M₁: {position_ids_2[m1_pos].item()}")
    print(f"    - M₂: {position_ids_2[m1_pos+1].item() if m1_pos+1 < len(position_ids_2) else 'N/A'}")
    print(f"    - M₃: {position_ids_2[m1_pos+2].item() if m1_pos+2 < len(position_ids_2) else 'N/A'}")
    print(f"    - R₁[0]: {position_ids_2[m1_pos+3].item() if m1_pos+3 < len(position_ids_2) else 'N/A'}")
    print(f"    - R₁[1]: {position_ids_2[m1_pos+4].item() if m1_pos+4 < len(position_ids_2) else 'N/A'}")
    print(f"    - R₁[2]: {position_ids_2[m1_pos+5].item() if m1_pos+5 < len(position_ids_2) else 'N/A'}")
    print(f"    - R₁[3]: {position_ids_2[m1_pos+6].item() if m1_pos+6 < len(position_ids_2) else 'N/A'}")


def main():
    MODEL_PATH = "dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
    DATA_PATH = "data/openr1.parquet"

    print("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    model.train()

    print("加载数据集...")
    dataset = InterleavedSFTDataset(
        parquet_files=DATA_PATH,
        tokenizer=tokenizer,
        prompt_key="prompt",
        response_key="target",
        block_size=4,
        max_length=6000,
        truncation="right",
    )

    sample = dataset[0]

    print(f"\n样本信息:")
    print(f"  Prompt长度: {sample['prompt_len']}")
    print(f"  总序列长度: {sample['input_ids'].shape[0]}")

    # 对比masks
    compare_masks(model, tokenizer, sample, device)


if __name__ == "__main__":
    main()
