#!/usr/bin/env python3
"""
测试 SFTDataset 的验证日志输出
"""

import sys
sys.path.insert(0, '/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning')

import logging
from dllm_reasoning.trainer.sft_dataset import SFTDataset

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def main():
    print("🧪 测试 SFTDataset 验证日志输出\n")

    # 初始化 dataset
    dataset = SFTDataset(
        parquet_files="/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet",
        tokenizer="/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/checkpoints/iterative_refine/global_step_17172/huggingface",
        prompt_key="prompt",
        response_key="target",
        max_length=4096,
        truncation="error",
    )

    print(f"\n✅ Dataset 初始化完成,共 {len(dataset)} 个样本\n")

    # 获取第一个样本 (会触发验证日志)
    print("📝 获取第一个样本 (会触发验证日志)...\n")
    sample = dataset[0]

    print(f"\n✅ 第一个样本处理完成")
    print(f"   - input_ids shape: {sample['input_ids'].shape}")
    print(f"   - attention_mask shape: {sample['attention_mask'].shape}")
    print(f"   - loss_mask shape: {sample['loss_mask'].shape}")

    # 获取第二个样本 (不会触发验证日志)
    print(f"\n📝 获取第二个样本 (不应触发验证日志)...\n")
    sample2 = dataset[1]
    print(f"✅ 第二个样本处理完成 (无额外日志)")

    print("\n" + "="*80)
    print("✅ 测试完成!")
    print("="*80)

if __name__ == "__main__":
    main()
