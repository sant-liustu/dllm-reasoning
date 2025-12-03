#!/usr/bin/env python3
"""
测试修改后的模型能否正确在内部创建 BlockMask
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset


def test_model_block_mask_creation():
    """测试模型能否正确在内部创建 BlockMask"""

    # 使用本地源代码模型（而不是 checkpoint）
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/model/DLLM"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("测试模型内部 BlockMask 创建")
    print("="*80)
    print(f"\n使用本地模型: {MODEL_PATH}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载本地模型
    print("加载本地模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B",
        trust_remote_code=True
    )

    # 从本地代码加载模型
    from dllm_reasoning.model.DLLM.modeling_dllm import DLLMForCausalLM
    from dllm_reasoning.model.DLLM.configuration_dllm import DLLMConfig

    config = DLLMConfig.from_pretrained("dllm_reasoning/dllm_reasoning/models/DLLM-1.5B")
    model = DLLMForCausalLM(config)

    # 加载预训练权重
    from transformers import AutoModel
    pretrained = AutoModelForCausalLM.from_pretrained(
        "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.load_state_dict(pretrained.state_dict())
    model = model.to(device).to(torch.bfloat16)

    # 设置为训练模式
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
    print(f"  总序列长度: {sample['input_ids'].shape[0]}")
    print(f"  Prompt长度: {sample['prompt_len']}")
    print(f"  Block数量: {len(sample['block_info'])}")

    # 准备batch（batch_size=1）
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    position_ids = sample['position_ids'].unsqueeze(0).to(device)
    labels = sample['labels'].unsqueeze(0).to(device)
    block_info = [sample['block_info']]
    prompt_len = [sample['prompt_len']]
    seq_lens = [input_ids.size(1)]

    print(f"\n测试1: 训练模式下传入 block_info 参数")
    print("-" * 80)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                block_info=block_info,
                prompt_len=prompt_len,
                seq_lens=seq_lens,
                use_cache=False,
            )

        print(f"✅ 成功！模型正确创建了 BlockMask")
        print(f"  Logits shape: {outputs.logits.shape}")

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n测试2: 训练模式下不传 block_info 参数（应该报错）")
    print("-" * 80)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            )

        print(f"❌ 意外：应该报错但成功了！")
        return False

    except ValueError as e:
        print(f"✅ 正确报错: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n测试3: 推理模式下不需要 block_info 参数")
    print("-" * 80)

    model.eval()

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, :100],  # 只用前100个token测试
                position_ids=position_ids[:, :100],
                use_cache=False,
            )

        print(f"✅ 成功！推理模式正常工作")
        print(f"  Logits shape: {outputs.logits.shape}")

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n" + "="*80)
    print(f"✅ 所有测试通过！模型修改成功")
    print(f"="*80)

    return True


if __name__ == "__main__":
    success = test_model_block_mask_creation()
    sys.exit(0 if success else 1)
