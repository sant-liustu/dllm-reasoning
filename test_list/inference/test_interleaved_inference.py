#!/usr/bin/env python3
"""
测试 Interleaved 生成器 (Next Block Prediction)

目标:
1. 验证推理代码的正确性
2. 测试模型的多步解码能力 (0 mask, 1 mask, 2 mask, ...)
3. 对比不同 mask 数量下的生成结果

测试场景:
- num_masks=0: 标准自回归 (每次预测1个token)
- num_masks=1: 跳1步预测 (每次预测2个token)
- num_masks=2: 跳2步预测 (每次预测3个token)
- num_masks=3: 跳3步预测 (每次预测4个token)
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.inference.interleaved_generator import interleaved_generate

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_interleaved_generation(
    model_path: str,
    test_prompts: list,
    num_masks_list: list = [0, 1, 2, 3],
    max_new_tokens: int = 50,
):
    """
    测试不同 mask 数量下的生成效果

    Args:
        model_path: 模型路径
        test_prompts: 测试 prompt 列表
        num_masks_list: 要测试的 mask 数量列表
        max_new_tokens: 最大生成 token 数
    """
    print("="*100)
    print("🚀 测试 Interleaved 生成器 (Next Block Prediction)")
    print("="*100)

    # ========== 加载模型 ==========
    print(f"\n📦 加载模型: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True  # 使用本地模型，避免路径验证问题
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    print(f"   ✅ 模型加载完成")

    # 获取特殊 token ID
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else eos_token_id

    # 获取 mask token ID
    # 检查是否有 mask_token
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
        mask_token_id = tokenizer.mask_token_id
        print(f"   Mask token: {tokenizer.mask_token} (ID: {mask_token_id})")
    else:
        # 如果没有 mask_token，使用 unk_token 或 eos_token
        if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
            mask_token_id = tokenizer.unk_token_id
            print(f"   ⚠️  没有 mask_token，使用 unk_token: {tokenizer.unk_token} (ID: {mask_token_id})")
        else:
            mask_token_id = eos_token_id
            print(f"   ⚠️  没有 mask_token 和 unk_token，使用 eos_token (ID: {mask_token_id})")

    print(f"   EOS token ID: {eos_token_id}")
    print(f"   PAD token ID: {pad_token_id}")

    # ========== 对每个 prompt 进行测试 ==========
    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\n" + "="*100)
        print(f"📝 Prompt {prompt_idx + 1}/{len(test_prompts)}")
        print(f"="*100)
        print(f"内容: {prompt}")
        print()

        # 准备输入
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        print(f"输入长度: {input_ids.size(1)} tokens")

        # 对每个 num_masks 进行测试
        results = {}

        for num_masks in num_masks_list:
            print(f"\n{'-'*100}")
            print(f"🔬 测试 num_masks={num_masks} (每次预测 {num_masks+1} 个 token)")
            print(f"{'-'*100}")

            try:
                # 生成
                with torch.no_grad():
                    output_ids = interleaved_generate(
                        model=model,
                        input_ids=input_ids,
                        eos_token_id=eos_token_id,
                        mask_token_id=mask_token_id,
                        pad_token_id=pad_token_id,
                        max_new_tokens=max_new_tokens,
                        num_masks=num_masks,
                        max_length=8192,
                        verbose=False,  # 设为 True 可以看到详细过程
                        tokenizer=tokenizer,
                    )

                # 解码
                full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_text = tokenizer.decode(
                    output_ids[0, input_ids.size(1):],
                    skip_special_tokens=True
                )

                # 统计
                total_len = output_ids.size(1)
                generated_len = total_len - input_ids.size(1)
                tokens_per_block = num_masks + 1
                num_blocks = (generated_len + tokens_per_block - 1) // tokens_per_block

                results[num_masks] = {
                    'generated_text': generated_text,
                    'generated_len': generated_len,
                    'num_blocks': num_blocks,
                    'success': True
                }

                print(f"✅ 生成成功")
                print(f"   生成长度: {generated_len} tokens")
                print(f"   块数: {num_blocks} blocks (每块 {tokens_per_block} tokens)")
                print(f"   生成内容: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")

            except Exception as e:
                print(f"❌ 生成失败: {str(e)}")
                results[num_masks] = {
                    'generated_text': None,
                    'generated_len': 0,
                    'num_blocks': 0,
                    'success': False,
                    'error': str(e)
                }

        # ========== 结果对比 ==========
        print(f"\n{'='*100}")
        print(f"📊 结果对比 (Prompt {prompt_idx + 1})")
        print(f"{'='*100}")

        print(f"\n{'num_masks':<12} {'每块tokens':<12} {'块数':<10} {'总tokens':<12} {'状态':<10}")
        print(f"{'-'*60}")
        for num_masks in num_masks_list:
            res = results[num_masks]
            tokens_per_block = num_masks + 1
            status = "✅ 成功" if res['success'] else "❌ 失败"
            print(f"{num_masks:<12} {tokens_per_block:<12} {res['num_blocks']:<10} {res['generated_len']:<12} {status:<10}")

        # 检查生成的多样性
        print(f"\n生成内容是否相同:")
        unique_texts = set()
        for num_masks in num_masks_list:
            if results[num_masks]['success']:
                text = results[num_masks]['generated_text']
                unique_texts.add(text)

        if len(unique_texts) == 1:
            print(f"   ✅ 所有 num_masks 生成的内容完全相同")
            print(f"   说明: 模型对不同的 mask 数量给出了一致的预测")
        else:
            print(f"   ⚠️  不同 num_masks 生成了不同的内容 ({len(unique_texts)} 种)")
            print(f"   说明: 不同的 mask 数量可能影响了预测结果")

    print(f"\n{'='*100}")
    print(f"✅ 测试完成")
    print(f"{'='*100}")


def main():
    """主函数"""
    # ========== 配置 ==========
    # 使用 Interleaved SFT 训练完成的模型
    MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"

    TEST_PROMPTS = [
        "What is 2+2?",
        "写一首关于春天的诗",
        "Solve: If x + 5 = 10, what is x?",
    ]

    NUM_MASKS_LIST = [0, 1, 2, 3]  # 测试不同的 mask 数量
    MAX_NEW_TOKENS = 50  # 限制生成长度，便于快速测试

    # ========== 运行测试 ==========
    test_interleaved_generation(
        model_path=MODEL_PATH,
        test_prompts=TEST_PROMPTS,
        num_masks_list=NUM_MASKS_LIST,
        max_new_tokens=MAX_NEW_TOKENS,
    )


if __name__ == "__main__":
    main()
