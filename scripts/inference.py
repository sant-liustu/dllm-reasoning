#!/usr/bin/env python3
"""
测试演化追踪功能：观察 blockwise 生成的完整演化过程
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.inference.generator import iterative_generate


def main():
    # ========================================
    # 配置区域 - 根据你的实际情况修改
    # ========================================
    MODEL_PATH = "checkpoints/iterative_refine/global_step_17172/huggingface"  # 模型检查点路径
    PROMPT = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."

    # 其他测试 prompt 示例：
    # PROMPT = "what's your name"
    # PROMPT = "请用中文介绍一下你自己。"
    # ========================================


    print("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    print(f"✅ 模型加载完成\n")

    # 测试配置：使用较小的 add_eos_length 和 refine_iter，以便观察演化过程
    config = {"add_eos_length": 0, "refine_iter": 1}

    print("=" * 80)
    print(f"测试演化追踪功能")
    print(f"配置: add_eos_length={config['add_eos_length']}, refine_iter={config['refine_iter']}")
    print("=" * 80)
    print()

    # 准备输入
    messages = [{"role": "user", "content": PROMPT}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    print(f"Prompt: {PROMPT}")
    print(f"输入 token 数量: {input_ids.size(1)}")
    print()

    # 生成 - 开启演化追踪
    with torch.no_grad():
        output_ids = iterative_generate(
            model=model,
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            add_eos_length=config["add_eos_length"],
            refine_iter=config["refine_iter"],
            max_new_tokens=16384,  # 限制生成长度，只观察前几个块
            max_length=8192,
            verbose_trace=False,  # ✅ 开启演化追踪
            tokenizer=tokenizer,
        )

    print()
    print("=" * 80)
    print("最终生成结果")
    print("=" * 80)
    response = tokenizer.decode(
        output_ids[0, input_ids.size(1):],
        skip_special_tokens=True
    )
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    main()
