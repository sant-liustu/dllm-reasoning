"""
dLLM Reasoning 推理演示脚本

功能:
- 支持单个 prompt 或批量 prompts (从文件读取)
- 自动应用 chat template (可选)
- 支持多种参数配置
- 结果同时打印到终端和保存到文件

使用示例:

# 单个 prompt
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompt "What is 2+2?"

# 批量推理 (从文件)
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompts_file prompts.txt \
    --output_file results.jsonl \
    --batch_size 4

# 使用 chat template
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompt "What is 2+2?" \
    --use_chat_template

# 自定义参数
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompt "Explain quantum physics" \
    --add_eos_length 127 \
    --refine_iter 3 \
    --max_new_tokens 1024 \
    --max_length 8192
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.inference.generator import iterative_generate

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    trust_remote_code: bool = True,
) -> tuple:
    """
    加载模型和 tokenizer

    Args:
        model_path: 模型路径 (本地路径或 HuggingFace model ID)
        device: 设备 (cuda/cpu)
        trust_remote_code: 是否信任远程代码

    Returns:
        (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",  # 批量推理时需要左填充
    )

    # 确保有 pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added new pad_token: [PAD]")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    ).to(device).eval()

    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Model type: {model.__class__.__name__}")
    logger.info(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return model, tokenizer


def prepare_prompts(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    use_chat_template: bool = False,
) -> tuple:
    """
    准备 prompts (应用 chat template 如果需要)

    Args:
        prompts: 原始 prompts
        tokenizer: tokenizer
        use_chat_template: 是否应用 chat template

    Returns:
        (processed_prompts, is_templated): 处理后的 prompts 和是否已模板化标记
    """
    if not use_chat_template:
        return prompts, False

    # 检查 tokenizer 是否有 chat_template
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        logger.warning(
            "use_chat_template is True but tokenizer has no chat_template. "
            "Using prompts as-is."
        )
        return prompts, False

    # 应用 chat template - 直接返回 token IDs 避免双 BOS token
    # 返回原始 prompts 和标记，让调用者使用 apply_chat_template
    return prompts, True


def batch_generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int,
    add_eos_length: int,
    refine_iter: int,
    max_new_tokens: int,
    max_length: int,
    device: str,
    use_chat_template: bool = False,
) -> List[Dict[str, Any]]:
    """
    批量推理

    Args:
        model: 模型
        tokenizer: tokenizer
        prompts: prompts 列表
        batch_size: 批大小
        add_eos_length: 每块添加的 EOS 数量
        refine_iter: refine 轮数
        max_new_tokens: 最大生成 token 数
        max_length: 序列最大长度
        device: 设备
        use_chat_template: 是否使用 chat template

    Returns:
        结果列表，每个元素包含 prompt 和 response
    """
    results = []

    # 分批处理
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # 如果使用 chat template，直接用 apply_chat_template 获取 token IDs
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            # 为每个 prompt 构建消息并获取 token IDs
            batch_input_ids = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,  # 直接返回 token IDs，避免双 BOS
                    add_generation_prompt=True,
                    return_tensors="pt"
                )[0]  # 取出单个序列
                batch_input_ids.append(input_ids)

            # 手动填充到相同长度（左填充）
            max_len = max(len(ids) for ids in batch_input_ids)
            padded_input_ids = []
            attention_masks = []

            for input_ids in batch_input_ids:
                pad_len = max_len - len(input_ids)
                if pad_len > 0:
                    padded_ids = torch.cat([
                        torch.full((pad_len,), tokenizer.pad_token_id, dtype=input_ids.dtype),
                        input_ids
                    ])
                    attention_mask = torch.cat([
                        torch.zeros(pad_len, dtype=torch.long),
                        torch.ones(len(input_ids), dtype=torch.long)
                    ])
                else:
                    padded_ids = input_ids
                    attention_mask = torch.ones(len(input_ids), dtype=torch.long)

                padded_input_ids.append(padded_ids)
                attention_masks.append(attention_mask)

            input_ids = torch.stack(padded_input_ids).to(device)
            attention_mask = torch.stack(attention_masks).to(device)
        else:
            # 不使用 chat template，正常 tokenize
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length - max_new_tokens,  # 为生成留出空间
            ).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

        # 生成
        try:
            output_ids = iterative_generate(
                model=model,
                input_ids=input_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                add_eos_length=add_eos_length,
                refine_iter=refine_iter,
                max_length=max_length,
                temperature=0.0,
                attention_mask=attention_mask,
            )

            # 解码
            for i, (prompt, output) in enumerate(zip(batch_prompts, output_ids)):
                prompt_len = input_ids[i].ne(tokenizer.pad_token_id).sum().item()
                response_ids = output[prompt_len:]

                # 解码完整序列（保留特殊 token 以便检查）
                full_text = tokenizer.decode(output, skip_special_tokens=False)

                # 解码 response (去除特殊 token)
                response = tokenizer.decode(response_ids, skip_special_tokens=True)

                results.append({
                    "prompt": prompt,
                    "response": response,
                    "full_text": full_text,
                    "prompt_length": prompt_len,
                    "response_length": len(response_ids),
                    "total_length": len(output),
                })

        except Exception as e:
            logger.error(f"Error generating for batch {batch_idx}: {e}")
            # 添加失败的结果
            for prompt in batch_prompts:
                results.append({
                    "prompt": prompt,
                    "response": f"[ERROR: {str(e)}]",
                    "prompt_length": 0,
                    "response_length": 0,
                })

    return results


def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    保存结果到文件

    Args:
        results: 结果列表
        output_file: 输出文件路径 (支持 .jsonl 或 .json)
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".jsonl":
        # JSONL 格式 (每行一个 JSON)
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        # JSON 格式
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_file}")


def print_results(results: List[Dict[str, Any]], max_display: int = 5):
    """
    打印结果到终端

    Args:
        results: 结果列表
        max_display: 最多显示的结果数量
    """
    print("\n" + "=" * 80)
    print(f"Generated {len(results)} responses")
    print("=" * 80)

    for i, result in enumerate(results[:max_display]):
        print(f"\n--- Example {i+1} ---")
        print(f"原始 Prompt:")
        print(result['prompt'][:200] + ("..." if len(result['prompt']) > 200 else ""))

        # 显示完整序列（包含特殊 token）
        if 'full_text' in result:
            print(f"\n{'='*80}")
            print(f"完整序列 (包含所有 token):")
            print(f"{'='*80}")
            print(result['full_text'])
            print(f"{'='*80}")

        # 显示统计信息
        print(f"\n统计信息:")
        print(f"  - Prompt 长度: {result['prompt_length']} tokens")
        print(f"  - 生成长度: {result['response_length']} tokens")
        if 'total_length' in result:
            print(f"  - 总长度: {result['total_length']} tokens")

        # 显示仅回复部分
        print(f"\n{'='*80}")
        print(f"仅回复部分 (去除特殊 token):")
        print(f"{'='*80}")
        print(result['response'])
        print(f"{'='*80}")

    if len(results) > max_display:
        print(f"\n... and {len(results) - max_display} more results (see output file)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="dLLM Reasoning Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (local path or HuggingFace model ID)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available)"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading model"
    )

    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt for generation"
    )
    input_group.add_argument(
        "--prompts_file",
        type=str,
        help="File containing prompts (one per line)"
    )

    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Apply tokenizer's chat template to prompts"
    )

    # 生成参数
    parser.add_argument(
        "--add_eos_length",
        type=int,
        default=7,
        help="Number of EOS tokens to add per block (generates add_eos_length+1 tokens per block)"
    )
    parser.add_argument(
        "--refine_iter",
        type=int,
        default=2,
        help="Number of refine iterations per block"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length (prompt + generated)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference"
    )

    # 输出参数
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.jsonl",
        help="Output file to save results (supports .json or .jsonl)"
    )
    parser.add_argument(
        "--max_display",
        type=int,
        default=5,
        help="Maximum number of results to display in terminal"
    )

    args = parser.parse_args()

    # ==================== 加载模型 ====================
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    # ==================== 准备 prompts ====================
    if args.prompt:
        prompts = [args.prompt]
    else:
        # 从文件读取
        prompts_file = Path(args.prompts_file)
        if not prompts_file.exists():
            logger.error(f"Prompts file not found: {args.prompts_file}")
            sys.exit(1)

        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    # ==================== 生成 ====================
    logger.info(
        f"Starting generation: "
        f"num_prompts={len(prompts)}, batch_size={args.batch_size}, "
        f"add_eos_length={args.add_eos_length}, refine_iter={args.refine_iter}"
    )

    results = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        add_eos_length=args.add_eos_length,
        refine_iter=args.refine_iter,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        device=args.device,
        use_chat_template=args.use_chat_template,
    )

    # ==================== 保存和显示结果 ====================
    # 保存到文件
    save_results(results, args.output_file)

    # 打印到终端
    print_results(results, max_display=args.max_display)

    logger.info("Done!")


if __name__ == "__main__":
    main()
