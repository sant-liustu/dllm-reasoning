#!/usr/bin/env python3
"""
直接使用 FSDP checkpoint 进行推理（不转换为 HuggingFace 格式）

这个脚本：
1. 使用与训练时相同的 FSDP 配置初始化模型
2. 直接加载 FSDP checkpoint
3. 进行推理（保持在 FSDP 环境中）

用法:
    # 单 GPU 推理（最简单）
    python -m dllm_reasoning.inference.fsdp_demo \\
        --checkpoint_dir checkpoints/openr1_test_fixed/global_step_21000 \\
        --prompt "What is 2+2?"

    # 多 GPU 推理（使用训练时的 GPU 数量）
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        -m dllm_reasoning.inference.fsdp_demo \\
        --checkpoint_dir checkpoints/openr1_test_fixed/global_step_21000 \\
        --prompt "What is 2+2?"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.inference import iterative_generate
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with FSDP checkpoint (no conversion)")

    # Checkpoint 参数
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to FSDP checkpoint directory",
    )

    # 输入参数
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for inference",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Apply tokenizer's chat template to prompts",
    )

    # 生成参数
    parser.add_argument(
        "--add_eos_length",
        type=int,
        default=127,
        help="Number of EOS tokens to add per block",
    )
    parser.add_argument(
        "--refine_iter",
        type=int,
        default=2,
        help="Number of refine iterations per block",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )

    # 输出参数
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save results (optional)",
    )

    # 调试参数
    parser.add_argument(
        "--verbose_trace",
        action="store_true",
        help="Enable verbose trace to show evolution of token sequences",
    )

    return parser.parse_args()


def setup_distributed():
    """初始化分布式环境（如果需要）"""
    if "RANK" in os.environ:
        # 多 GPU 模式
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        logger.info(f"Initialized distributed: rank={rank}/{world_size}, device={device}")

        return rank, world_size, device, True
    else:
        # 单 GPU 模式
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
        else:
            device = torch.device("cpu")

        logger.info(f"Running in single-device mode: device={device}")

        return 0, 1, device, False


def load_fsdp_model(checkpoint_dir: str, rank: int, world_size: int, device, is_distributed: bool):
    """加载 FSDP checkpoint"""

    # 加载 config 和 tokenizer
    hf_config_dir = os.path.join(checkpoint_dir, "huggingface")
    if not os.path.isdir(hf_config_dir):
        raise FileNotFoundError(f"Config directory not found: {hf_config_dir}")

    logger.info(f"Loading config and tokenizer from {hf_config_dir}")
    config = AutoConfig.from_pretrained(hf_config_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_config_dir)

    if not is_distributed:
        # 单 GPU 模式：直接加载模型，不使用 FSDP
        logger.info("Loading model in single-GPU mode (no FSDP)")

        # 我们需要手动加载 FSDP checkpoint 的权重
        # 这比较复杂，所以单 GPU 模式下我们建议先转换
        logger.error("单 GPU 模式暂不支持直接加载 FSDP checkpoint")
        logger.error("请使用多 GPU 模式（与训练时 GPU 数量一致）")
        logger.error(f"例如: torchrun --nproc_per_node={world_size} -m dllm_reasoning.inference.fsdp_demo ...")
        sys.exit(1)

    # 多 GPU 模式：使用 FSDP
    logger.info("Initializing model with FSDP...")

    # 创建 device mesh（与训练时相同 - 包括 mesh_dim_names！）
    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",)  # ← 关键：必须与训练时一致！
    )

    # 初始化模型
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    # FSDP 包装（与训练时相同的配置）
    wrap_policy_config = {"min_num_params": 0}
    auto_wrap_policy = get_fsdp_wrap_policy(model, config=wrap_policy_config)

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )

    fsdp_model = FSDP(
        module=model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_mesh=device_mesh,
        sync_module_states=True,
        device_id=torch.cuda.current_device(),
        use_orig_params=False,  # 必须与训练时一致
    )

    logger.info("FSDP model initialized")

    # 使用 checkpoint manager 加载权重
    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    checkpoint_config = DictConfig({
        "load_contents": ["model"],
        "save_contents": ["model"],
    })

    checkpoint_manager = FSDPCheckpointManager(
        model=fsdp_model,
        optimizer=None,
        lr_scheduler=None,
        processing_class=tokenizer,
        checkpoint_config=checkpoint_config,
    )

    checkpoint_manager.load_checkpoint(local_path=checkpoint_dir)

    logger.info("Checkpoint loaded successfully")

    # 设置为评估模式
    fsdp_model.eval()

    return fsdp_model, tokenizer, config


def generate_single(model, tokenizer, prompt: str, args, device, rank: int):
    """单个 prompt 生成"""

    # 应用 chat template（如果需要）
    if args.use_chat_template:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            # 直接让 apply_chat_template 返回 token IDs，避免双 BOS token
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,  # 直接返回 token IDs
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
        else:
            logger.warning("Tokenizer does not have chat template, using raw prompt")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
    else:
        # 不使用 chat template
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

    if rank == 0:
        logger.info(f"Prompt ({input_ids.size(1)} tokens): {prompt[:100]}...")

    # 生成
    with torch.no_grad():
        output_ids = iterative_generate(
            model=model,
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            add_eos_length=args.add_eos_length,
            refine_iter=args.refine_iter,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            verbose_trace=args.verbose_trace,  # ✅ 传递演化追踪参数
            tokenizer=tokenizer if args.verbose_trace else None,  # ✅ 传递 tokenizer
        )

    # 解码 - 分别解码完整序列和仅回复部分
    full_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=False,  # 保留特殊 token 以便检查
    )

    response = tokenizer.decode(
        output_ids[0, input_ids.size(1):],
        skip_special_tokens=True,
    )

    return response, full_text, input_ids.size(1), output_ids.size(1)


def main():
    args = parse_args()

    # 检查输入参数
    if args.prompt is None and args.prompts_file is None:
        logger.error("请提供 --prompt 或 --prompts_file")
        sys.exit(1)

    if args.prompt and args.prompts_file:
        logger.error("--prompt 和 --prompts_file 不能同时使用")
        sys.exit(1)

    # 初始化分布式环境
    rank, world_size, device, is_distributed = setup_distributed()

    # 加载模型
    model, tokenizer, config = load_fsdp_model(
        args.checkpoint_dir,
        rank,
        world_size,
        device,
        is_distributed,
    )

    # 准备 prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    if rank == 0:
        logger.info(f"Processing {len(prompts)} prompt(s)...")

    # 生成
    results = []
    for i, prompt in enumerate(prompts):
        if rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"Prompt {i+1}/{len(prompts)}")
            logger.info(f"{'='*80}")

        try:
            response, full_text, prompt_len, total_len = generate_single(model, tokenizer, prompt, args, device, rank)

            if rank == 0:
                print(f"\n{'='*80}")
                print(f"原始 Prompt: {prompt}")
                print(f"\n{'='*80}")
                print(f"完整序列 (包含所有 token):")
                print(f"{'='*80}")
                print(full_text)
                print(f"{'='*80}")
                print(f"\n统计信息:")
                print(f"  - Prompt 长度: {prompt_len} tokens")
                print(f"  - 生成长度: {total_len - prompt_len} tokens")
                print(f"  - 总长度: {total_len} tokens")
                print(f"\n{'='*80}")
                print(f"仅回复部分 (去除特殊 token):")
                print(f"{'='*80}")
                print(response)
                print(f"{'='*80}\n")

            results.append({
                "prompt": prompt,
                "response": response,
                "full_text": full_text,
                "prompt_len": prompt_len,
                "total_len": total_len,
            })

        except Exception as e:
            if rank == 0:
                logger.error(f"Error generating for prompt {i+1}: {e}")
                import traceback
                traceback.print_exc()

    # 保存结果（如果指定）
    if args.output_file and rank == 0:
        import json

        if args.output_file.endswith(".jsonl"):
            with open(args.output_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {args.output_file}")

    # 清理
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        logger.info("✅ Inference completed successfully!")


if __name__ == "__main__":
    main()
