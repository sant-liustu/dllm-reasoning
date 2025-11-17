#!/usr/bin/env python3
"""
迭代精炼训练主脚本

使用方法：
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        -m dllm_reasoning.train_iterative_refine \
        data.train_files=/path/to/train.parquet \
        data.val_files=/path/to/val.parquet \
        model.partial_pretrain=meta-llama/Llama-3-8B \
        trainer.default_local_dir=./checkpoints/my_exp

或者使用 bash 脚本：
    bash dllm_reasoning/scripts/run_train.sh 4 ./checkpoints/my_exp
"""

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import hydra
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh

# VERL 工具
from verl.utils.distributed import initialize_global_process_group
from verl.utils import hf_tokenizer

# 本地模块
from dllm_reasoning.trainer.sft_dataset import SFTDataset
from dllm_reasoning.trainer.iterative_refine_trainer import IterativeRefineTrainer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="iterative_refine"
)
def main(config: DictConfig):
    """
    主函数

    Args:
        config: Hydra 配置对象
    """

    # ============================================================
    # 1. 初始化分布式环境（使用 VERL 工具）
    # ============================================================
    logger.info("初始化分布式环境...")
    local_rank, rank, world_size = initialize_global_process_group()

    logger.info(f"分布式环境初始化完成: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    logger.info(f"CUDA 可用: {torch.cuda.is_available()}, GPU 数量: {torch.cuda.device_count()}")

    # ============================================================
    # 2. 创建 DeviceMesh
    # ============================================================
    logger.info("创建 DeviceMesh...")
    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",)
    )
    logger.info(f"DeviceMesh 创建完成: {device_mesh}")

    # ============================================================
    # 3. 打印配置（仅 rank 0）
    # ============================================================
    if rank == 0:
        logger.info("=" * 80)
        logger.info("训练配置:")
        logger.info("=" * 80)
        logger.info(OmegaConf.to_yaml(config))
        logger.info("=" * 80)

    # ============================================================
    # 4. 加载 tokenizer
    # ============================================================
    logger.info(f"加载 tokenizer: {config.model.partial_pretrain}")
    tokenizer = hf_tokenizer(config.model.partial_pretrain)
    logger.info(f"Tokenizer 加载完成: vocab_size={len(tokenizer)}, "
               f"eos_token={tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # ============================================================
    # 5. 创建数据集
    # ============================================================
    logger.info("创建训练数据集...")
    train_dataset = SFTDataset(
        parquet_files=config.data.train_files,
        tokenizer=tokenizer,
        prompt_key=config.data.prompt_key,
        response_key=config.data.response_key,
        max_length=config.data.max_length,
        truncation=config.data.truncation,
        pad_token_id=config.data.get("pad_token_id"),
    )
    logger.info(f"训练数据集创建完成: size={len(train_dataset)}")

    # 验证集（可选）
    val_dataset = None
    if config.data.get("val_files"):
        logger.info("创建验证数据集...")
        val_dataset = SFTDataset(
            parquet_files=config.data.val_files,
            tokenizer=tokenizer,
            prompt_key=config.data.prompt_key,
            response_key=config.data.response_key,
            max_length=config.data.max_length,
            truncation=config.data.truncation,
            pad_token_id=config.data.get("pad_token_id"),
        )
        logger.info(f"验证数据集创建完成: size={len(val_dataset)}")

    # ============================================================
    # 6. 创建训练器
    # ============================================================
    logger.info("创建迭代精炼训练器...")
    trainer = IterativeRefineTrainer(
        config=config,
        device_mesh=device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    logger.info("训练器创建完成")

    # ============================================================
    # 7. 开始训练
    # ============================================================
    logger.info("开始训练...")
    trainer.fit()

    logger.info("训练完成！")


if __name__ == "__main__":
    main()
