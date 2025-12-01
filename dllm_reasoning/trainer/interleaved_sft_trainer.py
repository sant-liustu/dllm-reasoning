"""
交错训练器 (Interleaved SFT Trainer)

核心思想：
1. 使用 InterleavedSFTDataset（返回 block_info）
2. 使用 FlexAttention BlockMask 实现复杂的 attention pattern
3. 标准的 SFT 训练（使用 cross-entropy loss）
4. 充分利用所有位置计算 loss

这是一个独立的训练器实现，不依赖于 IterativeRefineTrainer。
"""

import os
import logging
import re
from typing import Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullStateDictConfig
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizer
from omegaconf import DictConfig

# VERL 工具函数
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils import hf_tokenizer
from verl.utils.logger import log_with_rank
import verl.utils.hdfs_io as hdfs_io

# VERL checkpoint 相关
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

# 交错训练专用工具
# Note: create_block_mask_from_batch 已移至模型内部,不再需要导入
from dllm_reasoning.trainer.interleaved_sft_dataset import collate_interleaved_batch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("INTERLEAVED_SFT_LOGGING_LEVEL", "INFO"))


def extract_step(path):
    """从检查点路径中提取步数"""
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class InterleavedSFTTrainer:
    """
    交错训练器 (Interleaved SFT Trainer)

    完全独立的实现，不继承 IterativeRefineTrainer。

    特点：
    1. 使用 FlexAttention BlockMask 实现交错的 attention pattern
    2. 标准的 next-token prediction loss（cross-entropy）
    3. 支持 FSDP 分布式训练
    4. 完整的 checkpoint 管理
    """

    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,  # InterleavedSFTDataset
        val_dataset: Optional[Dataset] = None,
    ):
        """
        初始化交错训练器

        Args:
            config: 训练配置（DictConfig 或 dict）
            device_mesh: FSDP 设备网格
            tokenizer: Tokenizer
            train_dataset: InterleavedSFTDataset 实例
            val_dataset: 验证集（可选）
        """
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.rank = device_mesh.get_rank()
        self.world_size = device_mesh.size()

        # ========== 初始化调试日志文件 ==========
        self._setup_debug_logger()

        # 构建数据加载器
        self._build_dataloader()

        # 构建模型和优化器
        self._build_model_optimizer()

        # 训练状态
        self.current_epoch = 0

        # 交错训练特定配置
        self.block_size = getattr(self.config.data, 'block_size', 4)

        # 初始化 checkpoint manager 和恢复训练状态
        self.resume_global_step = 0
        self._init_checkpoint_manager()
        self.load_checkpoint()

        logger.info("=" * 80)
        logger.info("InterleavedSFTTrainer 初始化完成")
        logger.info(f"  Block size: {self.block_size}")
        logger.info(f"  使用 FlexAttention BlockMask")
        logger.info("=" * 80)

    def _setup_debug_logger(self):
        """设置调试日志文件"""
        if self.rank == 0:
            # 只在 rank 0 上创建调试日志
            project_root = Path(__file__).resolve().parents[2]
            log_dir = project_root / "log"
            log_dir.mkdir(parents=True, exist_ok=True)
            debug_log_path = str(log_dir / "debug.log")

            # 创建文件处理器
            file_handler = logging.FileHandler(debug_log_path, mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            # 添加到 logger
            debug_logger = logging.getLogger('DEBUG')
            debug_logger.setLevel(logging.INFO)
            debug_logger.addHandler(file_handler)
            debug_logger.propagate = False  # 不传播到父logger

            self.debug_logger = debug_logger
            self.debug_logger.info("=" * 80)
            self.debug_logger.info("调试日志初始化完成")
            self.debug_logger.info(f"日志文件: {debug_log_path}")
            self.debug_logger.info("=" * 80)
        else:
            self.debug_logger = None

    def _build_dataloader(self):
        """构建数据加载器"""
        # 训练集
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        # 使用自定义 collate_fn 处理变长序列
        from functools import partial
        collate_fn = partial(
            collate_interleaved_batch,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            ignore_index=-100,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.train_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        # 验证集（如果有）
        if self.val_dataset is not None:
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )

            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.data.micro_batch_size_per_gpu,
                sampler=self.val_sampler,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True,
            )
        else:
            self.val_dataloader = None

        logger.info(f"数据加载器构建完成: train_size={len(self.train_dataset)}, "
                   f"val_size={len(self.val_dataset) if self.val_dataset else 0}")

    def _build_model_optimizer(self):
        """构建模型、FSDP 包装、优化器和学习率调度器"""

        # 1. 下载模型（使用 VERL 的文件系统工具）
        logger.info(f"加载模型: {self.config.model.partial_pretrain}")
        local_model_path = copy_local_path_from_hdfs(
            src=self.config.model.partial_pretrain,
            verbose=True
        )

        # 2. 加载配置
        config = AutoConfig.from_pretrained(
            local_model_path,
            trust_remote_code=self.config.model.get("trust_remote_code", False),
        )

        # 3. Meta Tensor 初始化（使用 VERL 工具，节省显存）
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings
        )

        with init_context():
            # 加载标准的因果语言模型
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch.float32,  # FSDP 会处理混合精度
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )

        logger.info(f"模型加载完成: {self.model.__class__.__name__}")

        # 4. 梯度检查点（可选）
        if self.config.model.get("enable_gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("已启用梯度检查点")

        # 5. FSDP 包装（使用 VERL 工具）
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
        )

        cpu_offload = None
        if self.config.model.fsdp_config.get("cpu_offload", False):
            cpu_offload = CPUOffload(offload_params=True)

        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )

        logger.info("FSDP 包装完成")

        # 6. 优化器和学习率调度器（使用 VERL 工具）
        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        # 梯度累积配置
        self.gradient_accumulation_steps = self.config.optim.get("gradient_accumulation_steps", 1)

        # 计算总步数
        steps_per_epoch = len(self.train_dataloader)
        self.total_steps = steps_per_epoch * self.config.trainer.total_epochs  # 总数据步数
        # 总的优化器更新次数 = 数据步数 / 梯度累积步数
        self.total_optimizer_steps = self.total_steps // self.gradient_accumulation_steps
        num_warmup_steps = int(self.total_optimizer_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_optimizer_steps,
        )

        logger.info(f"优化器和调度器创建完成: total_data_steps={self.total_steps}, "
                   f"gradient_accumulation_steps={self.gradient_accumulation_steps}, "
                   f"total_optimizer_steps={self.total_optimizer_steps}, "
                   f"warmup_steps={num_warmup_steps}, "
                   f"effective_batch_size={self.config.data.micro_batch_size_per_gpu * self.world_size * self.gradient_accumulation_steps}")

    def _init_checkpoint_manager(self):
        """初始化 checkpoint manager (参考 VERL 的实现)"""
        # 获取 checkpoint 配置，设置默认值
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # 默认保存和加载所有内容: model, optimizer, extra (lr_scheduler + rng)
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # 创建 checkpoint 配置字典
        checkpoint_config_dict = DictConfig({
            "load_contents": load_contents,
            "save_contents": save_contents,
        })

        # 初始化 checkpoint manager
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

        log_with_rank(
            f"Checkpoint manager 已初始化: save_contents={save_contents}, load_contents={load_contents}",
            logger=logger,
            rank=self.rank,
            log_only_rank_0=True,
        )

    def load_checkpoint(self):
        """加载 checkpoint (参考 VERL 的实现)"""
        # 根据配置确定恢复路径
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            log_with_rank(
                "没有找到 checkpoint，从头开始训练",
                logger=logger,
                rank=self.rank,
                log_only_rank_0=True,
            )
            return 0

        # 从 checkpoint 路径中提取步数
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"警告: 无法从 {checkpoint_path} 中提取步数，从 step 0 开始",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0

        self.resume_global_step = resume_step

        # 使用 checkpoint manager 加载模型、优化器、学习率调度器
        self.checkpoint_manager.load_checkpoint(checkpoint_path)

        log_with_rank(
            f"成功从 {checkpoint_path} 加载 checkpoint (step {resume_step})",
            logger=logger,
            rank=self.rank,
            log_only_rank_0=True,
        )

        return resume_step

    def _determine_resume_path(self):
        """根据配置确定恢复路径 (参考 VERL 的实现)"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path 必须是 null 或一个存在的路径 (当 resume_mode='auto' 时)"
                )
                assert "global_step_" in resume_from_path, "resume_from_path 必须包含 global_step_"
                return resume_from_path
            # 尝试在默认目录中查找最新的 checkpoint
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path 必须是一个存在的路径 (当 resume_mode='resume_path' 时)"
            )
            assert "global_step_" in resume_from_path, "resume_from_path 必须包含 global_step_"
            return resume_from_path
        else:
            raise ValueError(f"无效的 resume_mode: {resume_mode}。必须是 'auto', 'disable', 或 'resume_path'")

    def _find_latest_checkpoint(self):
        """在默认目录中查找最新的 checkpoint (参考 VERL 的实现)"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.rank == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"找到最新 checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def training_step(self, batch: dict, global_step: int):
        """
        单个训练步骤（交错训练版本）

        使用 FlexAttention BlockMask 实现复杂的 attention pattern，
        标准的 cross-entropy loss。

        Args:
            batch: dict 包含:
                - input_ids: [B, seq_len]
                - position_ids: [B, seq_len]
                - labels: [B, seq_len]
                - block_info: List[List[Tuple]]
                - prompt_len: List[int]
                - seq_lens: List[int]
            global_step: int - 全局步数

        Returns:
            metrics: dict - 训练指标
        """
        self.fsdp_model.train()

        # 判断是否需要在这一步更新优化器
        is_accumulation_step = (global_step + 1) % self.gradient_accumulation_steps != 0
        should_update_optimizer = not is_accumulation_step

        # ========== 调试日志 ==========
        # 修改为每步都输出详细日志(用于debug)
        should_log_detail = self.debug_logger is not None

        if should_log_detail:
            input_ids = batch["input_ids"]

            self.debug_logger.info("\n" + "=" * 80)
            self.debug_logger.info(f"[Step {global_step}] 交错训练批次详细信息")
            self.debug_logger.info("=" * 80)
            self.debug_logger.info(f"Batch size: {input_ids.shape[0]}")
            self.debug_logger.info(f"Sequence length: {input_ids.shape[1]}")
            self.debug_logger.info(f"Block size: {self.block_size}")
            self.debug_logger.info(f"Gradient accumulation: {(global_step % self.gradient_accumulation_steps) + 1}/{self.gradient_accumulation_steps}")
            self.debug_logger.info(f"Will update optimizer: {should_update_optimizer}")

            # 显示第一个样本的 block_info
            if 'block_info' in batch and len(batch['block_info']) > 0:
                self.debug_logger.info(f"\n样本0的 block_info: {batch['block_info'][0]}")
                self.debug_logger.info(f"样本0的 prompt_len: {batch['prompt_len'][0]}")
                self.debug_logger.info(f"样本0的实际长度: {batch['seq_lens'][0]}")

        # ========== 前向传播 ==========
        # 传递轻量参数给模型,让模型内部创建 BlockMask (避免 CPU tensor 优化问题)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.fsdp_model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                # 传递 Interleaved Training 轻量参数
                block_info=batch["block_info"],
                prompt_len=batch["prompt_len"],
                seq_lens=batch["seq_lens"],
            )

            if should_log_detail:
                self.debug_logger.info(f"✓ 前向传播完成,模型内部已创建 BlockMask")

        logits = outputs.logits  # [B, seq_len, vocab_size]
        labels = batch["labels"].to(logits.device)  # [B, seq_len] - 确保在同一设备上

        # ========== 计算 loss ==========
        # 使用标准的 cross-entropy loss
        # labels 中已经用 -100 标记了不计算 loss 的位置
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )

        # 计算准确率（仅在有效位置）
        with torch.no_grad():
            valid_mask = (labels != -100)
            if valid_mask.sum() > 0:
                predictions = logits.argmax(dim=-1)
                correct = (predictions == labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()

                # ========== Debug: 分析Mask vs Real准确率 ==========
                if should_log_detail:
                    # 从batch中获取block_info和prompt_len
                    block_info_list = batch.get('block_info', [[]])
                    prompt_len_list = batch.get('prompt_len', [0])

                    # 只分析第一个样本
                    if len(block_info_list) > 0 and len(prompt_len_list) > 0:
                        block_info = block_info_list[0]
                        prompt_len = prompt_len_list[0]

                        # 获取第一个样本的数据
                        labels_0 = labels[0]
                        predictions_0 = predictions[0]

                        # 分类统计不同位置
                        mask_positions = []
                        real_positions = []

                        current_pos = prompt_len
                        for seg_type, seg_len in block_info:
                            if seg_type == 'mask':
                                mask_positions.extend(range(current_pos, current_pos + seg_len))
                            elif seg_type == 'real':
                                real_positions.extend(range(current_pos, current_pos + seg_len))
                            current_pos += seg_len

                        # 计算Mask准确率
                        if len(mask_positions) > 0:
                            mask_pos_tensor = torch.tensor(mask_positions, device=labels.device)
                            mask_labels = labels_0[mask_pos_tensor]
                            mask_valid = mask_labels != -100
                            if mask_valid.sum() > 0:
                                mask_correct = ((predictions_0[mask_pos_tensor] == mask_labels) & mask_valid).sum()
                                mask_acc = mask_correct.float() / mask_valid.sum().float()
                            else:
                                mask_acc = torch.tensor(0.0, device=labels.device)
                        else:
                            mask_acc = torch.tensor(0.0, device=labels.device)

                        # 计算Real准确率
                        if len(real_positions) > 0:
                            real_pos_tensor = torch.tensor(real_positions, device=labels.device)
                            real_labels = labels_0[real_pos_tensor]
                            real_valid = real_labels != -100
                            if real_valid.sum() > 0:
                                real_correct = ((predictions_0[real_pos_tensor] == real_labels) & real_valid).sum()
                                real_acc = real_correct.float() / real_valid.sum().float()
                            else:
                                real_acc = torch.tensor(0.0, device=labels.device)
                        else:
                            real_acc = torch.tensor(0.0, device=labels.device)

                        # 计算Prompt[-1]准确率
                        prompt_last_acc = torch.tensor(0.0, device=labels.device)
                        if prompt_len > 0 and labels_0[prompt_len - 1].item() != -100:
                            if predictions_0[prompt_len - 1].item() == labels_0[prompt_len - 1].item():
                                prompt_last_acc = torch.tensor(1.0, device=labels.device)

                        self.debug_logger.info(f"\n准确率详细分析 (Sample 0):")
                        self.debug_logger.info(f"  总体准确率: {accuracy.item():.4f}")
                        self.debug_logger.info(f"  Prompt[-1] 准确率: {prompt_last_acc.item():.4f}")
                        self.debug_logger.info(f"  Mask 准确率: {mask_acc.item():.4f} ({len(mask_positions)} positions)")
                        self.debug_logger.info(f"  Real 准确率: {real_acc.item():.4f} ({len(real_positions)} positions)")

                        # 显示前200个tokens的预测详情
                        self.debug_logger.info(f"\n前200个有效位置的预测详情:")
                        valid_positions = []
                        for i in range(len(labels_0)):
                            if labels_0[i].item() != -100:
                                valid_positions.append(i)
                                if len(valid_positions) >= 200:
                                    break

                        for i, pos in enumerate(valid_positions[:200]):
                            pred_id = predictions_0[pos].item()
                            true_id = labels_0[pos].item()
                            is_correct = "✅" if pred_id == true_id else "❌"

                            # 判断位置类型
                            if pos == prompt_len - 1:
                                pos_type = "Prompt[-1]"
                            elif pos in mask_positions:
                                pos_type = "Mask"
                            elif pos in real_positions:
                                pos_type = "Real"
                            else:
                                pos_type = "Unknown"

                            if i < 50 or not is_correct == "✅":  # 前50个或错误的都显示
                                self.debug_logger.info(
                                    f"    [{i:3d}] pos={pos:4d} {pos_type:12s}: {is_correct} "
                                    f"pred={pred_id:6d}, true={true_id:6d}"
                                )
            else:
                accuracy = torch.tensor(0.0, device=labels.device)

        # ========== 反向传播和优化器更新 ==========
        scaled_loss = loss / self.gradient_accumulation_steps

        if is_accumulation_step:
            with self.fsdp_model.no_sync():
                scaled_loss.backward()
        else:
            scaled_loss.backward()

        # 构造 metrics
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
        }

        # 只在累积完成时更新优化器
        if should_update_optimizer:
            # 梯度裁剪
            grad_norm = self.fsdp_model.clip_grad_norm_(
                max_norm=self.config.optim.clip_grad
            )

            # 优化器更新
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            # 添加额外的指标
            metrics['grad_norm'] = grad_norm.item()
            metrics['lr'] = self.lr_scheduler.get_last_lr()[0]
            metrics['is_optimizer_step'] = True

            if should_log_detail:
                self.debug_logger.info(f"\n训练指标:")
                self.debug_logger.info(f"  Loss: {loss.item():.4f}")
                self.debug_logger.info(f"  Accuracy: {accuracy.item():.4f}")
                self.debug_logger.info(f"  Grad norm: {grad_norm.item():.4f}")
                self.debug_logger.info(f"  Learning rate: {metrics['lr']:.6e}")
        else:
            metrics['grad_norm'] = 0.0
            metrics['lr'] = self.lr_scheduler.get_last_lr()[0]
            metrics['is_optimizer_step'] = False

        return metrics

    def save_checkpoint(self, step: int):
        """
        保存检查点 (使用 VERL 的 checkpoint_manager)

        Args:
            step: int - 全局步数
        """
        path = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{step}"
        )

        # 获取 max_ckpt_to_keep 配置
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # 使用 checkpoint_manager 保存
        # 它会自动处理 FSDP sharded state、optimizer、lr_scheduler、rng 等
        self.checkpoint_manager.save_checkpoint(
            local_path=path,
            hdfs_path=None,  # HDFS 由 checkpoint_manager 内部处理
            global_step=step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

        log_with_rank(
            f"检查点已保存到: {path}",
            logger=logger,
            rank=self.rank,
            log_only_rank_0=True,
        )

        # 可选：复制到 HDFS（如果配置了）
        if self.rank == 0 and self.config.trainer.get("default_hdfs_dir"):
            hdfs_path = os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{step}"
            )
            try:
                hdfs_io.makedirs(os.path.dirname(hdfs_path), exist_ok=True)
                hdfs_io.copy(src=path, dst=hdfs_path, dirs_exist_ok=True)
                logger.info(f"检查点已备份到 HDFS: {hdfs_path}")
            except Exception as e:
                logger.warning(f"HDFS 备份失败: {e}")

        torch.distributed.barrier()

    def validate(self):
        """
        验证步骤（简化版）

        注意：这里只计算 loss，不做生成评估
        如果需要生成评估，需要另外实现
        """
        if self.val_dataloader is None:
            return {}

        self.fsdp_model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                # 移动到设备
                batch = {k: v.to(self.device_mesh.get_local_device()) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 构造 BlockMask
                device = batch["input_ids"].device
                block_mask = create_block_mask_from_batch(batch, device)

                # 前向传播
                outputs = self.fsdp_model(
                    input_ids=batch["input_ids"],
                    position_ids=batch["position_ids"],
                    attention_mask=block_mask,
                )

                logits = outputs.logits
                labels = batch["labels"]

                # 计算 loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='mean',
                )

                # 计算准确率
                valid_mask = (labels != -100)
                if valid_mask.sum() > 0:
                    predictions = logits.argmax(dim=-1)
                    correct = (predictions == labels) & valid_mask
                    accuracy = correct.sum().float() / valid_mask.sum().float()
                else:
                    accuracy = torch.tensor(0.0, device=labels.device)

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

        metrics = {
            'val_loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'val_accuracy': total_accuracy / num_batches if num_batches > 0 else 0.0,
        }

        return metrics

    def fit(self):
        """
        主训练循环
        """
        # 初始化追踪（使用 VERL 的 Tracking）
        if self.rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.get("logger", ["console"]),
            )

        # 从恢复的步数开始（如果有 checkpoint）
        global_step = self.resume_global_step

        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info(f"Total epochs: {self.config.trainer.total_epochs}")
        logger.info(f"Steps per epoch: {len(self.train_dataloader)}")
        logger.info(f"Total steps: {self.total_steps}")
        if self.resume_global_step > 0:
            logger.info(f"从 checkpoint 恢复: global_step={self.resume_global_step}")
        logger.info("=" * 60)

        # ========== 记录完整的超参数配置 ==========
        if self.debug_logger is not None:
            self.debug_logger.info("\n" + "=" * 80)
            self.debug_logger.info("超参数配置")
            self.debug_logger.info("=" * 80)
            self.debug_logger.info("\n训练配置:")
            self.debug_logger.info(f"  Total epochs: {self.config.trainer.total_epochs}")
            self.debug_logger.info(f"  Steps per epoch: {len(self.train_dataloader)}")
            self.debug_logger.info(f"  Total steps: {self.total_steps}")
            self.debug_logger.info(f"  Batch size per GPU: {self.config.data.micro_batch_size_per_gpu}")
            self.debug_logger.info(f"  World size: {self.world_size}")
            self.debug_logger.info(f"  Global batch size: {self.config.data.micro_batch_size_per_gpu * self.world_size}")

            self.debug_logger.info("\n优化器配置:")
            self.debug_logger.info(f"  Learning rate: {self.config.optim.lr}")
            self.debug_logger.info(f"  Betas: {self.config.optim.betas}")
            self.debug_logger.info(f"  Weight decay: {self.config.optim.weight_decay}")
            self.debug_logger.info(f"  Gradient clipping: {self.config.optim.clip_grad}")
            self.debug_logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
            self.debug_logger.info(f"  Effective batch size: {self.config.data.micro_batch_size_per_gpu * self.world_size * self.gradient_accumulation_steps}")
            self.debug_logger.info(f"  Warmup steps ratio: {self.config.optim.warmup_steps_ratio}")
            self.debug_logger.info(f"  Total optimizer steps: {self.total_optimizer_steps}")
            warmup_steps = int(self.total_optimizer_steps * self.config.optim.warmup_steps_ratio)
            self.debug_logger.info(f"  Warmup steps: {warmup_steps}")

            self.debug_logger.info("\n交错训练配置:")
            self.debug_logger.info(f"  Block size: {self.block_size}")

            self.debug_logger.info("\n数据配置:")
            self.debug_logger.info(f"  Train files: {self.config.data.train_files}")
            self.debug_logger.info(f"  Max length: {self.config.data.max_length}")
            self.debug_logger.info(f"  Truncation: {self.config.data.truncation}")
            self.debug_logger.info(f"  Prompt key: {self.config.data.prompt_key}")
            self.debug_logger.info(f"  Response key: {self.config.data.response_key}")

            self.debug_logger.info("\n模型配置:")
            self.debug_logger.info(f"  Model: {self.config.model.partial_pretrain}")
            self.debug_logger.info(f"  Gradient checkpointing: {self.config.model.get('enable_gradient_checkpointing', False)}")

            self.debug_logger.info("=" * 80)

        # 训练开始前清零梯度
        self.optimizer.zero_grad()

        # 主训练循环
        for epoch in range(self.current_epoch, self.config.trainer.total_epochs):
            self.current_epoch = epoch
            self.train_sampler.set_epoch(epoch)  # 重要：让 DDP 的 shuffle 正确工作

            # Epoch 内的训练
            dataloader_iter = iter(self.train_dataloader)
            pbar = tqdm(
                dataloader_iter,
                desc=f"Epoch {epoch}",
                total=len(self.train_dataloader),
                disable=(self.rank != 0),
            )

            for batch in pbar:
                # 训练步骤
                metrics = self.training_step(batch, global_step)

                # 更新进度条
                if self.rank == 0:
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'lr': f"{metrics['lr']:.2e}",
                    })

                    # 记录到追踪系统
                    tracking.log(
                        data={f"train/{k}": v for k, v in metrics.items()},
                        step=global_step,
                    )

                global_step += 1

                # 保存检查点
                if global_step % self.config.trainer.save_checkpoint_steps == 0:
                    self.save_checkpoint(step=global_step)

                # ========== 调试模式：最大步数限制 ==========
                max_debug_steps = self.config.trainer.get("max_debug_steps", None)
                if max_debug_steps is not None and global_step >= max_debug_steps:
                    if self.rank == 0:
                        logger.info(f"\n达到调试最大步数 {max_debug_steps}，停止训练")
                    return

            # Epoch 结束后保存
            if self.rank == 0:
                logger.info(f"Epoch {epoch} 完成")

        # 训练结束
        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("训练完成！")
            logger.info("=" * 60)

        # 保存最终检查点
        self.save_checkpoint(step=global_step)
