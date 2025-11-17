"""
迭代精炼训练器 (Iterative Refine Trainer)

核心思想：
1. 对 response 区域加噪（用 EOS token）
2. 模型前向传播得到 logits
3. 计算 loss（对原始 response 的 next token prediction）
4. 贪婪解码得到精炼后的序列
5. 重复 2-4 步若干次
6. 聚合所有轮次的 loss 进行一次梯度更新
"""

import os
import logging
import re
from typing import Optional

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullStateDictConfig
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizer
from tensordict import TensorDict

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
from omegaconf import DictConfig

# 本地模块
from dllm_reasoning.utils.noise_utils import q_sample, greedy_decode_response
from dllm_reasoning.losses import compute_loss_on_response, compute_iterative_loss

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ITERATIVE_SFT_LOGGING_LEVEL", "INFO"))


def extract_step(path):
    """从检查点路径中提取步数"""
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class IterativeRefineTrainer:
    """
    迭代精炼训练器

    与 Dream 的主要区别：
    1. 不需要修改模型架构（使用标准的 AR 模型）
    2. 多轮前向传播（s0 → s1 → ...）
    3. 每轮都对原始 token 计算 loss
    4. 只在 response 区域操作
    """

    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
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

        # 获取 EOS token ID
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            raise ValueError("tokenizer 没有定义 eos_token_id!")

        logger.info(f"使用 EOS token ID: {self.eos_token_id} ({self.tokenizer.eos_token})")

        # 迭代配置
        self.num_iterations = config.iterative.get("num_iterations", 2)
        self.noise_min = config.iterative.get("noise_min", 0.1)
        self.noise_max = config.iterative.get("noise_max", 0.9)
        self.loss_weights = config.iterative.get("loss_weights", [1.0] * self.num_iterations)

        logger.info(f"迭代配置: num_iterations={self.num_iterations}, "
                   f"noise_range=[{self.noise_min}, {self.noise_max}], "
                   f"loss_weights={self.loss_weights}")

        # 初始化 checkpoint manager 和恢复训练状态
        self.resume_global_step = 0
        self._init_checkpoint_manager()
        self.load_checkpoint()

    def _setup_debug_logger(self):
        """设置调试日志文件"""
        if self.rank == 0:
            # 只在 rank 0 上创建调试日志
            # 使用 Path 计算项目根目录（更清晰）
            from pathlib import Path
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

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.train_sampler,
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
            # 加载标准的因果语言模型（不修改架构！）
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

    def _compute_iterative_loss(self, batch: dict, global_step: int = -1):
        """
        计算多轮迭代的 loss（核心逻辑）

        流程：
        1. 从原始 batch 获取 t0 和 response_mask
        2. 加噪得到 s0
        3. 前向传播 s0，计算 loss_s0
        4. 贪婪解码得到 s1
        5. 前向传播 s1，计算 loss_s1
        6. ... (可继续更多轮)
        7. 聚合所有 loss

        Args:
            batch: dict 包含 input_ids, attention_mask, position_ids, loss_mask
            global_step: int - 全局步数，用于调试日志

        Returns:
            total_loss: scalar tensor
            metrics: dict - 用于日志的指标
        """
        # 准备数据
        t0 = batch["input_ids"].cuda()  # [batch_size, seq_len] - 原始 token
        attention_mask = batch["attention_mask"].cuda()
        position_ids = batch["position_ids"].cuda()
        response_mask = batch["loss_mask"].cuda()  # [batch_size, seq_len] - 1=response, 0=instruction

        batch_size = t0.shape[0]

        # 采样噪声比例 t
        t = torch.rand((batch_size,), dtype=torch.float, device=t0.device)
        t = self.noise_min + (self.noise_max - self.noise_min) * t

        # ========== 阶段3：加噪和迭代过程调试（每10步详细记录） ==========
        should_log_detail = (global_step % 10 == 0) and self.debug_logger is not None and global_step >= 0

        if should_log_detail:
            self.debug_logger.info("\n" + "=" * 80)
            self.debug_logger.info(f"[Step {global_step}] 迭代精炼过程详解")
            self.debug_logger.info("=" * 80)

            # 记录原始序列 t0
            t0_sample = t0[0].cpu().tolist()
            response_mask_sample = response_mask[0].cpu().tolist()

            # 找到 response 区域的起始和结束位置
            response_indices = [i for i, mask in enumerate(response_mask_sample) if mask == 1]
            if response_indices:
                response_start = response_indices[0]
                response_end = response_indices[-1] + 1
                response_tokens = t0_sample[response_start:response_end]
                response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=False)

                self.debug_logger.info(f"\n[原始序列 t0]")
                self.debug_logger.info(f"  Response 区域: tokens[{response_start}:{response_end}] (共 {len(response_tokens)} tokens)")
                self.debug_logger.info(f"  Response 文本: {response_text[:200]}")
                self.debug_logger.info(f"  噪声比例 t: {t[0].item():.3f}")

        # 用于存储每一轮的 loss
        losses = []

        # 当前序列（初始为原始序列）
        current_input_ids = t0

        # 多轮迭代
        for iter_idx in range(self.num_iterations):
            # ===========================================================
            # Step 1: 加噪（第一轮）或使用上一轮的解码结果（后续轮）
            # ===========================================================
            if iter_idx == 0:
                # 第一轮：加噪
                # 使用 Dream 的 q_sample 函数，但这里 mask_token_id 用 eos_token_id
                s_i, _, _ = q_sample(
                    input_ids=t0,
                    maskable_mask=response_mask.bool(),
                    mask_token_id=self.eos_token_id,
                    min=0.0,  # 我们已经在外面采样了 t，这里不再重复采样
                    max=1.0,
                    eos_token_id=self.eos_token_id,
                    t=t,
                )

                # ========== 记录加噪效果 ==========
                if should_log_detail:
                    s_i_sample = s_i[0].cpu().tolist()
                    response_indices = [i for i, mask in enumerate(response_mask_sample) if mask == 1]
                    if response_indices:
                        response_start = response_indices[0]
                        response_end = response_indices[-1] + 1

                        # 对比原始和加噪后的 response 区域
                        original_response = t0_sample[response_start:response_end]
                        noised_response = s_i_sample[response_start:response_end]

                        # 统计被替换的 token 数量
                        num_replaced = sum(1 for orig, noised in zip(original_response, noised_response)
                                          if orig != noised)
                        replace_ratio = num_replaced / len(original_response) if original_response else 0

                        noised_text = self.tokenizer.decode(noised_response, skip_special_tokens=False)

                        self.debug_logger.info(f"\n[轮次 {iter_idx}] 加噪后的序列 s{iter_idx}")
                        self.debug_logger.info(f"  被替换的 token 数: {num_replaced}/{len(original_response)} ({replace_ratio:.1%})")
                        self.debug_logger.info(f"  理论噪声比例: {t[0].item():.1%}")
                        self.debug_logger.info(f"  加噪后文本: {noised_text[:200]}")

                        # 显示前 10 个 token 的对比
                        self.debug_logger.info(f"  前10个token对比 (原始 -> 加噪):")
                        for i in range(min(10, len(original_response))):
                            orig_tok = original_response[i]
                            noise_tok = noised_response[i]
                            if orig_tok != noise_tok:
                                orig_str = self.tokenizer.decode([orig_tok])
                                noise_str = self.tokenizer.decode([noise_tok])
                                self.debug_logger.info(f"    位置{i}: {orig_tok}('{orig_str}') -> {noise_tok}('{noise_str}') [EOS={self.eos_token_id}]")
            else:
                # 后续轮：使用上一轮的解码结果
                s_i = current_input_ids

                # ========== 记录精炼后的序列 ==========
                if should_log_detail:
                    s_i_sample = s_i[0].cpu().tolist()
                    response_indices = [i for i, mask in enumerate(response_mask_sample) if mask == 1]
                    if response_indices:
                        response_start = response_indices[0]
                        response_end = response_indices[-1] + 1
                        refined_response = s_i_sample[response_start:response_end]
                        refined_text = self.tokenizer.decode(refined_response, skip_special_tokens=False)

                        # 与原始序列对比
                        original_response = t0_sample[response_start:response_end]
                        num_diff = sum(1 for orig, refined in zip(original_response, refined_response)
                                      if orig != refined)
                        diff_ratio = num_diff / len(original_response) if original_response else 0

                        self.debug_logger.info(f"\n[轮次 {iter_idx}] 精炼后的序列 s{iter_idx}")
                        self.debug_logger.info(f"  与原始序列的差异: {num_diff}/{len(original_response)} ({diff_ratio:.1%})")
                        self.debug_logger.info(f"  精炼后文本: {refined_text[:200]}")

                        # 🔍 额外调试：显示 s1 的前5个 response token
                        self.debug_logger.info(f"  精炼后 response 的前5个token:")
                        for idx, (orig_tok, refined_tok) in enumerate(zip(original_response[:5], refined_response[:5])):
                            orig_str = self.tokenizer.decode([orig_tok])
                            refined_str = self.tokenizer.decode([refined_tok])
                            match = "✓" if orig_tok == refined_tok else "✗"
                            self.debug_logger.info(f"    位置{response_start+idx}: 原始={orig_tok}('{orig_str}') vs 精炼={refined_tok}('{refined_str}') {match}")

            # ===========================================================
            # Step 2: 前向传播
            # ===========================================================
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.fsdp_model(
                    input_ids=s_i,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                logits = output.logits  # [batch_size, seq_len, vocab_size]

            # ===========================================================
            # Step 3: 计算 loss（对原始 t0 的 next token prediction）
            # ===========================================================
            loss_i = compute_loss_on_response(
                logits=logits,
                labels=t0,  # 注意：始终是对原始 t0 计算 loss
                response_mask=response_mask,
            )
            losses.append(loss_i)

            # ========== 记录 loss 和预测质量 ==========
            if should_log_detail:
                self.debug_logger.info(f"  Loss (对原始t0): {loss_i.item():.6f}")

                # 分析预测的 top-1 token
                with torch.no_grad():
                    response_indices = [i for i, mask in enumerate(response_mask_sample) if mask == 1]
                    if response_indices and len(response_indices) > 0:
                        # 🔍 关键：先显示 instruction 最后一个位置预测 response 第一个位置
                        response_start = response_indices[0]
                        if response_start > 0:  # 确保有 instruction 区域
                            instruction_last_pos = response_start - 1
                            pred_first_response = torch.argmax(logits[0, instruction_last_pos]).item()
                            target_first_response = t0_sample[response_start]
                            pred_str = self.tokenizer.decode([pred_first_response])
                            target_str = self.tokenizer.decode([target_first_response])
                            match = "✓" if pred_first_response == target_first_response else "✗"

                            self.debug_logger.info(f"  【关键】Instruction最后位置预测Response第一个token:")
                            self.debug_logger.info(f"    位置{instruction_last_pos}->位置{response_start}: 预测={pred_first_response}('{pred_str}') vs 目标={target_first_response}('{target_str}') {match}")

                        # 然后看 response 区域的前几个 token 的预测（预测的是下一个）
                        sample_positions = response_indices[:5]  # 前5个 response token
                        logits_sample = logits[0, sample_positions].cpu()  # [5, vocab_size]
                        predicted_tokens = torch.argmax(logits_sample, dim=-1).tolist()
                        target_tokens = [t0_sample[pos + 1] for pos in sample_positions if pos + 1 < len(t0_sample)]

                        self.debug_logger.info(f"  Response前5个位置预测下一个token:")
                        for idx, (pred, target) in enumerate(zip(predicted_tokens, target_tokens)):
                            pred_str = self.tokenizer.decode([pred])
                            target_str = self.tokenizer.decode([target])
                            match = "✓" if pred == target else "✗"
                            next_pos = sample_positions[idx] + 1
                            self.debug_logger.info(f"    位置{sample_positions[idx]}->位置{next_pos}: 预测={pred}('{pred_str}') vs 目标={target}('{target_str}') {match}")

            # ===========================================================
            # Step 4: 贪婪解码（为下一轮准备）
            # ===========================================================
            if iter_idx < self.num_iterations - 1:  # 最后一轮不需要解码
                # 使用 with torch.no_grad() 和 detach() 来释放显存
                with torch.no_grad():
                    current_input_ids = greedy_decode_response(
                        logits=logits.detach(),  # 立即detach，释放梯度占用的显存
                        original_input_ids=t0,  # 🔧 修复：应该传入 t0，保留原始 instruction
                        response_mask=response_mask,
                    )

                # 注意：不显式删除 logits，因为 loss_i 的计算图可能还依赖它
                # Python 的垃圾回收会在循环结束时自动清理
                # 显式 del 可能在某些情况下破坏计算图，导致 backward() 失败

        # ===========================================================
        # Step 5: 聚合所有轮次的 loss
        # ===========================================================
        total_loss, loss_dict = compute_iterative_loss(
            losses=losses,
            weights=self.loss_weights,
        )

        # 添加额外的指标
        loss_dict['noise_mean'] = t.mean().item()

        # ========== 记录最终的 loss 聚合结果 ==========
        if should_log_detail:
            self.debug_logger.info(f"\n[Loss 聚合]")
            self.debug_logger.info(f"  各轮 loss:")
            for idx, (loss_val, weight) in enumerate(zip(losses, self.loss_weights)):
                self.debug_logger.info(f"    轮次{idx}: loss={loss_val.item():.6f}, weight={weight}, 加权loss={loss_val.item()*weight:.6f}")
            self.debug_logger.info(f"  Total loss: {total_loss.item():.6f}")
            self.debug_logger.info("=" * 80)

        return total_loss, loss_dict

    def training_step(self, batch: dict, global_step: int):
        """
        单个训练步骤（支持梯度累积）

        Args:
            batch: dict 包含 input_ids, attention_mask, position_ids, loss_mask
            global_step: int - 全局步数

        Returns:
            metrics: dict - 训练指标（仅在优化器更新时返回完整指标）
        """
        self.fsdp_model.train()

        # 判断是否需要在这一步更新优化器
        is_accumulation_step = (global_step + 1) % self.gradient_accumulation_steps != 0
        should_update_optimizer = not is_accumulation_step

        # ========== 阶段2：训练过程监控日志（每10步详细记录一次） ==========
        should_log_detail = (global_step % 10 == 0) and self.debug_logger is not None

        if should_log_detail:
            input_ids = batch["input_ids"]
            loss_mask = batch["loss_mask"]

            self.debug_logger.info("\n" + "=" * 80)
            self.debug_logger.info(f"[Step {global_step}] 训练批次详细信息")
            self.debug_logger.info("=" * 80)
            self.debug_logger.info(f"Batch size: {input_ids.shape[0]}")
            self.debug_logger.info(f"Sequence length: {input_ids.shape[1]}")
            self.debug_logger.info(f"Gradient accumulation: {(global_step % self.gradient_accumulation_steps) + 1}/{self.gradient_accumulation_steps}")
            self.debug_logger.info(f"Will update optimizer: {should_update_optimizer}")

            # 解码第一个样本查看内容
            first_sample_ids = input_ids[0].tolist()
            first_sample_text = self.tokenizer.decode(first_sample_ids, skip_special_tokens=False)
            first_sample_loss_mask = loss_mask[0].tolist()

            # 统计 response 区域
            response_length = sum(first_sample_loss_mask)
            prompt_length = len(first_sample_loss_mask) - response_length

            self.debug_logger.info(f"\n样本0内容（前500字符）:")
            self.debug_logger.info(f"  {first_sample_text[:500]}")
            self.debug_logger.info(f"\nPrompt 长度: {prompt_length} tokens")
            self.debug_logger.info(f"Response 长度: {response_length} tokens")

        # 计算多轮迭代的 loss
        loss, metrics = self._compute_iterative_loss(batch, global_step)

        # 梯度累积：loss需要除以累积步数
        scaled_loss = loss / self.gradient_accumulation_steps

        # 反向传播（累积阶段禁止FSDP梯度同步）
        if is_accumulation_step:
            # 累积阶段：使用 no_sync() 禁止梯度同步
            with self.fsdp_model.no_sync():
                scaled_loss.backward()
        else:
            # 最后一步：正常 backward，会进行梯度同步
            scaled_loss.backward()

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
        else:
            # 累积阶段，不更新优化器，但显示当前学习率
            metrics['grad_norm'] = 0.0  # 累积阶段不裁剪梯度
            metrics['lr'] = self.lr_scheduler.get_last_lr()[0]
            metrics['is_optimizer_step'] = False

        # ========== 记录训练指标 ==========
        if should_log_detail:
            self.debug_logger.info(f"\n[Step {global_step}] 训练指标:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    self.debug_logger.info(f"  {key}: {value:.6f}")
                else:
                    self.debug_logger.info(f"  {key}: {value}")
            self.debug_logger.info("=" * 80)

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

        # ========== 阶段4：记录完整的超参数配置 ==========
        if self.debug_logger is not None:
            self.debug_logger.info("\n" + "=" * 80)
            self.debug_logger.info("[阶段4] 超参数配置")
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

            self.debug_logger.info("\n迭代精炼配置:")
            self.debug_logger.info(f"  Num iterations: {self.num_iterations}")
            self.debug_logger.info(f"  Noise range: [{self.noise_min}, {self.noise_max}]")
            self.debug_logger.info(f"  Loss weights: {self.loss_weights}")
            self.debug_logger.info(f"  EOS token ID: {self.eos_token_id}")

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
                        'loss': f"{metrics['loss_total']:.4f}",
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
