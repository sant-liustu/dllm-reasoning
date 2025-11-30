"""
Interleaved 生成器 (Next Block Prediction)

核心思想:
1. 根据指定的 mask 数量，一次性预测多个 token
2. 0 个 mask → 预测 1 个 token (标准自回归)
3. 1 个 mask → 预测 2 个 token (跳1步预测)
4. N 个 mask → 预测 N+1 个 token (跳N步预测)
5. 循环生成直到遇到 EOS 或达到最大长度

与 Interleaved SFT Training 的对应关系:
- 训练时: [P][M][R][M][R]... 格式，Mask token 用于 Next Block Prediction
- 推理时: 添加 N 个 Mask token，预测 N+1 个 Real token，保留预测结果并继续

训练中的 attention mask 模式:
- Prompt 看 Prompt (causal)
- Real 看 Prompt + 之前所有 Real (causal)
- Mask 看 Prompt + 之前所有 Real (不看同块的 Mask)

推理中的对应:
- 已生成部分 (Prompt + Real) 看所有之前内容 (causal)
- 新添加的 Mask 看所有已生成内容 (不看同块的 Mask)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def interleaved_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    eos_token_id: int,
    mask_token_id: int,
    pad_token_id: int,
    max_new_tokens: int = 1024,
    num_masks: int = 1,
    max_length: int = 8192,
    temperature: float = 0.0,
    verbose: bool = False,
    tokenizer = None,
) -> torch.Tensor:
    """
    Interleaved 生成函数 (Next Block Prediction)

    核心逻辑:
    1. 添加 num_masks 个 Mask token
    2. 前向传播得到 logits
    3. 提取预测位置的 logits (考虑 next token prediction 偏移)
    4. 解码生成 num_masks + 1 个新 token
    5. 将预测结果追加到序列，继续下一轮

    Args:
        model: 因果语言模型 (需要支持 Mask token)
        input_ids: Prompt token IDs，形状 [batch, prompt_len]
        eos_token_id: EOS token ID
        mask_token_id: Mask token ID (用于 Next Block Prediction)
        pad_token_id: PAD token ID
        max_new_tokens: 最大生成的 token 数量
        num_masks: 每次添加的 Mask token 数量
                   0: 标准自回归 (预测1个token)
                   1: 跳1步预测 (预测2个token)
                   N: 跳N步预测 (预测N+1个token)
        max_length: 序列最大长度
        temperature: 采样温度 (0.0 = 贪婪解码)
        verbose: 是否打印详细信息
        tokenizer: 用于 verbose 模式的解码

    Returns:
        generated_ids: 完整生成序列 [batch, prompt_len + generated_len]

    Example:
        >>> # 标准自回归 (每次预测1个token)
        >>> output = interleaved_generate(model, input_ids, num_masks=0, ...)

        >>> # 跳1步预测 (每次预测2个token)
        >>> output = interleaved_generate(model, input_ids, num_masks=1, ...)

        >>> # 跳3步预测 (每次预测4个token)
        >>> output = interleaved_generate(model, input_ids, num_masks=3, ...)
    """
    # ==================== 参数验证 ====================
    if input_ids.size(1) == 0:
        raise ValueError("Empty prompt: input_ids has length 0")

    if input_ids.size(1) >= max_length:
        raise ValueError(
            f"Prompt length {input_ids.size(1)} exceeds or equals max_length {max_length}"
        )

    if num_masks < 0:
        raise ValueError(f"num_masks must be >= 0, got {num_masks}")

    if temperature != 0.0:
        logger.warning(
            f"temperature={temperature} is set but sampling is not yet implemented. "
            f"Will use greedy decoding (temperature=0.0)."
        )

    # ==================== 初始化 ====================
    batch_size = input_ids.size(0)
    device = input_ids.device
    generated_ids = input_ids.clone()

    # 每个块生成的 token 数
    tokens_per_block = num_masks + 1
    max_blocks = (max_new_tokens + tokens_per_block - 1) // tokens_per_block

    if verbose:
        logger.info(
            f"Starting interleaved generation: "
            f"batch_size={batch_size}, prompt_len={input_ids.size(1)}, "
            f"num_masks={num_masks}, tokens_per_block={tokens_per_block}, "
            f"max_blocks={max_blocks}"
        )

    # ==================== 主生成循环 ====================
    for block_idx in range(max_blocks):
        pre_length = generated_ids.size(1)

        # ===== 步骤1: 添加 Mask token 并构造 position_ids =====
        if num_masks > 0:
            mask_block = torch.full(
                (batch_size, num_masks),
                mask_token_id,
                device=device,
                dtype=input_ids.dtype,
            )
            current_ids = torch.cat([generated_ids, mask_block], dim=1)

            # 构造 position_ids:
            # - 已生成部分: [0, 1, 2, ..., pre_length-1]
            # - Mask 部分: [pre_length, pre_length+1, ..., pre_length+num_masks-1]
            #   这样 Mask 的 position 告诉它要预测接下来的位置
            position_ids = torch.arange(
                0, pre_length + num_masks,
                dtype=torch.long,
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            # num_masks=0 时，不添加 mask，直接用当前序列
            current_ids = generated_ids
            position_ids = torch.arange(
                0, pre_length,
                dtype=torch.long,
                device=device
            ).unsqueeze(0).expand(batch_size, -1)

        if verbose and tokenizer is not None:
            text = tokenizer.decode(current_ids[0], skip_special_tokens=False)
            print(f"\n{'='*80}")
            print(f"[Block {block_idx}] 添加了 {num_masks} 个 Mask token")
            print(f"  序列长度: {current_ids.size(1)} tokens")
            print(f"  Position IDs: {position_ids[0].tolist()}")
            print(f"  文本: {text}")
            print(f"{'='*80}")

        # ===== 步骤2: 前向传播 (带 position_ids) =====
        with torch.no_grad():
            outputs = model(current_ids, position_ids=position_ids, use_cache=False)
            logits = outputs.logits  # [batch, seq_len, vocab]

        # ===== 步骤3: 提取预测位置的 logits =====
        # 我们要预测的位置: [pre_length, pre_length+1, ..., pre_length+num_masks]
        # 共 num_masks + 1 个位置
        #
        # 因为 logits[:, i, :] 预测的是 input_ids[:, i+1]
        # 所以:
        # - logits[:, pre_length-1, :] 预测位置 pre_length
        # - logits[:, pre_length, :]   预测位置 pre_length+1
        # - ...
        # - logits[:, pre_length+num_masks-1, :] 预测位置 pre_length+num_masks
        #
        # 因此需要提取: logits[:, pre_length-1 : pre_length+num_masks, :]

        pred_logits = logits[:, pre_length-1 : pre_length+num_masks, :]
        # pred_logits 形状: [batch, num_masks+1, vocab]

        # ===== 步骤4: 解码 (贪婪) =====
        predicted_tokens = pred_logits.argmax(dim=-1)  # [batch, num_masks+1]

        if verbose and tokenizer is not None:
            print(f"\n[Block {block_idx}] 预测了 {tokens_per_block} 个 token:")
            for i in range(tokens_per_block):
                token_id = predicted_tokens[0, i].item()
                token_text = tokenizer.decode([token_id])
                print(f"  位置 {pre_length + i}: {repr(token_text)} (ID: {token_id})")

        # ===== 步骤5: 更新序列 =====
        # 将预测的 token 追加到已生成序列
        generated_ids = torch.cat([generated_ids, predicted_tokens], dim=1)

        if verbose and tokenizer is not None:
            text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            print(f"\n[Block {block_idx}] 更新后的序列:")
            print(f"  长度: {generated_ids.size(1)} tokens")
            print(f"  文本: {text}")
            print(f"{'='*80}\n")

        # ===== 步骤6: 检查停止条件 =====
        # 新生成的部分
        new_block = generated_ids[:, pre_length:]

        # 条件1: 检测到 EOS
        if _has_eos(new_block, eos_token_id):
            if verbose:
                logger.info(f"EOS detected at block {block_idx}, stopping generation")
            generated_ids = _truncate_at_eos(generated_ids, eos_token_id, pad_token_id)
            break

        # 条件2: 超过最大长度
        if generated_ids.size(1) >= max_length:
            if verbose:
                logger.info(f"Reached max_length {max_length} at block {block_idx}")
            break

        # 条件3: 已生成足够多的 token
        total_generated = generated_ids.size(1) - input_ids.size(1)
        if total_generated >= max_new_tokens:
            if verbose:
                logger.info(f"Generated {total_generated} tokens, reaching max_new_tokens {max_new_tokens}")
            break

    if verbose:
        final_len = generated_ids.size(1)
        logger.info(f"Generation completed: final length {final_len}, generated {final_len - input_ids.size(1)} tokens")

    return generated_ids


def _has_eos(tensor: torch.Tensor, eos_token_id: int) -> bool:
    """检查 tensor 中是否有 EOS token"""
    return (tensor == eos_token_id).any().item()


def _truncate_at_eos(
    sequences: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """
    截断到第一个 EOS 之后，后面用 PAD 填充

    Args:
        sequences: 序列 tensor，形状 [batch, seq_len]
        eos_token_id: EOS token ID
        pad_token_id: PAD token ID

    Returns:
        截断后的序列，形状 [batch, seq_len]

    示例:
        输入: [tok1, tok2, eos, tok3, tok4]
        输出: [tok1, tok2, eos, pad, pad]
    """
    batch_size = sequences.size(0)
    sequences = sequences.clone()

    for b in range(batch_size):
        eos_positions = (sequences[b] == eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            if first_eos + 1 < sequences.size(1):
                sequences[b, first_eos+1:] = pad_token_id

    return sequences
