"""
迭代块状生成 (Iterative Block-wise Generation)

核心思想:
1. 每次添加 N 个 EOS token，可以生成 N+1 个新 token（利用 next token prediction）
2. 对每个新块进行多轮 refine（迭代优化）
3. 检测到 EOS 或达到最大长度时停止

与训练的对应关系:
- 训练时: 对 response 区域加噪 → refine → 计算 loss
- 推理时: 拼接 EOS 块 → refine → 解码生成
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def iterative_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
    max_new_tokens: int = 1024,
    add_eos_length: int = 127,
    refine_iter: int = 2,
    max_length: int = 8192,
    temperature: float = 0.0,
    attention_mask: Optional[torch.Tensor] = None,
    verbose_trace: bool = False,
    tokenizer = None,
) -> torch.Tensor:
    """
    迭代块状生成函数

    核心逻辑:
    1. 拼接 add_eos_length 个 EOS token
    2. 前向传播得到 logits
    3. 提取新块的 logits (处理 next token prediction 偏移)
    4. 解码生成 add_eos_length + 1 个新 token
    5. Refine refine_iter 轮
    6. 检测 EOS 或最大长度，决定是否继续下一块

    Args:
        model: 标准的因果语言模型 (AutoModelForCausalLM)
        input_ids: Prompt token IDs，形状 [batch, prompt_len]
        eos_token_id: EOS token ID (用于初始化块和检测停止)
        pad_token_id: PAD token ID (用于填充截断后的序列)
        max_new_tokens: 最大生成的 token 数量
        add_eos_length: 每块添加的 EOS token 数量
                       实际生成 add_eos_length + 1 个新 token
        refine_iter: 每块的 refine 轮数
        max_length: 序列的最大长度限制 (prompt + generated)
        temperature: 采样温度 (0.0 表示贪婪解码，暂时只支持贪婪)
        attention_mask: 注意力掩码，形状 [batch, prompt_len]，可选

    Returns:
        generated_ids: 生成的完整序列，形状 [batch, prompt_len + generated_len]
                      包含原始 prompt

    Raises:
        ValueError: 如果 input_ids 为空或超过 max_length

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("path/to/model")
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/model")
        >>> prompt = "What is 2+2?"
        >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        >>> output_ids = iterative_generate(
        ...     model=model,
        ...     input_ids=input_ids,
        ...     eos_token_id=tokenizer.eos_token_id,
        ...     pad_token_id=tokenizer.pad_token_id,
        ...     add_eos_length=127,
        ...     refine_iter=2,
        ... )
        >>> response = tokenizer.decode(output_ids[0, input_ids.size(1):])
    """
    # ==================== 参数验证 ====================
    if input_ids.size(1) == 0:
        raise ValueError("Empty prompt: input_ids has length 0")

    if input_ids.size(1) >= max_length:
        raise ValueError(
            f"Prompt length {input_ids.size(1)} exceeds or equals max_length {max_length}"
        )

    if eos_token_id is None:
        logger.warning(
            "No EOS token defined. Generation will only stop at max_new_tokens or max_length."
        )

    if temperature != 0.0:
        logger.warning(
            f"temperature={temperature} is set but sampling is not yet implemented. "
            f"Will use greedy decoding (temperature=0.0)."
        )

    if refine_iter < 1:
        raise ValueError(
            f"refine_iter must be at least 1 (got {refine_iter}). "
            f"At least one forward pass is required to generate tokens."
        )

    # ==================== 初始化 ====================
    batch_size = input_ids.size(0)
    device = input_ids.device
    generated_ids = input_ids.clone()

    # 计算需要的块数
    num_tokens_per_block = add_eos_length + 1  # 每块实际生成的 token 数
    max_blocks = (max_new_tokens + num_tokens_per_block - 1) // num_tokens_per_block

    logger.info(
        f"Starting iterative generation: "
        f"batch_size={batch_size}, prompt_len={input_ids.size(1)}, "
        f"add_eos_length={add_eos_length}, refine_iter={refine_iter}, "
        f"max_blocks={max_blocks}"
    )

    # 打印初始状态（只在 rank 0 打印）
    if verbose_trace and tokenizer is not None:
        # 检查是否在分布式环境中，只让 rank 0 打印
        import torch.distributed as dist
        should_print = not dist.is_initialized() or dist.get_rank() == 0

        if should_print:
            text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            print(f"\n{'='*80}")
            print(f"[初始状态] Prompt")
            print(f"  长度: {generated_ids.size(1)} tokens")
            print(f"  文本: [{text}]")
            print(f"{'='*80}")

    # ==================== 主生成循环 ====================
    for block_idx in range(max_blocks):
        pre_length = generated_ids.size(1)

        # ===== 步骤1: 添加 EOS 块 =====
        eos_block = torch.full(
            (batch_size, add_eos_length),
            eos_token_id,
            device=device,
            dtype=input_ids.dtype,
        )
        current_ids = torch.cat([generated_ids, eos_block], dim=1)
        # current_ids 长度: pre_length + add_eos_length

        # 打印添加 EOS 后的状态（只在 rank 0 打印）
        if verbose_trace and tokenizer is not None:
            import torch.distributed as dist
            should_print = not dist.is_initialized() or dist.get_rank() == 0

            if should_print:
                text = tokenizer.decode(current_ids[0], skip_special_tokens=False)
                print(f"\n{'='*80}")
                print(f"[Block {block_idx}] 添加了 {add_eos_length} 个 EOS token")
                print(f"  长度: {current_ids.size(1)} tokens")
                print(f"  文本: [{text}]")
                print(f"{'='*80}")

        # 构造 attention_mask (如果需要)
        if attention_mask is not None:
            # 扩展 attention_mask 以匹配 current_ids
            eos_mask = torch.ones(
                (batch_size, add_eos_length),
                device=device,
                dtype=attention_mask.dtype,
            )
            current_attention_mask = torch.cat([attention_mask, eos_mask], dim=1)
        else:
            current_attention_mask = None

        # ===== 步骤2: Refine 循环 =====
        current_ids = _refine_simple(
            model=model,
            current_ids=current_ids,
            pre_length=pre_length,
            add_eos_length=add_eos_length,
            refine_iter=refine_iter,
            attention_mask=current_attention_mask,
            block_idx=block_idx,
            verbose_trace=verbose_trace,
            tokenizer=tokenizer,
        )
        # current_ids 长度: pre_length + add_eos_length + 1

        # 更新 attention_mask (如果需要)
        if attention_mask is not None:
            last_token_mask = torch.ones(
                (batch_size, 1),
                device=device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([current_attention_mask, last_token_mask], dim=1)

        # ===== 步骤3: 检查停止条件 =====
        # 新生成的部分 (不包括 prompt)
        new_block = current_ids[:, pre_length:]

        # 条件1: 检测到 EOS
        if eos_token_id is not None and _has_eos(new_block, eos_token_id):
            logger.info(f"EOS detected at block {block_idx}, stopping generation")
            generated_ids = _truncate_at_eos(current_ids, eos_token_id, pad_token_id)
            break

        # 条件2: 超过最大长度
        if current_ids.size(1) >= max_length:
            logger.info(f"Reached max_length {max_length} at block {block_idx}, stopping generation")
            generated_ids = current_ids
            break

        # ===== 步骤4: 继续下一块 =====
        generated_ids = current_ids

        logger.debug(f"Block {block_idx} completed, current length: {generated_ids.size(1)}")

    logger.info(f"Generation completed: final length {generated_ids.size(1)}")
    return generated_ids


def _refine_simple(
    model: torch.nn.Module,
    current_ids: torch.Tensor,
    pre_length: int,
    add_eos_length: int,
    refine_iter: int,
    attention_mask: Optional[torch.Tensor] = None,
    block_idx: int = 0,
    verbose_trace: bool = False,
    tokenizer = None,
) -> torch.Tensor:
    """
    对当前块进行 refine (简单版本，不使用 KV cache)

    每轮 refine:
    1. 前向传播整个序列
    2. 提取新块的 logits (处理 next token prediction 偏移)
    3. 解码得到 add_eos_length + 1 个新 token
    4. 更新 current_ids

    Args:
        model: 语言模型
        current_ids: 当前序列，形状 [batch, pre_length + add_eos_length]
        pre_length: prompt 的长度（已生成的部分）
        add_eos_length: 本块添加的 EOS 数量
        refine_iter: refine 轮数
        attention_mask: 注意力掩码，形状 [batch, pre_length + add_eos_length]

    Returns:
        refined_ids: 更新后的序列，形状 [batch, pre_length + add_eos_length + 1]

    核心理解:
        - current_ids 长度: N = pre_length + add_eos_length
        - logits 长度: N
        - logits[:, i, :] 预测 input_ids[:, i+1]
        - 我们想预测位置 [pre_length, pre_length+1, ..., pre_length+add_eos_length]
          共 add_eos_length + 1 个位置
        - 需要 logits[:, pre_length-1 : pre_length+add_eos_length, :]
    """
    batch_size = current_ids.size(0)
    device = current_ids.device

    for refine_step in range(refine_iter):
        # ===== 前向传播 =====
        # 重要：只取前 pre_length + add_eos_length 个 token 作为输入
        # 这样确保每轮 refine 的输入长度相同，避免位置编码等因素影响
        input_length = pre_length + add_eos_length
        input_ids_for_refine = current_ids[:, :input_length]

        with torch.no_grad():
            if attention_mask is not None:
                attention_mask_for_refine = attention_mask[:, :input_length]
                outputs = model(input_ids_for_refine, attention_mask=attention_mask_for_refine, use_cache=False)
            else:
                outputs = model(input_ids_for_refine, use_cache=False)
            logits = outputs.logits  # [batch, input_length, vocab]

        # ===== 提取新块的 logits =====
        # 想预测位置: [pre_length, pre_length+1, ..., pre_length+add_eos_length]
        # 共 add_eos_length + 1 个位置
        # 需要 logits[:, pre_length-1 : pre_length+add_eos_length, :]
        new_block_logits = logits[:, pre_length-1 : pre_length+add_eos_length, :]
        # new_block_logits 形状: [batch, add_eos_length+1, vocab]

        # ===== 解码 (贪婪) =====
        predicted_tokens = new_block_logits.argmax(dim=-1)  # [batch, add_eos_length+1]

        # ===== Debug: 记录 logits 信息 =====
        if verbose_trace and tokenizer is not None:
            import torch.distributed as dist
            should_print = not dist.is_initialized() or dist.get_rank() == 0

            if should_print:
                print(f"\n{'='*80}")
                print(f"[Block {block_idx}] Refine 步骤 {refine_step+1}/{refine_iter} - Logits 调试信息")
                print(f"{'='*80}")
                print(f"当前序列长度: {current_ids.size(1)}")
                print(f"提取的 logits 范围: [{pre_length-1}:{pre_length+add_eos_length}]")
                print(f"需要预测的位置: [{pre_length}:{pre_length+add_eos_length+1}]")
                print()

                # 对于每个预测位置，记录：
                # 1. logits 的最大值位置（即预测的 token）
                # 2. logits 的 top-3 概率分布
                for i in range(add_eos_length + 1):
                    logit_pos = pre_length - 1 + i  # logits 的位置
                    pred_pos = pre_length + i        # 预测的 token 位置

                    # 获取这个位置的 logits
                    logits_at_pos = new_block_logits[0, i, :]  # [vocab_size]

                    # Top-3 预测
                    top3_values, top3_indices = torch.topk(logits_at_pos, k=3)
                    top3_tokens = [tokenizer.decode([idx.item()]) for idx in top3_indices]
                    top3_probs = torch.softmax(logits_at_pos, dim=0)[top3_indices]

                    # 预测的 token
                    predicted_token_id = predicted_tokens[0, i].item()
                    predicted_token_text = tokenizer.decode([predicted_token_id])

                    print(f"  位置 {pred_pos}:")
                    print(f"    - 由 logits[{logit_pos}] 预测")
                    print(f"    - 预测 token: {repr(predicted_token_text)} (ID: {predicted_token_id})")
                    print(f"    - Top-3: {[(repr(t), f'{p:.4f}') for t, p in zip(top3_tokens, top3_probs)]}")

                    # 如果不是第一次 refine，比较与当前位置的 token 是否相同
                    if refine_step > 0 and pred_pos < current_ids.size(1):
                        current_token_id = current_ids[0, pred_pos].item()
                        current_token_text = tokenizer.decode([current_token_id])
                        if current_token_id != predicted_token_id:
                            print(f"    - ⚠️  变化: {repr(current_token_text)} → {repr(predicted_token_text)}")
                        else:
                            print(f"    - ✅ 不变: {repr(current_token_text)}")
                    print()

                print(f"{'='*80}\n")

        # ===== 更新序列 =====
        # 第一轮 refine: 需要拼接最后一个 token（因为位置 pre_length+add_eos_length 还不存在）
        if refine_step == 0:
            # 前 add_eos_length 个位置: 直接替换
            current_ids[:, pre_length:] = predicted_tokens[:, :add_eos_length]

            # 最后 1 个位置: 拼接
            last_token = predicted_tokens[:, add_eos_length:add_eos_length+1]  # [batch, 1]
            current_ids = torch.cat([current_ids, last_token], dim=1)

            # 更新 attention_mask (如果需要)
            if attention_mask is not None:
                last_token_mask = torch.ones(
                    (batch_size, 1),
                    device=device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([attention_mask, last_token_mask], dim=1)
        else:
            # 后续 refine 轮: 所有位置都存在，直接替换
            current_ids[:, pre_length:pre_length+add_eos_length+1] = predicted_tokens

        # 打印当前 refine 步骤后的状态（只在 rank 0 打印）
        if verbose_trace and tokenizer is not None:
            import torch.distributed as dist
            should_print = not dist.is_initialized() or dist.get_rank() == 0

            if should_print:
                text = tokenizer.decode(current_ids[0], skip_special_tokens=False)
                print(f"\n{'='*80}")
                print(f"[Block {block_idx}] Refine 步骤 {refine_step+1}/{refine_iter} 完成")
                print(f"  长度: {current_ids.size(1)} tokens")
                print(f"  文本: [{text}]")
                print(f"{'='*80}")

    return current_ids


def _has_eos(tensor: torch.Tensor, eos_token_id: int) -> bool:
    """
    检查 tensor 中是否有 EOS token

    Args:
        tensor: 要检查的 tensor，形状 [batch, seq_len]
        eos_token_id: EOS token ID

    Returns:
        bool: 如果任意位置有 EOS，返回 True
    """
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
    sequences = sequences.clone()  # 避免修改原始 tensor

    for b in range(batch_size):
        # 找到第一个 EOS 的位置
        eos_positions = (sequences[b] == eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            # 保留到 EOS (包括 EOS)，后面填充 PAD
            if first_eos + 1 < sequences.size(1):
                sequences[b, first_eos+1:] = pad_token_id

    return sequences
