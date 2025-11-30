"""
FlexAttention utilities for interleaved training.

Provides functions to create mask_mod and BlockMask from block_info.
"""

import torch
from typing import List, Tuple, Callable

try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    print("Warning: FlexAttention not available in this PyTorch version")


def create_mask_mod_from_block_info(
    block_info: List[Tuple[str, int]],
    prompt_len: int,
    seq_len: int,
) -> Callable:
    """
    Create a mask_mod function for FlexAttention from block_info.

    This implements the 6 attention mask rules for interleaved training:
    1. Prompt: standard causal
    2. Mask tokens:
       - Can see prompt and all PREVIOUS real blocks
       - Causal within same mask group
       - Cannot see corresponding real block
       - Cannot see any other mask groups
    3. Real block tokens:
       - Can see prompt and all PREVIOUS real blocks
       - Causal within same block
       - Cannot see ANY mask tokens
    4. Different mask groups cannot see each other

    Args:
        block_info: List of (seg_type, seg_len) tuples
        prompt_len: Length of prompt
        seq_len: Total sequence length

    Returns:
        mask_mod: function(b, h, q_idx, kv_idx) -> bool
    """
    # Build segments list with (seg_type, seg_start, seg_len)
    segments = []
    segments.append(('prompt', 0, prompt_len))

    current_pos = prompt_len
    for seg_type, seg_len in block_info:
        segments.append((seg_type, current_pos, seg_len))
        current_pos += seg_len

    # Precompute segment assignments for each position (optimization)
    pos_to_segment = {}
    for seg_idx, (seg_type, seg_start, seg_len) in enumerate(segments):
        for pos in range(seg_start, seg_start + seg_len):
            pos_to_segment[pos] = (seg_idx, seg_type, seg_start, seg_len)

    def mask_mod(b, h, q_idx, kv_idx):
        """
        Determine if position q_idx can attend to position kv_idx.

        Args:
            b: batch index (not used, same mask for all batches)
            h: head index (not used, same mask for all heads)
            q_idx: query position
            kv_idx: key/value position

        Returns:
            True if q_idx can attend to kv_idx, False otherwise
        """
        # Out of bounds check
        if q_idx >= seq_len or kv_idx >= seq_len:
            return False

        # Get segment info for q and kv
        if q_idx not in pos_to_segment or kv_idx not in pos_to_segment:
            return False

        q_seg_idx, q_type, q_start, q_len = pos_to_segment[q_idx]
        kv_seg_idx, kv_type, kv_start, kv_len = pos_to_segment[kv_idx]

        # Rule 1: Prompt - standard causal
        if q_type == 'prompt':
            if kv_type == 'prompt':
                # Causal within prompt
                return kv_idx <= q_idx
            else:
                # Prompt cannot see response
                return False

        # Rule 2: Real block tokens
        elif q_type == 'real':
            if kv_type == 'prompt':
                # Can see entire prompt
                return True
            elif kv_type == 'real':
                if kv_seg_idx < q_seg_idx:
                    # Can see all previous real blocks
                    return True
                elif kv_seg_idx == q_seg_idx:
                    # Causal within same block
                    return kv_idx <= q_idx
                else:
                    # Cannot see future blocks
                    return False
            elif kv_type == 'mask':
                # Real blocks CANNOT see ANY mask tokens
                return False

        # Rule 3: Mask tokens
        elif q_type == 'mask':
            if kv_type == 'prompt':
                # Can see entire prompt
                return True
            elif kv_type == 'real':
                # Find which real block this mask group corresponds to
                # Mask at segment q_seg_idx comes BEFORE the real block at q_seg_idx + 1
                # So it can see real blocks at indices < q_seg_idx
                # (because mask precedes its corresponding block)
                if kv_seg_idx < q_seg_idx:
                    # Can see all previous real blocks
                    return True
                else:
                    # Cannot see current or future real blocks
                    return False
            elif kv_type == 'mask':
                if kv_seg_idx == q_seg_idx:
                    # Causal within same mask group
                    return kv_idx <= q_idx
                else:
                    # Different mask groups cannot see each other
                    return False

        # Default: cannot attend
        return False

    return mask_mod


def create_block_mask_from_batch(
    batch: dict,
    device: torch.device,
) -> 'BlockMask':
    """
    Create FlexAttention BlockMask from a batch.

    使用 SDAR 的方法：预先计算 3D bool tensor [B, Q_LEN, KV_LEN]，
    使用纯 Tensor 操作（无 Python 控制流），然后用简单的 lambda 函数索引。

    这种方法的优势：
    1. 避免 vmap 限制：不使用 if 语句、.item()、字典查找
    2. 保证 GPU 执行：所有操作都是 Tensor 操作
    3. 与 PyTorch 编译器兼容

    Args:
        batch: Dict containing:
            - block_info: List[List[Tuple[str, int]]] - block_info for each sample
            - prompt_len: List[int] - prompt length for each sample
            - seq_lens: List[int] - actual sequence length for each sample
            - input_ids: [B, max_len] - padded input
        device: Device for BlockMask

    Returns:
        BlockMask for FlexAttention
    """
    if not FLEX_ATTN_AVAILABLE:
        raise RuntimeError("FlexAttention is not available in this PyTorch version")

    block_info_batch = batch["block_info"]
    prompt_len_batch = batch["prompt_len"]
    seq_lens = batch["seq_lens"]
    batch_size = len(block_info_batch)
    max_seq_len = batch["input_ids"].shape[1]

    # 预先计算 3D mask tensor [B, max_seq_len, max_seq_len]
    # 使用纯 Tensor 操作（类似 SDAR 的 block_diff_mask）
    attention_masks = []

    for b_idx in range(batch_size):
        # 为每个样本构造 2D mask [max_seq_len, max_seq_len]
        # 使用向量化的 Tensor 操作

        # 创建索引 grid
        q_indices = torch.arange(max_seq_len, device=device)[:, None]  # [max_seq_len, 1]
        kv_indices = torch.arange(max_seq_len, device=device)[None, :]  # [1, max_seq_len]

        # 构建 segment type 和 segment index 的 tensor
        # seg_types: [max_seq_len] - 0=prompt, 1=real, 2=mask
        # seg_indices: [max_seq_len] - segment index
        seg_types = torch.full((max_seq_len,), -1, dtype=torch.long, device=device)
        seg_indices = torch.full((max_seq_len,), -1, dtype=torch.long, device=device)

        # 填充 prompt
        prompt_len = prompt_len_batch[b_idx]
        seg_types[:prompt_len] = 0  # prompt type
        seg_indices[:prompt_len] = 0  # prompt segment index

        # 填充 block_info (real/mask segments)
        current_pos = prompt_len
        for seg_idx, (seg_type, seg_len) in enumerate(block_info_batch[b_idx], start=1):
            seg_type_id = 1 if seg_type == 'real' else 2  # 1=real, 2=mask
            seg_types[current_pos:current_pos + seg_len] = seg_type_id
            seg_indices[current_pos:current_pos + seg_len] = seg_idx
            current_pos += seg_len

        # 使用向量化操作构造 mask
        # [max_seq_len, 1]
        q_type = seg_types[:, None]
        kv_type = seg_types[None, :]
        q_seg_idx = seg_indices[:, None]
        kv_seg_idx = seg_indices[None, :]

        # 规则 1: Prompt tokens
        # prompt can only see earlier prompt tokens (causal)
        prompt_mask = (q_type == 0) & (kv_type == 0) & (kv_indices <= q_indices)

        # 规则 2: Real tokens
        # can see: all prompt, all previous real blocks, causal within same block
        real_see_prompt = (q_type == 1) & (kv_type == 0)
        real_see_prev_real = (q_type == 1) & (kv_type == 1) & (kv_seg_idx < q_seg_idx)
        real_see_same_real = (q_type == 1) & (kv_type == 1) & (kv_seg_idx == q_seg_idx) & (kv_indices <= q_indices)

        # 规则 3: Mask tokens
        # can see: all prompt, all previous real blocks, causal within same mask block
        mask_see_prompt = (q_type == 2) & (kv_type == 0)
        mask_see_prev_real = (q_type == 2) & (kv_type == 1) & (kv_seg_idx < q_seg_idx)
        mask_see_same_mask = (q_type == 2) & (kv_type == 2) & (kv_seg_idx == q_seg_idx) & (kv_indices <= q_indices)

        # 组合所有规则
        mask_2d = (prompt_mask |
                   real_see_prompt | real_see_prev_real | real_see_same_real |
                   mask_see_prompt | mask_see_prev_real | mask_see_same_mask)

        # 边界检查：超过实际序列长度的部分设为 False
        valid_mask = (q_indices < seq_lens[b_idx]) & (kv_indices < seq_lens[b_idx])
        mask_2d = mask_2d & valid_mask

        attention_masks.append(mask_2d)

    # Stack 成 [B, max_seq_len, max_seq_len]
    attention_mask = torch.stack(attention_masks, dim=0)

    # 使用简单的 lambda 索引（SDAR 方法）
    # 注意：需要使用 clamp 防止 FlexAttention 查询越界索引
    def safe_mask_lookup(b, h, q_idx, kv_idx):
        # FlexAttention 在 block-wise 处理时可能查询边界外的索引
        # 使用 clamp 确保索引在有效范围内
        q_safe = torch.clamp(q_idx, 0, max_seq_len - 1)
        kv_safe = torch.clamp(kv_idx, 0, max_seq_len - 1)
        return attention_mask[b, q_safe, kv_safe]

    block_mask = create_block_mask(
        safe_mask_lookup,
        B=batch_size,
        H=None,  # Broadcast across all heads
        Q_LEN=max_seq_len,
        KV_LEN=max_seq_len,
        device=device,
    )

    return block_mask


def verify_mask_rules(
    block_info: List[Tuple[str, int]],
    prompt_len: int,
    seq_len: int,
) -> bool:
    """
    Verify that the mask_mod created from block_info follows all 6 rules.

    This is for testing/debugging purposes.

    Returns:
        True if all rules are satisfied, False otherwise
    """
    mask_mod = create_mask_mod_from_block_info(block_info, prompt_len, seq_len)

    # Build segments
    segments = []
    segments.append(('prompt', 0, prompt_len))
    current_pos = prompt_len
    for seg_type, seg_len in block_info:
        segments.append((seg_type, current_pos, seg_len))
        current_pos += seg_len

    # Check rules
    errors = []

    # Find all segments by type
    prompt_seg = segments[0]
    mask_segs = [(idx, seg) for idx, seg in enumerate(segments) if seg[0] == 'mask']
    real_segs = [(idx, seg) for idx, seg in enumerate(segments) if seg[0] == 'real']

    # Rule 1: Real blocks cannot see mask tokens
    for real_idx, (real_type, real_start, real_len) in real_segs:
        for mask_idx, (mask_type, mask_start, mask_len) in mask_segs:
            for q in range(real_start, real_start + real_len):
                for kv in range(mask_start, mask_start + mask_len):
                    if mask_mod(0, 0, q, kv):
                        errors.append(f"Rule 1 violated: Real block at {q} can see mask at {kv}")

    # Rule 2: Masks can see prompt and previous real blocks
    for mask_idx, (mask_type, mask_start, mask_len) in mask_segs:
        # Should see entire prompt
        for q in range(mask_start, mask_start + mask_len):
            for kv in range(prompt_len):
                if not mask_mod(0, 0, q, kv):
                    errors.append(f"Rule 2 violated: Mask at {q} cannot see prompt at {kv}")

        # Should see previous real blocks
        for real_idx, (real_type, real_start, real_len) in real_segs:
            if real_idx < mask_idx:  # Previous real block
                for q in range(mask_start, mask_start + mask_len):
                    for kv in range(real_start, real_start + real_len):
                        if not mask_mod(0, 0, q, kv):
                            errors.append(f"Rule 2 violated: Mask at {q} cannot see previous real block at {kv}")

    # Rule 3: Masks are causal within their group
    for mask_idx, (mask_type, mask_start, mask_len) in mask_segs:
        for i in range(mask_len):
            for j in range(mask_len):
                q = mask_start + i
                kv = mask_start + j
                should_see = (j <= i)  # Causal
                actually_sees = mask_mod(0, 0, q, kv)
                if should_see != actually_sees:
                    errors.append(f"Rule 3 violated: Mask at {q} {'cannot' if should_see else 'can'} see mask at {kv}")

    # Print errors if any
    if errors:
        for error in errors[:10]:  # Print first 10 errors
            print(f"  ❌ {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False

    return True
