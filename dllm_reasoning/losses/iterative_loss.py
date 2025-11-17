"""
迭代精炼训练的损失计算函数

关键点：
1. 这是因果语言模型（AR model），所以预测的是 next token
2. 只在 response 区域计算 loss
3. 支持多轮迭代的 loss 聚合
"""

import torch
import torch.nn.functional as F


def compute_loss_on_response(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    只在 response 区域计算 next token prediction 的 loss

    【重要】这里实现的是标准的因果语言模型 loss，即：
    - logits[:, :-1] 预测 labels[:, 1:]
    - 即每个位置预测下一个 token

    Args:
        logits: [batch_size, seq_len, vocab_size] - 模型输出的 logits
        labels: [batch_size, seq_len] - 原始的 token IDs（ground truth）
        response_mask: [batch_size, seq_len] - 1 表示 response 区域，0 表示 instruction 区域

    Returns:
        loss: scalar tensor - 平均损失
    """
    batch_size, seq_len, vocab_size = logits.shape

    # ============================================================
    # 关键：这是因果语言模型，预测下一个 token
    # ============================================================
    # logits[:, i] 预测 labels[:, i+1]
    # 所以我们需要 shift
    shift_logits = logits[:, :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[:, 1:].contiguous()      # [batch_size, seq_len-1]
    shift_mask = response_mask[:, 1:].contiguous()  # [batch_size, seq_len-1]

    # Flatten for cross entropy
    shift_logits_flat = shift_logits.view(-1, vocab_size)  # [batch_size * (seq_len-1), vocab_size]
    shift_labels_flat = shift_labels.view(-1)              # [batch_size * (seq_len-1)]

    # 计算 cross entropy loss（不做 reduction，保持每个 token 的 loss）
    loss_per_token = F.cross_entropy(
        shift_logits_flat,
        shift_labels_flat,
        reduction='none'
    )  # [batch_size * (seq_len-1)]

    # Reshape 回来
    loss_per_token = loss_per_token.view(batch_size, seq_len - 1)  # [batch_size, seq_len-1]

    # ============================================================
    # 只保留 response 区域的 loss
    # ============================================================
    loss_per_token = loss_per_token * shift_mask.float()

    # 计算平均 loss（只在有效 token 上平均）
    valid_tokens = shift_mask.sum()
    if valid_tokens > 0:
        loss = loss_per_token.sum() / valid_tokens
    else:
        # 重要：返回一个在计算图中的 0 loss（有 grad_fn）
        # 不能用 torch.tensor(0.0)，因为它不在计算图中会导致 backward() 失败
        loss = (logits.sum() * 0.0)  # 保留计算图，但值为 0

    return loss


def compute_iterative_loss(
    losses: list,
    weights: list = None,
) -> tuple:
    """
    聚合多轮迭代的 loss

    Args:
        losses: list of scalar tensors - 每一轮的 loss
        weights: list of floats - 每一轮 loss 的权重（默认均等）

    Returns:
        total_loss: scalar tensor - 加权总 loss
        loss_dict: dict - 各轮 loss 的详细信息（用于日志）
    """
    num_iterations = len(losses)

    if weights is None:
        weights = [1.0] * num_iterations

    assert len(weights) == num_iterations, "weights 的长度必须与 losses 一致"

    # 加权求和
    total_loss = sum(w * l for w, l in zip(weights, losses))

    # 构造日志字典
    loss_dict = {
        f'loss_s{i}': losses[i].item()
        for i in range(num_iterations)
    }
    loss_dict['loss_total'] = total_loss.item()

    return total_loss, loss_dict
