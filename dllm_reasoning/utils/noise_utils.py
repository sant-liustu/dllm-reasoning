import torch
import torch.nn.functional as F


def q_sample(
    input_ids,
    maskable_mask,
    mask_token_id,
    min=0.0,
    max=1.0,
    eos_token_id=None,
    t=None,
    t_mask=None,
):
    x_0 = input_ids

    if t_mask is None:
        if t is None:
            t = torch.rand((x_0.shape[0],), dtype=torch.float, device=input_ids.device)
            t = min + (max - min) * t
        u = torch.rand_like(x_0, dtype=torch.float)  # t/T prob to mask
        t_mask = (u < t[:, None]) & maskable_mask

    x_t = x_0.masked_fill(t_mask, mask_token_id)

    if eos_token_id is not None:
        # get the last non-eos token index
        last_non_eos_token_idx = ((input_ids != eos_token_id) | (~maskable_mask)).sum(
            dim=-1
        ) - 1
        seq_len = x_0.shape[1]

        for i in range(x_0.shape[0]):
            if last_non_eos_token_idx[i] < seq_len - 1:  # with eos tokens
                t_mask_at_eos = t_mask[
                    i, last_non_eos_token_idx[i] + 1
                ]  # use arbitrary eos token
                # t_mask[i, last_non_eos_token_idx[i] + 2:] = False  # only learn the first eos token
                if t_mask_at_eos:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = mask_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = True
                else:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = eos_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = False

    return x_t, t, t_mask  #  True means it's "MASK" token and should have loss


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


def greedy_decode_response(
    logits: torch.Tensor,
    original_input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    只对 response 区域进行贪婪解码，instruction 区域保持不变

    这用于从加噪的序列中解码出精炼后的序列

    关键修复：logits[i] 预测的是 input_ids[i+1]，因此需要右移 predicted_ids

    Args:
        logits: [batch_size, seq_len, vocab_size] - 模型输出的 logits
        original_input_ids: [batch_size, seq_len] - 当前的 input_ids（包含 instruction + noisy response）
        response_mask: [batch_size, seq_len] - 1 表示 response 区域，0 表示 instruction 区域

    Returns:
        decoded_input_ids: [batch_size, seq_len] - 解码后的 input_ids
    """
    # 贪婪解码：选择概率最大的 token
    predicted_ids = logits.argmax(dim=-1)  # [batch_size, seq_len]

    # 🔧 修复：右移 predicted_ids，因为 logits[i] 预测的是 input_ids[i+1]
    # 例如：response 位置 244 应该使用 logits[243] 的预测
    # 创建一个右移后的 predicted_ids
    shifted_predicted_ids = torch.zeros_like(predicted_ids)
    shifted_predicted_ids[:, 1:] = predicted_ids[:, :-1]  # 向右移动 1 位
    # 第一个位置保持不变（通常是 BOS token，不在 response 区域）
    shifted_predicted_ids[:, 0] = original_input_ids[:, 0]

    # 只替换 response 区域，instruction 区域保持原样
    decoded_input_ids = torch.where(
        response_mask.bool(),
        shifted_predicted_ids,  # 使用右移后的预测
        original_input_ids
    )

    return decoded_input_ids
