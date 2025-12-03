#!/usr/bin/env python3
"""
深入调试：为什么R₁会影响M₁的预测？

按照FlexAttention mask机制，M₁应该看不到R₁。
但实验显示：
- [P][M₁]: M₁准确率 0%
- [P][M₁][R₁]: M₁准确率 66.67%

这个脚本会：
1. 对比两种情况下M₁位置的logits
2. 检查每一层的hidden states
3. 验证attention mask是否正确
4. 找出R₁影响M₁的具体机制
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset


def get_m1_hidden_states_and_logits(model, input_ids, position_ids, block_info, prompt_len, device):
    """
    获取M₁位置的每层hidden states和最终logits

    Returns:
        - all_hidden_states: 每层的hidden states
        - final_logits: M₁位置的最终logits
    """
    # 注册hook来获取每层的hidden states
    hidden_states_dict = {}

    def hook_fn(name):
        def hook(module, input, output):
            # output是hidden states
            if isinstance(output, tuple):
                hidden_states_dict[name] = output[0].detach().clone()
            else:
                hidden_states_dict[name] = output.detach().clone()
        return hook

    # 注册hooks
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(hook_fn(f'layer_{i}'))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            block_info=[block_info],
            prompt_len=[prompt_len],
            seq_lens=[input_ids.size(0)],
            use_cache=False,
            output_hidden_states=True,
        )

    # 移除hooks
    for hook in hooks:
        hook.remove()

    # M₁位置在prompt_len
    m1_pos = prompt_len

    # 提取每层M₁的hidden states
    layer_hiddens = []
    for i in range(len(model.model.layers)):
        layer_name = f'layer_{i}'
        if layer_name in hidden_states_dict:
            hidden = hidden_states_dict[layer_name][0, m1_pos, :]  # [hidden_size]
            layer_hiddens.append(hidden)

    # M₁的最终logits
    m1_logits = outputs.logits[0, m1_pos, :]  # [vocab_size]

    return layer_hiddens, m1_logits


def compare_configurations(model, tokenizer, sample, device):
    """
    对比[P][M₁]和[P][M₁][R₁]两种配置
    """
    print("="*80)
    print("对比分析：[P][M₁] vs [P][M₁][R₁]")
    print("="*80)

    prompt_len = sample['prompt_len']
    input_ids_full = sample['input_ids'].to(device)
    position_ids_full = sample['position_ids'].to(device)
    labels_full = sample['labels'].to(device)
    block_info = sample['block_info']

    # 配置1: [P][M₁]
    print("\n1️⃣  配置1: [P][M₁]")
    truncate_pos_1 = prompt_len + 3  # Prompt + 3个mask tokens
    input_ids_1 = input_ids_full[:truncate_pos_1]
    position_ids_1 = position_ids_full[:truncate_pos_1]
    block_info_1 = [('mask', 3)]

    print(f"   序列长度: {input_ids_1.size(0)}")
    print(f"   Block info: {block_info_1}")

    hiddens_1, logits_1 = get_m1_hidden_states_and_logits(
        model, input_ids_1, position_ids_1, block_info_1, prompt_len, device
    )

    # 配置2: [P][M₁][R₁]
    print("\n2️⃣  配置2: [P][M₁][R₁]")
    truncate_pos_2 = prompt_len + 7  # Prompt + 3 mask + 4 real
    input_ids_2 = input_ids_full[:truncate_pos_2]
    position_ids_2 = position_ids_full[:truncate_pos_2]
    block_info_2 = [('mask', 3), ('real', 4)]

    print(f"   序列长度: {input_ids_2.size(0)}")
    print(f"   Block info: {block_info_2}")

    hiddens_2, logits_2 = get_m1_hidden_states_and_logits(
        model, input_ids_2, position_ids_2, block_info_2, prompt_len, device
    )

    # 对比分析
    print(f"\n{'='*80}")
    print("层级对比分析")
    print(f"{'='*80}\n")

    print(f"总共{len(hiddens_1)}层")

    # 对比每层的hidden states
    print(f"\n{'层号':^8} | {'L2距离':^15} | {'余弦相似度':^15} | {'差异说明':^20}")
    print("-"*80)

    for i, (h1, h2) in enumerate(zip(hiddens_1, hiddens_2)):
        # L2距离
        l2_dist = torch.norm(h1 - h2).item()

        # 余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()

        # 判断差异程度
        if l2_dist < 0.01:
            diff_desc = "几乎相同"
        elif l2_dist < 0.1:
            diff_desc = "轻微差异"
        elif l2_dist < 1.0:
            diff_desc = "中等差异"
        else:
            diff_desc = "显著差异"

        print(f"Layer {i:2d} | {l2_dist:>12.6f}  | {cos_sim:>12.6f}  | {diff_desc:^20}")

    # 对比最终logits
    print(f"\n{'='*80}")
    print("最终Logits对比")
    print(f"{'='*80}\n")

    # 获取ground truth
    m1_pos = prompt_len
    gt_token = labels_full[m1_pos].item()

    # 配置1的预测
    pred_1 = torch.argmax(logits_1).item()
    prob_1_gt = torch.softmax(logits_1, dim=-1)[gt_token].item()
    prob_1_pred = torch.softmax(logits_1, dim=-1)[pred_1].item()

    # 配置2的预测
    pred_2 = torch.argmax(logits_2).item()
    prob_2_gt = torch.softmax(logits_2, dim=-1)[gt_token].item()
    prob_2_pred = torch.softmax(logits_2, dim=-1)[pred_2].item()

    # 解码token
    gt_text = tokenizer.decode([gt_token])
    pred_1_text = tokenizer.decode([pred_1])
    pred_2_text = tokenizer.decode([pred_2])

    print(f"Ground truth: {gt_token:6d} '{gt_text}'")
    print()
    print(f"配置1 [P][M₁]:")
    print(f"  预测: {pred_1:6d} '{pred_1_text}' (prob={prob_1_pred:.4f})")
    print(f"  GT概率: {prob_1_gt:.4f}")
    print(f"  正确: {'✅' if pred_1 == gt_token else '❌'}")
    print()
    print(f"配置2 [P][M₁][R₁]:")
    print(f"  预测: {pred_2:6d} '{pred_2_text}' (prob={prob_2_pred:.4f})")
    print(f"  GT概率: {prob_2_gt:.4f}")
    print(f"  正确: {'✅' if pred_2 == gt_token else '❌'}")
    print()

    # Top-5对比
    print(f"{'='*80}")
    print("Top-5预测对比")
    print(f"{'='*80}\n")

    top5_1 = torch.topk(logits_1, 5)
    top5_2 = torch.topk(logits_2, 5)

    print("配置1 [P][M₁] Top-5:")
    for i, (val, idx) in enumerate(zip(top5_1.values, top5_1.indices)):
        token_text = tokenizer.decode([idx.item()])
        prob = torch.softmax(logits_1, dim=-1)[idx].item()
        marker = "⭐" if idx.item() == gt_token else "  "
        print(f"  {i+1}. {marker} {idx.item():6d} '{token_text}' (prob={prob:.4f})")

    print("\n配置2 [P][M₁][R₁] Top-5:")
    for i, (val, idx) in enumerate(zip(top5_2.values, top5_2.indices)):
        token_text = tokenizer.decode([idx.item()])
        prob = torch.softmax(logits_2, dim=-1)[idx].item()
        marker = "⭐" if idx.item() == gt_token else "  "
        print(f"  {i+1}. {marker} {idx.item():6d} '{token_text}' (prob={prob:.4f})")

    # Logits差异统计
    print(f"\n{'='*80}")
    print("Logits差异统计")
    print(f"{'='*80}\n")

    logits_diff = logits_2 - logits_1
    print(f"L2距离: {torch.norm(logits_diff).item():.6f}")
    print(f"最大增加: {logits_diff.max().item():.6f} (token {logits_diff.argmax().item()})")
    print(f"最大减少: {logits_diff.min().item():.6f} (token {logits_diff.argmin().item()})")
    print(f"平均差异: {logits_diff.abs().mean().item():.6f}")


def main():
    MODEL_PATH = "dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
    DATA_PATH = "data/openr1.parquet"

    print("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    model.train()

    print("加载数据集...")
    dataset = InterleavedSFTDataset(
        parquet_files=DATA_PATH,
        tokenizer=tokenizer,
        prompt_key="prompt",
        response_key="target",
        block_size=4,
        max_length=6000,
        truncation="right",
    )

    sample = dataset[0]

    print(f"\n样本信息:")
    print(f"  Prompt长度: {sample['prompt_len']}")
    print(f"  总序列长度: {sample['input_ids'].shape[0]}")
    print()

    # 运行对比分析
    compare_configurations(model, tokenizer, sample, device)


if __name__ == "__main__":
    main()
