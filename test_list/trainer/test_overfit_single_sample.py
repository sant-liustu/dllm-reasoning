"""
小规模过拟合测试

这是验证训练流程正确性最可靠的方法：
- 使用 1-2 个简单样本
- 训练足够多的步数（100-200 步）
- 验证 loss 能否降到接近 0

如果模型无法在 1-2 个样本上过拟合，说明训练流程有bug！

可能的问题：
1. 标签对齐错误
2. 梯度计算错误
3. 优化器配置错误
4. Loss 计算错误
5. Mask 注意力规则错误
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 禁用 torch.compile（避免 dynamo 跟踪错误）
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.optim import AdamW

from dllm_reasoning.model.DLLM.modeling_dllm import DLLMForCausalLM
from dllm_reasoning.trainer.flex_attention_utils import create_block_mask_from_batch


def create_simple_sample(tokenizer, device):
    """
    创建一个简单的样本用于过拟合测试

    结构：
    - Prompt: 3 个 token
    - Response: 8 个 token (2个完整的block，block_size=4)
    - 交错后: [P P P] [M M M] [R R R R] [M M M] [R R R R]
    """
    prompt_len = 3
    block_size = 4
    mask_len = block_size - 1  # 3

    # 使用固定的 token（确保可重复）
    prompt_tokens = torch.tensor([1000, 1001, 1002], device=device)
    mask_token_id = tokenizer.eos_token_id
    mask_tokens = torch.full((mask_len,), mask_token_id, device=device)

    # Response tokens (2个完整的block)
    response_tokens = torch.tensor([2000, 2001, 2002, 2003, 2010, 2011, 2012, 2013], device=device)

    # 构造交错序列
    # Block 0: [2000, 2001, 2002, 2003]
    # Block 1: [2010, 2011, 2012, 2013]
    block1 = response_tokens[:4]
    block2 = response_tokens[4:8]

    input_ids = torch.cat([
        prompt_tokens,   # [1000, 1001, 1002]
        mask_tokens,     # [EOS, EOS, EOS]
        block1,          # [2000, 2001, 2002, 2003]
        mask_tokens,     # [EOS, EOS, EOS]
        block2           # [2010, 2011, 2012, 2013]
    ], dim=0).unsqueeze(0)  # [1, 16]

    # Position IDs
    current_pos = 0
    position_ids_list = []

    # Prompt positions: [0, 1, 2]
    position_ids_list.append(torch.arange(0, prompt_len, device=device))
    current_pos = prompt_len

    # Mask 0 positions: [3, 4, 5]
    position_ids_list.append(torch.arange(current_pos, current_pos + mask_len, device=device))
    current_pos += mask_len

    # Block 0 positions: [3, 4, 5, 6] (共享 RoPE 位置)
    position_ids_list.append(torch.arange(current_pos - mask_len, current_pos - mask_len + 4, device=device))
    current_pos += 4

    # Mask 1 positions: [7, 8, 9]
    position_ids_list.append(torch.arange(current_pos, current_pos + mask_len, device=device))
    current_pos += mask_len

    # Block 1 positions: [7, 8, 9, 10]
    position_ids_list.append(torch.arange(current_pos - mask_len, current_pos - mask_len + 4, device=device))

    position_ids = torch.cat(position_ids_list, dim=0).unsqueeze(0)  # [1, 16]

    # Labels
    # Prompt: -100 (3个)
    # Mask 0: [2001, 2002, 2003] (预测block0的第2,3,4个token)
    # Block 0: [2001, 2002, 2003, 2010] (AR预测)
    # Mask 1: [2011, 2012, 2013] (预测block1的第2,3,4个token)
    # Block 1: [2011, 2012, 2013, -100] (AR预测，最后一个没有下一个)
    labels = torch.tensor([
        -100, -100, -100,        # Prompt
        2001, 2002, 2003,        # Mask 0
        2001, 2002, 2003, 2010,  # Block 0
        2011, 2012, 2013,        # Mask 1
        2011, 2012, 2013, -100   # Block 1
    ], device=device).unsqueeze(0)  # [1, 16]

    # Block info
    block_info = [
        ('mask', mask_len),
        ('real', 4),
        ('mask', mask_len),
        ('real', 4),
    ]

    batch = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'labels': labels,
        'block_info': [block_info],
        'prompt_len': [prompt_len],
        'seq_lens': [input_ids.shape[1]],
    }

    return batch


def test_overfit_single_sample():
    """
    在单个样本上过拟合测试
    """
    print("=" * 80)
    print("小规模过拟合测试")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B"

    # 加载模型和 tokenizer
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DLLMForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)

    model.train()
    print(f"✓ 模型加载成功（设备: {device}）")

    # 创建样本
    print("\n创建测试样本...")
    batch = create_simple_sample(tokenizer, device)
    print(f"✓ 样本创建成功")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Sequence length: {batch['seq_lens'][0]}")
    print(f"  Block info: {batch['block_info'][0]}")

    # 计算有效 loss 位置数
    valid_mask = (batch['labels'] != -100)
    num_valid = valid_mask.sum().item()
    print(f"  有效 loss 位置数: {num_valid}")

    # 配置优化器（较大的学习率用于快速过拟合）
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # 训练循环
    print("\n开始训练...")
    num_steps = 200
    log_interval = 20

    initial_loss = None
    loss_history = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # 构造 BlockMask
        block_mask = create_block_mask_from_batch(batch, device)

        # 前向传播
        outputs = model(
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

        # 反向传播
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)

        if initial_loss is None:
            initial_loss = loss_value

        if (step + 1) % log_interval == 0:
            # 计算准确率
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = (predictions == labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()

            print(f"  Step {step+1:3d}/{num_steps}: Loss = {loss_value:.6f}, Accuracy = {accuracy.item():.4f}")

    final_loss = loss_history[-1]

    # 评估结果
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)

    print(f"\n初始 loss: {initial_loss:.6f}")
    print(f"最终 loss: {final_loss:.6f}")
    print(f"Loss 下降: {initial_loss - final_loss:.6f}")
    print(f"下降比例: {(1 - final_loss / initial_loss) * 100:.2f}%")

    # 最终准确率
    with torch.no_grad():
        block_mask = create_block_mask_from_batch(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask=block_mask,
        )
        logits = outputs.logits
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels) & valid_mask
        final_accuracy = correct.sum().float() / valid_mask.sum().float()

    print(f"最终准确率: {final_accuracy.item():.4f}")

    # 逐位置检查
    print("\n逐位置预测检查（仅显示有效位置）:")
    input_ids = batch['input_ids'][0]
    labels_1d = batch['labels'][0]
    predictions_1d = predictions[0]

    for i in range(len(labels_1d)):
        if labels_1d[i] != -100:
            input_token = input_ids[i].item()
            expected = labels_1d[i].item()
            predicted = predictions_1d[i].item()
            match = "✓" if expected == predicted else "✗"
            print(f"  位置 {i:2d}: input={input_token:5d}, 期望={expected:5d}, 预测={predicted:5d} {match}")

    # 判断是否通过
    print("\n" + "=" * 80)
    print("判断标准")
    print("=" * 80)

    success = True

    # 标准 1: Loss 显著下降
    if final_loss < initial_loss * 0.1:  # 下降到初始的10%以下
        print("✅ Loss 显著下降（下降到初始值的10%以下）")
    else:
        print(f"⚠️  Loss 下降不够（当前是初始值的 {final_loss/initial_loss*100:.1f}%，应该 < 10%）")
        success = False

    # 标准 2: 最终 loss 接近 0
    if final_loss < 0.1:
        print(f"✅ 最终 loss 接近 0（{final_loss:.6f} < 0.1）")
    else:
        print(f"❌ 最终 loss 仍然较高（{final_loss:.6f}，应该 < 0.1）")
        success = False

    # 标准 3: 准确率高
    if final_accuracy.item() > 0.95:
        print(f"✅ 准确率高于 95%（{final_accuracy.item():.4f}）")
    else:
        print(f"⚠️  准确率低于 95%（{final_accuracy.item():.4f}）")
        success = False

    return success


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("小规模过拟合测试 - 验证训练流程正确性")
    print("=" * 100 + "\n")

    result = test_overfit_single_sample()

    print("\n" + "=" * 100)
    if result:
        print("✅ 过拟合测试通过！模型能够在单个样本上收敛，训练流程正确。")
    else:
        print("❌ 过拟合测试失败！模型无法在单个样本上充分收敛，可能存在以下问题：")
        print("   1. 标签对齐错误")
        print("   2. 梯度计算错误")
        print("   3. Mask 注意力规则错误")
        print("   4. 优化器配置问题")
    print("=" * 100 + "\n")

    exit(0 if result else 1)
