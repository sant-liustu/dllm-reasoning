"""
渐进式过拟合测试 + Block-by-Block 快速解码验证

测试策略：
1. Level 0: 基础 AR 预测（无 EOS）- 验证基础能力
2. Level 1: 1 个 EOS - 验证 P[-1]→R1, EOS→R2
3. Level 2: 2 个 EOS - 验证多 EOS 独立预测
4. Level 3: 3 个 EOS - 完整训练配置（block_size=4）
5. Block-by-Block 解码验证 - 验证快速推理能力

这是验证 Next Block Prediction 的完整测试！
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 禁用 torch.compile
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.optim import AdamW

from dllm_reasoning.model.DLLM.modeling_dllm import DLLMForCausalLM
from dllm_reasoning.trainer.flex_attention_utils import create_block_mask_from_batch


def test_level_0_basic_ar(model, tokenizer, device):
    """
    Level 0: 基础 AR 预测（无 EOS）

    序列: [P P P] [R1 R2 R3 R4]
    预测: P[-1]→R1, R1→R2, R2→R3, R3→R4
    """
    print("\n" + "=" * 100)
    print("Level 0: 基础 AR 预测（无 EOS）")
    print("=" * 100)

    # 构造样本
    prompt_tokens = torch.tensor([1000, 1001, 1002], device=device)
    response_tokens = torch.tensor([2000, 2001, 2002, 2003], device=device)

    input_ids = torch.cat([prompt_tokens, response_tokens], dim=0).unsqueeze(0)
    position_ids = torch.arange(0, len(input_ids[0]), device=device).unsqueeze(0)

    # Labels: P[-1]→R1, R1→R2, R2→R3, R3→R4
    labels = torch.tensor([-100, -100, 2000, 2001, 2002, 2003, -100], device=device).unsqueeze(0)

    print(f"输入序列: {input_ids[0].tolist()}")
    print(f"Labels:   {labels[0].tolist()}")
    print(f"DEBUG: 准备训练...")

    # 训练
    optimizer = AdamW(model.parameters(), lr=1e-4)
    print(f"DEBUG: Optimizer 创建成功")
    model.train()
    print(f"DEBUG: 模型设置为训练模式")

    num_steps = 100
    print(f"DEBUG: 开始训练 {num_steps} 步...")
    for step in range(num_steps):
        optimizer.zero_grad()

        print(f"DEBUG: Step {step+1} - 开始前向传播...", flush=True)
        outputs = model(input_ids=input_ids, position_ids=position_ids)
        print(f"DEBUG: Step {step+1} - 前向传播完成", flush=True)
        logits = outputs.logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                valid_mask = (labels != -100)
                correct = (predictions == labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()
            print(f"  Step {step+1:3d}: Loss = {loss.item():.6f}, Accuracy = {accuracy.item():.4f}")

    # 验证
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, position_ids=position_ids)
        logits = outputs.logits
        predictions = logits.argmax(dim=-1)

        print(f"\n最终预测验证:")
        print(f"  P[-1]({input_ids[0,2].item()}) → 期望{response_tokens[0].item()}, 预测{predictions[0,2].item()} {'✓' if predictions[0,2] == response_tokens[0] else '✗'}")
        print(f"  R1({response_tokens[0].item()}) → 期望{response_tokens[1].item()}, 预测{predictions[0,3].item()} {'✓' if predictions[0,3] == response_tokens[1] else '✗'}")
        print(f"  R2({response_tokens[1].item()}) → 期望{response_tokens[2].item()}, 预测{predictions[0,4].item()} {'✓' if predictions[0,4] == response_tokens[2] else '✗'}")
        print(f"  R3({response_tokens[2].item()}) → 期望{response_tokens[3].item()}, 预测{predictions[0,5].item()} {'✓' if predictions[0,5] == response_tokens[3] else '✗'}")

        valid_mask = (labels != -100)
        correct = (predictions == labels) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()

        success = accuracy.item() > 0.95
        print(f"\n{'✅' if success else '❌'} Level 0 {'通过' if success else '失败'}: 准确率 = {accuracy.item():.4f}")
        return success


def test_level_n_with_eos(model, tokenizer, device, num_eos, level_name):
    """
    Level N: 加 N 个 EOS 的过拟合测试

    Args:
        num_eos: EOS 的数量（1, 2, 或 3）
        level_name: 测试名称
    """
    print("\n" + "=" * 100)
    print(f"{level_name}: 加 {num_eos} 个 EOS")
    print("=" * 100)

    # 构造样本
    prompt_len = 3
    block_size = 4
    mask_token_id = tokenizer.eos_token_id

    prompt_tokens = torch.tensor([1000, 1001, 1002], device=device)
    response_tokens = torch.tensor([2000, 2001, 2002, 2003], device=device)
    mask_tokens = torch.full((num_eos,), mask_token_id, device=device)

    # 构造交错序列: [P P P] [EOS × num_eos] [R1 R2 R3 R4]
    input_ids = torch.cat([prompt_tokens, mask_tokens, response_tokens], dim=0).unsqueeze(0)

    # Position IDs (EOS 和 block 共享位置)
    position_ids = torch.cat([
        torch.arange(0, prompt_len, device=device),
        torch.arange(prompt_len, prompt_len + num_eos, device=device),
        torch.arange(prompt_len, prompt_len + block_size, device=device),
    ], dim=0).unsqueeze(0)

    # Labels
    labels_list = []
    # Prompt (除了最后一个)
    labels_list.extend([-100] * (prompt_len - 1))
    # P[-1] → R1
    labels_list.append(response_tokens[0].item())
    # EOS_i → R(i+1)
    for i in range(num_eos):
        if i + 1 < len(response_tokens):
            labels_list.append(response_tokens[i + 1].item())
        else:
            labels_list.append(-100)
    # Real tokens AR预测
    for i in range(len(response_tokens)):
        if i + 1 < len(response_tokens):
            labels_list.append(response_tokens[i + 1].item())
        else:
            labels_list.append(-100)

    labels = torch.tensor(labels_list, device=device).unsqueeze(0)

    # 构造 BlockMask
    block_info = [('mask', num_eos), ('real', block_size)]
    batch = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'labels': labels,
        'block_info': [block_info],
        'prompt_len': [prompt_len],
        'seq_lens': [input_ids.shape[1]],
    }

    print(f"输入序列: {input_ids[0].tolist()}")
    print(f"Position: {position_ids[0].tolist()}")
    print(f"Labels:   {labels[0].tolist()}")

    # 训练
    optimizer = AdamW(model.parameters(), lr=1e-4)
    model.train()

    num_steps = 150
    for step in range(num_steps):
        optimizer.zero_grad()

        block_mask = create_block_mask_from_batch(batch, device)
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=block_mask,
        )
        logits = outputs.logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        loss.backward()
        optimizer.step()

        if (step + 1) % 30 == 0:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                valid_mask = (labels != -100)
                correct = (predictions == labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()
            print(f"  Step {step+1:3d}: Loss = {loss.item():.6f}, Accuracy = {accuracy.item():.4f}")

    # 验证 (保持 model.train() 以使用 FlexAttention)
    with torch.no_grad():
        block_mask = create_block_mask_from_batch(batch, device)
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=block_mask,  # 使用 FlexAttention BlockMask
        )
        logits = outputs.logits
        predictions = logits.argmax(dim=-1)

        print(f"\n最终预测验证:")
        # P[-1] → R1
        p_pos = prompt_len - 1
        print(f"  P[-1] → 期望{response_tokens[0].item()}, 预测{predictions[0, p_pos].item()} {'✓' if predictions[0, p_pos] == response_tokens[0] else '✗'}")

        # EOS_i → R(i+1)
        for i in range(num_eos):
            eos_pos = prompt_len + i
            expected = response_tokens[i + 1].item() if i + 1 < len(response_tokens) else -100
            pred = predictions[0, eos_pos].item()
            if expected != -100:
                print(f"  EOS_{i+1} → 期望{expected}, 预测{pred} {'✓' if pred == expected else '✗'}")

        valid_mask = (labels != -100)
        correct = (predictions == labels) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()

        success = accuracy.item() > 0.95
        print(f"\n{'✅' if success else '❌'} {level_name} {'通过' if success else '失败'}: 准确率 = {accuracy.item():.4f}")
        return success


def test_block_by_block_decoding(model, tokenizer, device):
    """
    Block-by-Block 快速解码验证

    验证模型能否通过添加 EOS 实现快速生成：
    Round 1: [P P P] [EOS×3] → 生成 [R1 R2 R3 R4]
    Round 2: [P P P] [R1 R2 R3 R4] [EOS×3] → 生成 [R5 R6 R7 R8]
    """
    print("\n" + "=" * 100)
    print("Block-by-Block 快速解码验证")
    print("=" * 100)

    prompt_len = 3
    block_size = 4
    num_eos = 3
    mask_token_id = tokenizer.eos_token_id

    prompt_tokens = torch.tensor([1000, 1001, 1002], device=device)

    # 目标：生成两个完整的 block
    block1 = torch.tensor([2000, 2001, 2002, 2003], device=device)
    block2 = torch.tensor([2010, 2011, 2012, 2013], device=device)

    all_response = torch.cat([block1, block2], dim=0)

    # ========== 训练阶段：过拟合完整序列 ==========
    print("\n训练阶段：过拟合到完整的交错序列...")

    mask_tokens = torch.full((num_eos,), mask_token_id, device=device)

    # 完整训练序列: [P P P] [EOS×3] [R1 R2 R3 R4] [EOS×3] [R5 R6 R7 R8]
    input_ids = torch.cat([
        prompt_tokens,
        mask_tokens, block1,
        mask_tokens, block2,
    ], dim=0).unsqueeze(0)

    # Position IDs
    position_ids = torch.cat([
        torch.arange(0, prompt_len, device=device),                      # [0, 1, 2]
        torch.arange(prompt_len, prompt_len + num_eos, device=device),  # [3, 4, 5]
        torch.arange(prompt_len, prompt_len + block_size, device=device), # [3, 4, 5, 6]
        torch.arange(prompt_len + block_size, prompt_len + block_size + num_eos, device=device), # [7, 8, 9]
        torch.arange(prompt_len + block_size, prompt_len + 2 * block_size, device=device), # [7, 8, 9, 10]
    ], dim=0).unsqueeze(0)

    # Labels
    labels_list = []
    # Prompt
    labels_list.extend([-100, -100])  # 前两个 prompt
    labels_list.append(all_response[0].item())  # P[-1] → R1
    # Mask 1
    for i in range(num_eos):
        labels_list.append(all_response[i + 1].item())  # EOS_i → R(i+1)
    # Block 1
    for i in range(block_size):
        if i + 1 < len(all_response):
            labels_list.append(all_response[i + 1].item())
        else:
            labels_list.append(-100)
    # Mask 2
    for i in range(num_eos):
        idx = block_size + i + 1
        if idx < len(all_response):
            labels_list.append(all_response[idx].item())
        else:
            labels_list.append(-100)
    # Block 2
    for i in range(block_size):
        idx = block_size + i + 1
        if idx < len(all_response):
            labels_list.append(all_response[idx].item())
        else:
            labels_list.append(-100)

    labels = torch.tensor(labels_list, device=device).unsqueeze(0)

    # BlockMask
    block_info = [('mask', num_eos), ('real', block_size), ('mask', num_eos), ('real', block_size)]
    batch = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'labels': labels,
        'block_info': [block_info],
        'prompt_len': [prompt_len],
        'seq_lens': [input_ids.shape[1]],
    }

    # 训练
    optimizer = AdamW(model.parameters(), lr=1e-4)
    model.train()

    num_steps = 200
    print(f"训练 {num_steps} 步...")
    for step in range(num_steps):
        optimizer.zero_grad()

        block_mask = create_block_mask_from_batch(batch, device)
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=block_mask,
        )
        logits = outputs.logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        loss.backward()
        optimizer.step()

        if (step + 1) % 40 == 0:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                valid_mask = (labels != -100)
                correct = (predictions == labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()
            print(f"  Step {step+1:3d}: Loss = {loss.item():.6f}, Accuracy = {accuracy.item():.4f}")

    print(f"✅ 训练完成")

    # ========== 推理阶段：Block-by-Block 生成 ==========
    print("\n" + "=" * 80)
    print("推理阶段：Block-by-Block 快速解码")
    print("=" * 80)

    # 保持 model.train() 模式以使用 FlexAttention
    generated_tokens = []

    # Round 1: 生成第一个 block
    print("\n【Round 1】生成第一个 block...")

    input_round1 = torch.cat([
        prompt_tokens,
        torch.full((num_eos,), mask_token_id, device=device),
    ], dim=0).unsqueeze(0)

    position_round1 = torch.cat([
        torch.arange(0, prompt_len, device=device),
        torch.arange(prompt_len, prompt_len + num_eos, device=device),
    ], dim=0).unsqueeze(0)

    block_info_round1 = [('mask', num_eos)]
    batch_round1 = {
        'input_ids': input_round1,
        'position_ids': position_round1,
        'block_info': [block_info_round1],
        'prompt_len': [prompt_len],
        'seq_lens': [input_round1.shape[1]],
    }

    with torch.no_grad():
        block_mask_round1 = create_block_mask_from_batch(batch_round1, device)
        outputs_round1 = model(
            input_ids=input_round1,
            position_ids=position_round1,
            attention_mask=block_mask_round1,
        )
        logits_round1 = outputs_round1.logits

        # 并行预测 4 个 tokens
        r1_pred = logits_round1[0, prompt_len - 1].argmax().item()  # P[-1] → R1
        r2_pred = logits_round1[0, prompt_len].argmax().item()      # EOS_1 → R2
        r3_pred = logits_round1[0, prompt_len + 1].argmax().item()  # EOS_2 → R3
        r4_pred = logits_round1[0, prompt_len + 2].argmax().item()  # EOS_3 → R4

        print(f"  P[-1] → 预测 {r1_pred}, 期望 {block1[0].item()} {'✓' if r1_pred == block1[0].item() else '✗'}")
        print(f"  EOS_1 → 预测 {r2_pred}, 期望 {block1[1].item()} {'✓' if r2_pred == block1[1].item() else '✗'}")
        print(f"  EOS_2 → 预测 {r3_pred}, 期望 {block1[2].item()} {'✓' if r3_pred == block1[2].item() else '✗'}")
        print(f"  EOS_3 → 预测 {r4_pred}, 期望 {block1[3].item()} {'✓' if r4_pred == block1[3].item() else '✗'}")

        block1_generated = torch.tensor([r1_pred, r2_pred, r3_pred, r4_pred], device=device)
        generated_tokens.extend([r1_pred, r2_pred, r3_pred, r4_pred])

        block1_correct = torch.equal(block1_generated, block1)
        print(f"\n  {'✅' if block1_correct else '❌'} Block 1 生成 {'正确' if block1_correct else '错误'}: {block1_generated.tolist()}")

    # Round 2: 生成第二个 block
    print("\n【Round 2】生成第二个 block...")

    input_round2 = torch.cat([
        prompt_tokens,
        block1_generated,  # 使用生成的 block 1
        torch.full((num_eos,), mask_token_id, device=device),
    ], dim=0).unsqueeze(0)

    position_round2 = torch.cat([
        torch.arange(0, prompt_len, device=device),
        torch.arange(prompt_len, prompt_len + block_size, device=device),
        torch.arange(prompt_len + block_size, prompt_len + block_size + num_eos, device=device),
    ], dim=0).unsqueeze(0)

    block_info_round2 = [('real', block_size), ('mask', num_eos)]
    batch_round2 = {
        'input_ids': input_round2,
        'position_ids': position_round2,
        'block_info': [block_info_round2],
        'prompt_len': [prompt_len],
        'seq_lens': [input_round2.shape[1]],
    }

    with torch.no_grad():
        block_mask_round2 = create_block_mask_from_batch(batch_round2, device)
        outputs_round2 = model(
            input_ids=input_round2,
            position_ids=position_round2,
            attention_mask=block_mask_round2,
        )
        logits_round2 = outputs_round2.logits

        # 并行预测 4 个 tokens
        r5_pred = logits_round2[0, prompt_len + block_size - 1].argmax().item()  # R4 → R5
        r6_pred = logits_round2[0, prompt_len + block_size].argmax().item()      # EOS_1 → R6
        r7_pred = logits_round2[0, prompt_len + block_size + 1].argmax().item()  # EOS_2 → R7
        r8_pred = logits_round2[0, prompt_len + block_size + 2].argmax().item()  # EOS_3 → R8

        print(f"  R4 → 预测 {r5_pred}, 期望 {block2[0].item()} {'✓' if r5_pred == block2[0].item() else '✗'}")
        print(f"  EOS_1 → 预测 {r6_pred}, 期望 {block2[1].item()} {'✓' if r6_pred == block2[1].item() else '✗'}")
        print(f"  EOS_2 → 预测 {r7_pred}, 期望 {block2[2].item()} {'✓' if r7_pred == block2[2].item() else '✗'}")
        print(f"  EOS_3 → 预测 {r8_pred}, 期望 {block2[3].item()} {'✓' if r8_pred == block2[3].item() else '✗'}")

        block2_generated = torch.tensor([r5_pred, r6_pred, r7_pred, r8_pred], device=device)
        generated_tokens.extend([r5_pred, r6_pred, r7_pred, r8_pred])

        block2_correct = torch.equal(block2_generated, block2)
        print(f"\n  {'✅' if block2_correct else '❌'} Block 2 生成 {'正确' if block2_correct else '错误'}: {block2_generated.tolist()}")

    # 最终验证
    print("\n" + "=" * 80)
    print("最终验证")
    print("=" * 80)

    generated_sequence = torch.tensor(generated_tokens, device=device)
    expected_sequence = torch.cat([block1, block2], dim=0)

    print(f"生成序列: {generated_sequence.tolist()}")
    print(f"期望序列: {expected_sequence.tolist()}")

    success = torch.equal(generated_sequence, expected_sequence)

    if success:
        print(f"\n✅ Block-by-Block 解码测试通过！")
        print(f"   - 只用了 2 次前向传播生成了 8 个 tokens")
        print(f"   - 加速比: 8 / 2 = 4x")
    else:
        print(f"\n❌ Block-by-Block 解码测试失败！")
        mismatches = (generated_sequence != expected_sequence).sum().item()
        print(f"   - 有 {mismatches} 个位置预测错误")

    return success


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("渐进式过拟合测试 + Block-by-Block 快速解码验证")
    print("=" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B"

    print(f"\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DLLMForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    print(f"✅ 模型加载成功（设备: {device}）")

    results = []

    # Level 0: 基础 AR
    results.append(("Level 0: 基础 AR", test_level_0_basic_ar(model, tokenizer, device)))

    # Level 1: 1 个 EOS
    results.append(("Level 1: 1个EOS", test_level_n_with_eos(model, tokenizer, device, 1, "Level 1")))

    # Level 2: 2 个 EOS
    results.append(("Level 2: 2个EOS", test_level_n_with_eos(model, tokenizer, device, 2, "Level 2")))

    # Level 3: 3 个 EOS
    results.append(("Level 3: 3个EOS", test_level_n_with_eos(model, tokenizer, device, 3, "Level 3")))

    # Block-by-Block 解码
    results.append(("Block-by-Block 解码", test_block_by_block_decoding(model, tokenizer, device)))

    # 总结
    print("\n" + "=" * 100)
    print("测试总结")
    print("=" * 100)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    if all_passed:
        print("\n✅ 所有测试通过！Next Block Prediction 完全正确！")
        print("=" * 100 + "\n")
        exit(0)
    else:
        print("\n❌ 部分测试失败！请检查模型或训练逻辑。")
        print("=" * 100 + "\n")
        exit(1)
