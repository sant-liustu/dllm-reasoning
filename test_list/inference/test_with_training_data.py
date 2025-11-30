#!/usr/bin/env python3
"""
使用训练数据验证 Interleaved 推理的正确性

目标:
1. 读取一条训练数据 (prompt + target)
2. 使用 prompt + mask 预测第一个 block
3. 对比预测结果和真实 target
4. 逐个 block 验证，看模型是否真的学会了 Next Block Prediction
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 配置
MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft/global_step_17172/huggingface"
DATA_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"
BLOCK_SIZE = 4  # 训练时的 block_size
NUM_MASKS = 3   # block_size - 1

print("="*100)
print("🔬 使用训练数据验证 Interleaved 推理")
print("="*100)

# ========== 加载模型 ==========
print(f"\n📦 加载模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).to(device).train()  # 使用 train() 模式以启用 FlexAttention

print(f"   ✅ 模型加载完成 (training mode for FlexAttention)")

# 获取特殊 token
eos_token_id = tokenizer.eos_token_id
mask_token_id = eos_token_id  # 训练时使用 eos 作为 mask

print(f"   Mask token ID: {mask_token_id}")
print(f"   EOS token ID: {eos_token_id}")

# ========== 加载训练数据 ==========
print(f"\n📂 加载训练数据: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"   总样本数: {len(df)}")

# 取第一条数据
sample = df.iloc[0]
prompt = sample['prompt']
target = sample['target']

print(f"\n📝 测试样本 0:")
print(f"   Prompt: {prompt}")
print(f"   Target: {target}")

# ========== Tokenize ==========
print(f"\n🔤 Tokenization...")

# Tokenize prompt (使用 apply_chat_template)
prompt_messages = list(prompt) if isinstance(prompt, (list, tuple)) else prompt
if hasattr(prompt, 'tolist'):
    prompt_messages = prompt.tolist()

input_ids = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

prompt_len = input_ids.size(1)
print(f"   Prompt 长度: {prompt_len} tokens")
print(f"   Prompt tokens (前20个): {input_ids[0][:20].tolist()}")

# Tokenize target (完整的 response)
# 提取 assistant 的回复内容
target_list = target.tolist() if hasattr(target, 'tolist') else target
if isinstance(target_list, list) and len(target_list) > 0:
    target_content = target_list[0].get('content', '') if isinstance(target_list[0], dict) else str(target_list[0])
else:
    target_content = str(target_list)

print(f"   Target content (前100字符): {target_content[:100]}...")

# ========== 重要: 和训练时保持一致，去掉 <think> 标签 ==========
# 训练代码: interleaved_sft_dataset.py line 507-509
if target_content.strip().startswith('<think>'):
    think_start = target_content.find('<think>')
    target_content = target_content[think_start + 7:].lstrip()  # 去掉 '<think>' 这7个字符
    print(f"   ⚠️  检测到 <think> 标签，已去除")
    print(f"   处理后 target (前100字符): {target_content[:100]}...")

# Tokenize target content (不使用 chat template，直接 tokenize)
target_tokens = tokenizer(target_content, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
target_len = target_tokens.size(0)

print(f"   Target 长度: {target_len} tokens")
print(f"   Target tokens (前20个): {target_tokens[:20].tolist()}")

# ========== 逐 Block 验证 ==========
print(f"\n{'='*100}")
print(f"🧪 开始逐 Block 验证 (block_size={BLOCK_SIZE}, num_masks={NUM_MASKS})")
print(f"{'='*100}")

num_blocks = min((target_len + BLOCK_SIZE - 1) // BLOCK_SIZE, 10)  # 最多测试10个block
print(f"   将测试前 {num_blocks} 个 blocks")

generated_ids = input_ids.clone()  # 从 prompt 开始
all_correct = True

for block_idx in range(num_blocks):
    print(f"\n{'-'*100}")
    print(f"📦 Block {block_idx + 1}/{num_blocks}")
    print(f"{'-'*100}")

    # 当前已生成长度
    pre_length = generated_ids.size(1)
    print(f"   当前序列长度: {pre_length} tokens")

    # 真实的这个 block 的 token
    block_start = block_idx * BLOCK_SIZE
    block_end = min((block_idx + 1) * BLOCK_SIZE, target_len)
    true_block = target_tokens[block_start:block_end]
    actual_block_size = true_block.size(0)

    print(f"   真实 block 范围: [{block_start}:{block_end}]")
    print(f"   真实 block 大小: {actual_block_size} tokens")
    print(f"   真实 block tokens: {true_block.tolist()}")
    print(f"   真实 block 文本: {repr(tokenizer.decode(true_block))}")

    # 添加 mask tokens
    mask_block = torch.full(
        (1, NUM_MASKS),
        mask_token_id,
        device=device,
        dtype=torch.long,
    )
    current_ids = torch.cat([generated_ids, mask_block], dim=1)

    # 构造 position_ids (连续的)
    position_ids = torch.arange(
        0, pre_length + NUM_MASKS,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    print(f"\n   添加 {NUM_MASKS} 个 mask 后:")
    print(f"   - 序列长度: {current_ids.size(1)}")
    print(f"   - Position IDs (最后10个): {position_ids[0, -10:].tolist()}")

    # ========== 创建 FlexAttention BlockMask (和训练时一致) ==========
    # 训练时使用 FlexAttention，推理时也必须使用，否则 attention pattern 不一致
    from torch.nn.attention.flex_attention import create_block_mask

    # 构造 mask (和训练时保持一致):
    # - Prompt tokens: 看之前的所有 Prompt (causal)
    # - Mask tokens: 看所有 Prompt + 同一 group 内之前的 Mask (causal)
    # 必须使用纯 Tensor 操作，不能用 Python if (vmap 限制)
    def mask_fn(b, h, q_idx, kv_idx):
        # q_idx, kv_idx 是位置索引
        # Prompt 部分: [0, pre_length)
        # Mask 部分: [pre_length, pre_length + NUM_MASKS)

        q_is_prompt = q_idx < pre_length
        kv_is_prompt = kv_idx < pre_length

        # 1. Prompt 看 Prompt (causal)
        prompt_see_prompt = q_is_prompt & kv_is_prompt & (kv_idx <= q_idx)

        # 2. Mask 看 Prompt (全连接)
        mask_see_prompt = (~q_is_prompt) & kv_is_prompt

        # 3. Mask 看 Mask (causal within same group)
        # 同一个 mask group 内，causal attention
        q_is_mask = ~q_is_prompt
        kv_is_mask = ~kv_is_prompt
        mask_see_mask = q_is_mask & kv_is_mask & (kv_idx <= q_idx)

        # 4. Prompt 不能看 Mask: q_is_prompt & (~kv_is_prompt) → False (不包含)

        return prompt_see_prompt | mask_see_prompt | mask_see_mask

    block_mask = create_block_mask(
        mask_fn,
        B=1,
        H=None,
        Q_LEN=current_ids.size(1),
        KV_LEN=current_ids.size(1),
        device=device
    )

    # 前向传播 (使用 FlexAttention)
    with torch.no_grad():
        outputs = model(
            current_ids,
            position_ids=position_ids,
            attention_mask=block_mask,  # 使用 FlexAttention BlockMask
            use_cache=False
        )
        logits = outputs.logits

    # 提取预测位置的 logits
    # 我们要预测位置 [pre_length, pre_length+1, ..., pre_length+NUM_MASKS]
    # 需要 logits[pre_length-1 : pre_length+NUM_MASKS]
    pred_logits = logits[0, pre_length-1 : pre_length+NUM_MASKS, :]

    print(f"\n   提取 logits:")
    print(f"   - Logits 索引范围: [{pre_length-1}:{pre_length+NUM_MASKS}]")
    print(f"   - 预测位置范围: [{pre_length}:{pre_length+NUM_MASKS+1}]")
    print(f"   - 预测 token 数量: {pred_logits.size(0)}")

    # === DEBUG: 详细分析logits ===
    print(f"\n   === DEBUG: Logits详细分析 ===")
    for i in range(min(pred_logits.size(0), 4)):  # 只看前4个
        pos = pre_length - 1 + i
        print(f"\n   Position {pos} 预测 position {pos+1}:")

        # Top 5预测
        top5_logits, top5_indices = pred_logits[i].topk(5)
        print(f"     Top 5 predictions:")
        for j, (logit_val, token_id) in enumerate(zip(top5_logits, top5_indices)):
            token_text = tokenizer.decode([token_id.item()])
            print(f"       #{j+1}: ID={token_id.item():6d}, logit={logit_val.item():8.4f}, text={repr(token_text)}")

        # 真实token的logit
        if i < actual_block_size:
            true_token_id = true_block[i].item()
            true_logit = pred_logits[i, true_token_id].item()
            true_token_text = tokenizer.decode([true_token_id])
            print(f"     真实 token: ID={true_token_id:6d}, logit={true_logit:8.4f}, text={repr(true_token_text)}")

            # 计算真实token在所有token中的排名
            all_logits_sorted = pred_logits[i].sort(descending=True)
            true_rank = (all_logits_sorted.values > true_logit).sum().item() + 1
            print(f"     真实 token 排名: {true_rank}/{pred_logits.size(1)}")

    print(f"   === END DEBUG ===\n")

    # 解码 (贪婪)
    predicted_tokens = pred_logits.argmax(dim=-1)

    # 只取前 actual_block_size 个 (因为最后一个 block 可能不满)
    predicted_block = predicted_tokens[:actual_block_size]

    print(f"\n   预测结果:")
    print(f"   - 预测 block tokens: {predicted_block.tolist()}")
    print(f"   - 预测 block 文本: {repr(tokenizer.decode(predicted_block))}")

    # 对比
    is_correct = torch.equal(predicted_block.cpu(), true_block.cpu())

    if is_correct:
        print(f"\n   ✅ Block {block_idx + 1} 预测完全正确！")
    else:
        print(f"\n   ❌ Block {block_idx + 1} 预测错误")

        # 逐个 token 对比
        print(f"\n   逐 token 对比:")
        for i in range(actual_block_size):
            pred_tok = predicted_block[i].item()
            true_tok = true_block[i].item()
            pred_text = tokenizer.decode([pred_tok])
            true_text = tokenizer.decode([true_tok])

            if pred_tok == true_tok:
                print(f"      位置 {block_start + i}: ✅ {repr(pred_text)} (ID: {pred_tok})")
            else:
                print(f"      位置 {block_start + i}: ❌ 预测 {repr(pred_text)} (ID: {pred_tok}), 真实 {repr(true_text)} (ID: {true_tok})")

        all_correct = False
        print(f"\n   ⚠️  停止验证，因为第 {block_idx + 1} 个 block 预测错误")
        break

    # 更新序列 (使用真实的 token，继续下一个 block)
    generated_ids = torch.cat([generated_ids, true_block.unsqueeze(0).to(device)], dim=1)

print(f"\n{'='*100}")
if all_correct:
    print(f"🎉 所有 {num_blocks} 个 blocks 都预测正确！")
    print(f"✅ 模型确实学会了 Next Block Prediction！")
else:
    print(f"⚠️  验证失败，模型在某些 block 上预测错误")
    print(f"💡 可能的原因:")
    print(f"   1. Position IDs 设置不对")
    print(f"   2. Logits 提取位置不对")
    print(f"   3. 训练数据处理方式和推理不一致")
    print(f"   4. 其他未知问题")
print(f"{'='*100}")
