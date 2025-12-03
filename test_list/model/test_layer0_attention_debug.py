#!/usr/bin/env python3
"""
Debug Layer 0 attention计算,检查中间变量是否严格对齐

重点检查:
1. Q, K, V 投影是否对齐 (在M₁位置)
2. Attention scores 是否对齐 (M₁对所有可见位置的scores)
3. Attention weights 是否对齐
4. Attention output 是否对齐

如果BlockMask工作正确,这些中间变量在不同配置下应该完全相同!
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dllm_reasoning.trainer.interleaved_sft_dataset import InterleavedSFTDataset
from dllm_reasoning.model.DLLM.modeling_dllm import DLLMForCausalLM
from dllm_reasoning.model.DLLM.configuration_dllm import DLLMConfig


def capture_layer0_attention_details(model, input_ids, position_ids, block_info, prompt_len, seq_len, device):
    """
    捕获Layer 0的所有attention中间变量

    Returns:
        dict: {
            'q': Q at M₁ position [num_heads, head_dim],
            'k': K for all positions [seq_len, num_kv_heads, head_dim],
            'v': V for all positions [seq_len, num_kv_heads, head_dim],
            'attn_scores': raw scores at M₁ [num_heads, seq_len],
            'attn_weights': softmax weights at M₁ [num_heads, seq_len],
            'attn_output': final output at M₁ [num_heads, head_dim],
            'layer_output': layer output at M₁ [hidden_dim],
        }
    """
    m1_pos = prompt_len
    results = {}

    # Hook Layer 0的self_attn
    layer0_attn = model.model.layers[0].self_attn

    # 保存中间变量
    intermediate = {}

    # Hook q_proj, k_proj, v_proj的输出
    def hook_q_proj(module, input, output):
        # output shape: [batch, seq_len, num_heads * head_dim]
        intermediate['q_proj_output'] = output.detach().clone()

    def hook_k_proj(module, input, output):
        intermediate['k_proj_output'] = output.detach().clone()

    def hook_v_proj(module, input, output):
        intermediate['v_proj_output'] = output.detach().clone()

    # Hook attention forward 来捕获scores和weights
    original_forward = layer0_attn.forward

    def hooked_forward(hidden_states, *args, **kwargs):
        # 调用原始forward
        output = original_forward(hidden_states, *args, **kwargs)

        # 手动重新计算来捕获中间变量
        bsz, q_len, _ = hidden_states.size()

        # 获取Q, K, V
        query_states = layer0_attn.q_proj(hidden_states)
        key_states = layer0_attn.k_proj(hidden_states)
        value_states = layer0_attn.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(bsz, q_len, layer0_attn.num_attention_heads, layer0_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, layer0_attn.num_key_value_heads, layer0_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, layer0_attn.num_key_value_heads, layer0_attn.head_dim).transpose(1, 2)

        # 保存Q, K, V (在M₁位置)
        intermediate['query_states'] = query_states[:, :, m1_pos, :].detach().clone()  # [batch, num_heads, head_dim]
        intermediate['key_states'] = key_states.detach().clone()  # [batch, num_kv_heads, seq_len, head_dim]
        intermediate['value_states'] = value_states.detach().clone()

        return output

    # 注册hooks
    h1 = layer0_attn.q_proj.register_forward_hook(hook_q_proj)
    h2 = layer0_attn.k_proj.register_forward_hook(hook_k_proj)
    h3 = layer0_attn.v_proj.register_forward_hook(hook_v_proj)

    # 替换forward
    layer0_attn.forward = hooked_forward

    # Hook Layer 0输出
    def hook_layer0_output(module, input, output):
        hidden_states = output[0]
        intermediate['layer_output'] = hidden_states[0, m1_pos].detach().clone()

    h4 = model.model.layers[0].register_forward_hook(hook_layer0_output)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            block_info=[block_info],
            prompt_len=[prompt_len],
            seq_lens=[seq_len],
            use_cache=False
        )

    # 恢复
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    layer0_attn.forward = original_forward

    # 整理结果
    results['q'] = intermediate.get('query_states', None)  # [batch, num_heads, head_dim]
    results['k'] = intermediate.get('key_states', None)    # [batch, num_kv_heads, seq_len, head_dim]
    results['v'] = intermediate.get('value_states', None)
    results['layer_output'] = intermediate.get('layer_output', None)

    return results


def test_config_attention(model, sample, num_blocks_to_keep, device):
    """测试特定配置下的Layer 0 attention"""
    input_ids = sample['input_ids'].to(device)
    position_ids = sample['position_ids'].to(device)
    block_info = sample['block_info']
    prompt_len = sample['prompt_len']

    # 截断序列
    current_pos = prompt_len
    blocks_seen = 0
    truncate_pos = None

    for seg_type, seg_idx, seg_len in block_info:
        blocks_seen += 1
        current_pos += seg_len
        if blocks_seen >= num_blocks_to_keep:
            truncate_pos = current_pos
            break

    if truncate_pos is None:
        truncate_pos = input_ids.size(0)

    # 截断
    input_ids_truncated = input_ids[:truncate_pos]
    position_ids_truncated = position_ids[:truncate_pos]

    block_info_truncated = []
    blocks_added = 0
    for seg_type, seg_idx, seg_len in block_info:
        if blocks_added >= num_blocks_to_keep:
            break
        block_info_truncated.append((seg_type, seg_idx, seg_len))
        blocks_added += 1

    # 捕获attention细节
    results = capture_layer0_attention_details(
        model=model,
        input_ids=input_ids_truncated,
        position_ids=position_ids_truncated,
        block_info=block_info_truncated,
        prompt_len=prompt_len,
        seq_len=truncate_pos,
        device=device,
    )

    return results


def compare_tensors(t1, t2, name, threshold=1e-6):
    """比较两个tensor"""
    if t1 is None or t2 is None:
        print(f"  ⚠️  {name}: 有一个为None")
        return False

    if t1.shape != t2.shape:
        print(f"  ❌ {name}: 形状不同 {t1.shape} vs {t2.shape}")
        return False

    diff = torch.norm(t1 - t2).item()
    max_diff = torch.max(torch.abs(t1 - t2)).item()

    is_same = diff < threshold

    if is_same:
        print(f"  ✅ {name}: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")
    else:
        print(f"  ❌ {name}: L2差异={diff:.2e}, 最大差异={max_diff:.2e}")

    return is_same


def main():
    MODEL_PATH = "dllm_reasoning/dllm_reasoning/models/DLLM-1.5B"
    DATA_PATH = "data/openr1.parquet"

    print("="*80)
    print("Layer 0 Attention 详细Debug")
    print("="*80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    config = DLLMConfig.from_pretrained(MODEL_PATH)
    model = DLLMForCausalLM(config)

    from transformers import AutoModelForCausalLM
    pretrained = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.load_state_dict(pretrained.state_dict())
    model = model.to(device).to(torch.bfloat16)
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
    print(f"  总序列长度: {sample['input_ids'].shape[0]}")
    print(f"  Prompt长度: {sample['prompt_len']}")
    print(f"  M₁位置: {sample['prompt_len']}")
    print()

    # 测试两个配置: [P][M₁] vs [P][M₁][R₁]
    test_configs = [
        (1, "[P][M₁]"),
        (2, "[P][M₁][R₁]"),
        (len(sample['block_info']), f"完整序列({len(sample['block_info'])}块)"),
    ]

    print("="*80)
    print("捕获Layer 0 Attention中间变量")
    print("="*80)
    print()

    all_results = {}

    for num_blocks, config_name in test_configs:
        print(f"配置: {config_name}")
        results = test_config_attention(
            model=model,
            sample=sample,
            num_blocks_to_keep=num_blocks,
            device=device,
        )
        all_results[config_name] = results

        # 打印形状信息
        if results['q'] is not None:
            print(f"  Q shape at M₁: {results['q'].shape}")
        if results['k'] is not None:
            print(f"  K shape (全序列): {results['k'].shape}")
        if results['layer_output'] is not None:
            print(f"  Layer output shape at M₁: {results['layer_output'].shape}")
        print()

    # 对比
    print("="*80)
    print("对比分析")
    print("="*80)
    print()

    config_names = list(all_results.keys())
    base_config = config_names[0]

    for i, compare_config in enumerate(config_names[1:], 1):
        print(f"\n【对比 {i}】基准: {base_config} vs 对比: {compare_config}")
        print("-" * 60)

        base_results = all_results[base_config]
        compare_results = all_results[compare_config]

        # 比较Q (在M₁位置,应该完全相同)
        print("\n1. Query States at M₁:")
        if base_results['q'] is not None and compare_results['q'] is not None:
            # Q shape: [batch, num_heads, head_dim]
            # 只比较前面可见部分(M₁应该看到相同的内容)
            compare_tensors(base_results['q'], compare_results['q'], "Q at M₁")

        # 比较K (前278个位置应该完全相同)
        print("\n2. Key States (前278个位置,M₁可见范围):")
        if base_results['k'] is not None and compare_results['k'] is not None:
            # K shape: [batch, num_kv_heads, seq_len, head_dim]
            m1_visible_len = sample['prompt_len'] + 1  # M₁可见: [0, prompt_len]
            k_base_visible = base_results['k'][:, :, :m1_visible_len, :]
            k_compare_visible = compare_results['k'][:, :, :m1_visible_len, :]
            compare_tensors(k_base_visible, k_compare_visible, f"K[:, :, :278, :] (M₁可见范围)")

        # 比较V (前278个位置应该完全相同)
        print("\n3. Value States (前278个位置,M₁可见范围):")
        if base_results['v'] is not None and compare_results['v'] is not None:
            m1_visible_len = sample['prompt_len'] + 1
            v_base_visible = base_results['v'][:, :, :m1_visible_len, :]
            v_compare_visible = compare_results['v'][:, :, :m1_visible_len, :]
            compare_tensors(v_base_visible, v_compare_visible, f"V[:, :, :278, :] (M₁可见范围)")

        # 比较最终Layer输出
        print("\n4. Layer 0 Output at M₁:")
        compare_tensors(base_results['layer_output'], compare_results['layer_output'], "Layer 0 output")

        print()

    print("="*80)
    print("分析总结")
    print("="*80)
    print()
    print("如果Q相同但K/V在可见范围内不同,说明:")
    print("  - 可能是position encoding的问题")
    print("  - 或者是K/V投影受到了序列长度的影响")
    print()
    print("如果Q/K/V在可见范围内都相同,但layer output不同,说明:")
    print("  - 可能是attention计算时受到了mask外部分的影响")
    print("  - 或者是FlexAttention的实现问题")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
