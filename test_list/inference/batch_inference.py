#!/usr/bin/env python3
"""
批量推理脚本：测试模型在训练数据上的拟合能力

✅ 已对齐训练时的数据处理方式（apply_chat_template）

用途：
1. 读取训练数据（parquet 格式，chat messages）
2. 使用 apply_chat_template 处理 prompt（与训练一致）
3. 对每个 prompt 进行推理
4. 保存 prompt + inference 结果
5. 可与原始 target 对比，验证模型是否正确学习

使用方法：
    python scripts/batch_inference.py \
        --model_path checkpoints/my_exp/global_step_1000/huggingface \
        --data_file data/train.parquet \
        --output_file results/train_inference.jsonl \
        --num_samples 100 \
        --batch_size 4

注意：
- prompt_key 和 target_key 应为 chat messages 格式的列
- tokenization 方式与训练脚本 (sft_dataset.py) 完全对齐
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dllm_reasoning.inference.generator import iterative_generate


def load_data(data_file, prompt_key, target_key, num_samples=None):
    """
    加载数据文件（对齐训练时的数据处理方式）

    Args:
        data_file: parquet 文件路径
        prompt_key: prompt 列名（chat messages 格式）
        target_key: target 列名（chat messages 格式）
        num_samples: 采样数量（None=全部）

    Returns:
        list of dict: [{"prompt_messages": ..., "target_content": ...}, ...]
        - prompt_messages: 原始 chat messages（待 apply_chat_template）
        - target_content: assistant 回复的文本内容（用于对比）
    """
    print(f"📂 加载数据: {data_file}")

    # 读取 parquet 文件
    df = pd.read_parquet(data_file)

    print(f"   总样本数: {len(df)}")

    # 检查列名是否存在
    if prompt_key not in df.columns:
        raise ValueError(f"数据中没有找到 '{prompt_key}' 列。可用列: {df.columns.tolist()}")
    if target_key not in df.columns:
        raise ValueError(f"数据中没有找到 '{target_key}' 列。可用列: {df.columns.tolist()}")

    # 采样
    if num_samples is not None and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)
        print(f"   采样数量: {num_samples}")

    # 转换为 list of dict
    data = []
    for _, row in df.iterrows():
        prompt_val = row[prompt_key]
        target_val = row[target_key]

        # 处理 prompt: 保留原始 chat messages 格式
        if hasattr(prompt_val, 'tolist'):  # numpy array
            prompt_messages = prompt_val.tolist()
        elif isinstance(prompt_val, (list, tuple)):
            prompt_messages = list(prompt_val)
        else:
            # 如果不是消息格式,创建一个简单的 user 消息
            prompt_messages = [{"role": "user", "content": str(prompt_val)}]

        # 处理 target: 提取 assistant 回复的文本内容
        if hasattr(target_val, 'tolist'):  # numpy array
            target_list = target_val.tolist()
            if isinstance(target_list, list) and len(target_list) > 0 and isinstance(target_list[0], dict):
                # 提取 assistant 消息的 content
                target_content = '\n'.join([msg.get('content', '') for msg in target_list if msg.get('role') == 'assistant'])
            else:
                target_content = str(target_list)
        elif isinstance(target_val, (list, tuple)):
            target_content = '\n'.join([msg.get('content', '') for msg in target_val if msg.get('role') == 'assistant'])
        else:
            target_content = str(target_val)

        data.append({
            "prompt_messages": prompt_messages,  # 原始消息格式
            "target_content": target_content      # 文本内容
        })

    return data


def batch_inference(model, tokenizer, prompt_messages_list, batch_size,
                    add_eos_length=127, refine_iter=2, max_new_tokens=1024):
    """
    批量推理（对齐训练时的 tokenization 方式）

    Args:
        model: 模型
        tokenizer: tokenizer
        prompt_messages_list: list of chat messages（每个元素是一个消息列表）
        batch_size: 批大小
        add_eos_length: 每块添加的 EOS 数量
        refine_iter: refine 轮数
        max_new_tokens: 最大生成 token 数

    Returns:
        list of str: 生成的结果
    """
    responses = []

    # 分批处理
    import sys
    for i in tqdm(range(0, len(prompt_messages_list), batch_size), desc="推理进度"):
        batch_prompt_messages = prompt_messages_list[i:i+batch_size]

        print(f"\n[Batch {i//batch_size + 1}/{(len(prompt_messages_list) + batch_size - 1)//batch_size}] 处理样本 {i} ~ {min(i+batch_size, len(prompt_messages_list))-1}")
        sys.stdout.flush()

        # 准备输入 - 使用 apply_chat_template（对齐训练）
        batch_inputs = []
        for messages in batch_prompt_messages:
            # 使用 apply_chat_template，添加 generation prompt（对齐训练 sft_dataset.py:169）
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            # squeeze 去掉 batch 维度 (1, seq_len) -> (seq_len,)
            batch_inputs.append(input_ids.squeeze(0))

        # Padding
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence(
            batch_inputs,
            batch_first=True,
            padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        ).to(model.device)

        print(f"  输入序列长度: {[len(inp) for inp in batch_inputs]}")

        # 🔍 DEBUG: 打印第一个样本的输入解码结果
        if i == 0:
            print(f"\n  🔍 DEBUG - 样本 {i} 的输入 token 解码:")
            print(f"  " + "="*76)
            input_decoded = tokenizer.decode(batch_inputs[0], skip_special_tokens=False)
            print(f"  {input_decoded}")
            print(f"  " + "="*76)
            print(f"  输入最后100个字符: ...{input_decoded[-100:]}")
            print()

        print(f"  开始生成 (max_new_tokens={max_new_tokens})...")
        sys.stdout.flush()

        # 生成
        with torch.no_grad():
            output_ids = iterative_generate(
                model=model,
                input_ids=input_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                add_eos_length=add_eos_length,
                refine_iter=refine_iter,
                max_new_tokens=max_new_tokens,
                max_length=8192,
                verbose_trace=False,
            )

        print(f"  ✅ 生成完成，输出长度: {output_ids.shape[1]}")
        sys.stdout.flush()

        # 解码
        for j in range(len(batch_prompt_messages)):
            # 只解码生成的部分（去掉 prompt）
            response = tokenizer.decode(
                output_ids[j, input_ids[j].size(0):],
                skip_special_tokens=True
            )
            responses.append(response)
            print(f"  样本 {i+j}: 生成了 {len(response)} 个字符")

            # 🔍 DEBUG: 打印第一个样本的完整输出解码
            if i == 0 and j == 0:
                print(f"\n  🔍 DEBUG - 样本 {i+j} 的完整输出 token 解码:")
                print(f"  " + "="*76)
                full_output = tokenizer.decode(output_ids[j], skip_special_tokens=False)
                print(f"  {full_output[:500]}...")
                print(f"  " + "="*76)
                print(f"\n  🔍 DEBUG - 仅生成部分 (不含输入):")
                print(f"  " + "="*76)
                print(f"  {response[:500]}...")
                print(f"  " + "="*76)
                print()

            sys.stdout.flush()

    return responses


def save_results(results, output_file):
    """
    保存结果到 JSONL 文件

    Args:
        results: list of dict
        output_file: 输出文件路径
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 保存结果到: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"   ✅ 已保存 {len(results)} 条结果")


def compute_metrics(results):
    """
    计算简单的评估指标

    Args:
        results: list of dict with keys: prompt, target, prediction

    Returns:
        dict: 评估指标
    """
    print("\n" + "="*80)
    print("📊 评估指标")
    print("="*80)

    total = len(results)

    # 完全匹配率（严格）
    exact_match = sum(1 for r in results if r['prediction'].strip() == r['target'].strip())
    exact_match_rate = exact_match / total * 100

    # 包含匹配率（宽松）
    contain_match = sum(1 for r in results if r['target'].strip() in r['prediction'])
    contain_match_rate = contain_match / total * 100

    # 平均长度
    avg_target_len = sum(len(r['target']) for r in results) / total
    avg_pred_len = sum(len(r['prediction']) for r in results) / total

    metrics = {
        "total_samples": total,
        "exact_match": exact_match,
        "exact_match_rate": f"{exact_match_rate:.2f}%",
        "contain_match": contain_match,
        "contain_match_rate": f"{contain_match_rate:.2f}%",
        "avg_target_length": f"{avg_target_len:.1f}",
        "avg_prediction_length": f"{avg_pred_len:.1f}",
    }

    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("="*80)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="批量推理脚本")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型检查点路径")

    # 数据参数
    parser.add_argument("--data_file", type=str, required=True,
                       help="数据文件路径（parquet 格式）")
    parser.add_argument("--prompt_key", type=str, default="prompt",
                       help="prompt 列名（默认: prompt）")
    parser.add_argument("--target_key", type=str, default="target",
                       help="target 列名（默认: target）")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="采样数量（默认: 全部）")

    # 推理参数
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批大小（默认: 4）")
    parser.add_argument("--add_eos_length", type=int, default=127,
                       help="每块添加的 EOS 数量（默认: 127）")
    parser.add_argument("--refine_iter", type=int, default=2,
                       help="refine 轮数（默认: 2）")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                       help="最大生成 token 数（默认: 1024）")

    # 输出参数
    parser.add_argument("--output_file", type=str,
                       default="results/batch_inference.jsonl",
                       help="输出文件路径（默认: results/batch_inference.jsonl）")
    parser.add_argument("--save_metrics", action="store_true",
                       help="是否保存评估指标到单独的文件")

    args = parser.parse_args()

    from datetime import datetime
    start_time = datetime.now()

    print("="*80)
    print("🚀 批量推理脚本")
    print("="*80)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型路径: {args.model_path}")
    print(f"数据文件: {args.data_file}")
    print(f"输出文件: {args.output_file}")
    print(f"批大小: {args.batch_size}")
    print(f"推理参数: add_eos_length={args.add_eos_length}, refine_iter={args.refine_iter}, max_new_tokens={args.max_new_tokens}")
    print("="*80)
    print()

    # 1. 加载数据
    data = load_data(
        args.data_file,
        args.prompt_key,
        args.target_key,
        args.num_samples
    )
    print()

    # 2. 加载模型
    print("🔧 加载模型...")
    import os

    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"   可用GPU数量: {num_gpus}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   主设备: {device}")
    print(f"   模型路径: {args.model_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型路径不存在: {args.model_path}")
    print(f"   ✅ 模型路径确认存在")

    print(f"   正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"   ✅ Tokenizer 加载完成")

    print(f"   正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto" if num_gpus > 1 else None,  # 使用 device_map="auto" 启用多GPU
    )

    if num_gpus <= 1:
        model = model.to(device)

    model.eval()

    print(f"   ✅ 模型加载完成")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    if num_gpus > 1:
        print(f"   💡 使用多GPU加载 (device_map='auto')，模型已自动分布到 {num_gpus} 张卡")
        # 打印每张卡的显存使用
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            print(f"      GPU {i}: {allocated:.2f} GB")
    print()

    # 3. 批量推理
    print("🔮 开始推理...")
    prompt_messages_list = [item["prompt_messages"] for item in data]
    predictions = batch_inference(
        model=model,
        tokenizer=tokenizer,
        prompt_messages_list=prompt_messages_list,
        batch_size=args.batch_size,
        add_eos_length=args.add_eos_length,
        refine_iter=args.refine_iter,
        max_new_tokens=args.max_new_tokens,
    )
    print()

    # 4. 整合结果
    results = []
    for i, item in enumerate(data):
        # 将 prompt_messages 转成可读的文本（用于显示）
        prompt_text = '\n'.join([
            f"{msg['role']}: {msg['content']}"
            for msg in item["prompt_messages"]
        ])

        results.append({
            "index": i,
            "prompt": prompt_text,
            "target": item["target_content"],
            "prediction": predictions[i],
        })

    # 5. 保存结果
    save_results(results, args.output_file)
    print()

    # 6. 计算评估指标
    metrics = compute_metrics(results)

    if args.save_metrics:
        metrics_file = args.output_file.replace(".jsonl", "_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n💾 评估指标已保存到: {metrics_file}")

    # 7. 显示几个示例
    print("\n" + "="*80)
    print("📋 示例结果（前3条）")
    print("="*80)
    for i, result in enumerate(results[:3]):
        print(f"\n【样本 {i+1}】")
        print(f"Prompt: {result['prompt'][:100]}...")
        print(f"Target: {result['target'][:100]}...")
        print(f"Prediction: {result['prediction'][:100]}...")
        print("-"*80)

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*80)
    print("✅ 批量推理完成！")
    print("="*80)
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print(f"结果文件: {args.output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
