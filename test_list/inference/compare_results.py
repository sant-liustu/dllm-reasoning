#!/usr/bin/env python3
"""
对比推理结果和原始 target

用途：
1. 读取批量推理的结果文件（jsonl）
2. 逐条对比 prediction 和 target
3. 生成详细的对比报告

使用方法：
    python scripts/compare_results.py \
        --results_file results/batch_inference.jsonl \
        --output_file results/comparison_report.txt
"""

import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher


def load_results(results_file):
    """加载推理结果"""
    print(f"📂 加载结果文件: {results_file}")

    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"   总样本数: {len(results)}")
    return results


def calculate_similarity(str1, str2):
    """计算两个字符串的相似度（0-1）"""
    return SequenceMatcher(None, str1, str2).ratio()


def analyze_results(results, similarity_threshold=0.8):
    """分析结果"""
    print("\n" + "="*80)
    print("📊 详细分析")
    print("="*80)

    total = len(results)
    exact_matches = []
    high_similarity = []
    low_similarity = []

    for result in results:
        target = result['target'].strip()
        prediction = result['prediction'].strip()
        similarity = calculate_similarity(target, prediction)
        result['similarity'] = similarity

        if target == prediction:
            exact_matches.append(result)
        elif similarity >= similarity_threshold:
            high_similarity.append(result)
        else:
            low_similarity.append(result)

    # 打印统计信息
    print(f"\n✅ 完全匹配: {len(exact_matches)} / {total} ({len(exact_matches)/total*100:.2f}%)")
    print(f"🟢 高相似度 (>={similarity_threshold}): {len(high_similarity)} / {total} ({len(high_similarity)/total*100:.2f}%)")
    print(f"🔴 低相似度 (<{similarity_threshold}): {len(low_similarity)} / {total} ({len(low_similarity)/total*100:.2f}%)")

    # 计算平均相似度
    avg_similarity = sum(r['similarity'] for r in results) / total
    print(f"\n📈 平均相似度: {avg_similarity:.4f}")

    return {
        'exact_matches': exact_matches,
        'high_similarity': high_similarity,
        'low_similarity': low_similarity,
        'avg_similarity': avg_similarity
    }


def generate_report(results, analysis, output_file, max_display=10):
    """生成详细对比报告"""
    print(f"\n💾 生成对比报告: {output_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        # 标题
        f.write("="*80 + "\n")
        f.write("批量推理结果对比报告\n")
        f.write("="*80 + "\n\n")

        # 统计信息
        total = len(results)
        f.write("统计信息\n")
        f.write("-"*80 + "\n")
        f.write(f"总样本数: {total}\n")
        f.write(f"完全匹配: {len(analysis['exact_matches'])} ({len(analysis['exact_matches'])/total*100:.2f}%)\n")
        f.write(f"高相似度: {len(analysis['high_similarity'])} ({len(analysis['high_similarity'])/total*100:.2f}%)\n")
        f.write(f"低相似度: {len(analysis['low_similarity'])} ({len(analysis['low_similarity'])/total*100:.2f}%)\n")
        f.write(f"平均相似度: {analysis['avg_similarity']:.4f}\n")
        f.write("\n")

        # 完全匹配的样本（显示前几个）
        f.write("="*80 + "\n")
        f.write(f"✅ 完全匹配的样本（前 {min(max_display, len(analysis['exact_matches']))} 个）\n")
        f.write("="*80 + "\n\n")

        for i, result in enumerate(analysis['exact_matches'][:max_display]):
            f.write(f"【样本 {result['index']}】\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Target: {result['target']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"相似度: {result['similarity']:.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")

        # 高相似度但不完全匹配的样本
        f.write("="*80 + "\n")
        f.write(f"🟢 高相似度样本（前 {min(max_display, len(analysis['high_similarity']))} 个）\n")
        f.write("="*80 + "\n\n")

        for i, result in enumerate(analysis['high_similarity'][:max_display]):
            f.write(f"【样本 {result['index']}】\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Target: {result['target']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"相似度: {result['similarity']:.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")

        # 低相似度的样本（需要重点关注）
        f.write("="*80 + "\n")
        f.write(f"🔴 低相似度样本（前 {min(max_display, len(analysis['low_similarity']))} 个）⚠️ 需要重点关注\n")
        f.write("="*80 + "\n\n")

        for i, result in enumerate(analysis['low_similarity'][:max_display]):
            f.write(f"【样本 {result['index']}】\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Target: {result['target']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"相似度: {result['similarity']:.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")

        # 相似度分布
        f.write("="*80 + "\n")
        f.write("📊 相似度分布\n")
        f.write("="*80 + "\n\n")

        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(bins)-1):
            count = sum(1 for r in results if bins[i] <= r['similarity'] < bins[i+1])
            if i == len(bins)-2:  # 最后一个区间包含 1.0
                count = sum(1 for r in results if bins[i] <= r['similarity'] <= bins[i+1])
            f.write(f"[{bins[i]:.1f}, {bins[i+1]:.1f}]: {count} ({count/total*100:.2f}%)\n")

    print(f"   ✅ 报告已保存")


def main():
    parser = argparse.ArgumentParser(description="对比推理结果和原始 target")

    parser.add_argument("--results_file", type=str, required=True,
                       help="推理结果文件（jsonl 格式）")
    parser.add_argument("--output_file", type=str,
                       default="results/comparison_report.txt",
                       help="输出报告文件路径")
    parser.add_argument("--similarity_threshold", type=float, default=0.8,
                       help="高相似度阈值（默认: 0.8）")
    parser.add_argument("--max_display", type=int, default=10,
                       help="每个类别最多显示的样本数（默认: 10）")

    args = parser.parse_args()

    print("="*80)
    print("🔍 结果对比分析")
    print("="*80)

    # 1. 加载结果
    results = load_results(args.results_file)

    # 2. 分析结果
    analysis = analyze_results(results, args.similarity_threshold)

    # 3. 生成报告
    generate_report(results, analysis, args.output_file, args.max_display)

    print("\n" + "="*80)
    print("✅ 分析完成！")
    print(f"   报告文件: {args.output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
