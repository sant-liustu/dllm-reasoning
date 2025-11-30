#!/bin/bash
# 使用 FSDP checkpoint 直接推理的便捷脚本

set -e

# ==================== 参数检查 ====================
if [ $# -lt 2 ]; then
    echo "用法: bash $0 <world_size> <checkpoint_dir> [prompt]"
    echo ""
    echo "参数:"
    echo "  world_size      训练时使用的 GPU 数量（必须与训练时一致）"
    echo "  checkpoint_dir  FSDP checkpoint 目录"
    echo "  prompt          可选的测试 prompt（默认: 'What is 2+2?'）"
    echo ""
    echo "示例:"
    echo "  bash $0 4 checkpoints/openr1_test_fixed/global_step_21000"
    echo "  bash $0 4 checkpoints/openr1_test_fixed/global_step_21000 'Solve: 25 * 4 = ?'"
    exit 1
fi

WORLD_SIZE=$1
CHECKPOINT_DIR=$2
PROMPT=${3:-"What is 2+2?"}

# ==================== 显示信息 ====================
echo "=========================================="
echo "FSDP Checkpoint 直接推理"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "World Size: $WORLD_SIZE"
echo "Prompt:     $PROMPT"
echo "=========================================="
echo ""

# ==================== 切换到项目根目录 ====================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"
echo ""

# ==================== 激活环境 ====================
if [ -f "$PROJECT_ROOT/activate_dllm_env.sh" ]; then
    echo "激活 dLLM 环境..."
    source "$PROJECT_ROOT/activate_dllm_env.sh"
    echo "环境已激活"
    echo ""
else
    echo "警告: 未找到 activate_dllm_env.sh，使用当前环境"
    echo ""
fi

# ==================== 运行推理 ====================
echo "开始推理..."
echo ""

torchrun --standalone --nnodes=1 --nproc_per_node=$WORLD_SIZE \
    -m dllm_reasoning.inference.fsdp_demo \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --prompt "$PROMPT" \
    --add_eos_length 7 \
    --refine_iter 2 \
    --max_new_tokens 512

echo ""
echo "=========================================="
echo "✅ 推理完成！"
echo "=========================================="
