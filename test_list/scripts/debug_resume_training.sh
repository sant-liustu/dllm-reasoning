#!/bin/bash
# Debug训练脚本 - 从checkpoint恢复并运行少量步骤以观察详细日志
#
# 使用方法:
#   bash dllm_reasoning/test_list/scripts/debug_resume_training.sh <num_gpus> <checkpoint_path> [log_file]
#
# 示例:
#   bash dllm_reasoning/test_list/scripts/debug_resume_training.sh 4 dllm_reasoning/checkpoints/interleaved_sft/global_step_17172 debug_resume.log

set -x

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate dllm_zihan

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <num_gpus> <checkpoint_path> [log_file]"
    echo "Example: $0 4 dllm_reasoning/checkpoints/interleaved_sft/global_step_17172 debug_resume.log"
    exit 1
fi

nproc_per_node=$1
checkpoint_path=$2

# 如果提供了第三个参数，则视为日志文件名
log_file=""
if [ "$#" -ge 3 ]; then
    log_file=$3
fi

# 切换到项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"
echo "GPU 数量: $nproc_per_node"
echo "从checkpoint恢复: $checkpoint_path"
if [ -n "$log_file" ]; then
    echo "日志文件: $log_file"
fi

# 创建新的保存目录（基于checkpoint路径）
save_dir="${checkpoint_path}_debug"
echo "新的保存目录: $save_dir"

# 构建训练命令 - 只运行20步用于debug
# 使用checkpoint中的huggingface模型
model_path="${checkpoint_path}/huggingface"

cmd="torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m dllm_reasoning.train_interleaved_sft \
    trainer.default_local_dir=\"$save_dir\" \
    trainer.resume_mode=auto \
    trainer.resume_from_path=\"$checkpoint_path\" \
    trainer.max_debug_steps=20 \
    data.train_files=data/openr1.parquet \
    data.val_files=null \
    data.block_size=4 \
    data.micro_batch_size_per_gpu=2 \
    model.enable_gradient_checkpointing=true \
    model.partial_pretrain=\"$model_path\""

# 如果指定了日志文件，则重定向输出
if [ -n "$log_file" ]; then
    echo "运行训练，输出重定向到: $log_file"
    eval "$cmd" > "$log_file" 2>&1
else
    echo "运行训练（输出到终端）"
    eval "$cmd"
fi
