#!/bin/bash
# 交错训练启动脚本（Interleaved SFT Training）
#
# 使用方法:
#   bash dllm_reasoning/scripts/train_interleaved.sh <num_gpus> <save_dir> [log_file] [其他配置...]
#
# 示例:
#   bash dllm_reasoning/scripts/train_interleaved.sh 4 ./checkpoints/my_exp
#   bash dllm_reasoning/scripts/train_interleaved.sh 4 ./checkpoints/my_exp training.log
#   nohup bash dllm_reasoning/scripts/train_interleaved.sh 4 ./checkpoints/my_exp training.log &

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <num_gpus> <save_dir> [log_file] [other_configs...]"
    echo "Example: $0 4 ./checkpoints/my_exp"
    echo "Example: $0 4 ./checkpoints/my_exp training.log"
    echo "Example: nohup $0 4 ./checkpoints/my_exp training.log &"
    exit 1
fi

nproc_per_node=$1
save_dir=$2

# 如果提供了第三个参数且不包含'='，则视为日志文件名
log_file=""
if [ "$#" -ge 3 ] && [[ ! "$3" =~ "=" ]]; then
    log_file=$3
    shift 3
else
    shift 2
fi

# 切换到项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"
echo "GPU 数量: $nproc_per_node"
echo "保存目录: $save_dir"
if [ -n "$log_file" ]; then
    echo "日志文件: $log_file"
fi

# 构建训练命令（包含默认参数）
cmd="torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m dllm_reasoning.train_interleaved_sft \
    trainer.default_local_dir=\"$save_dir\" \
    trainer.resume_mode=auto \
    data.train_files=data/openr1.parquet \
    data.val_files=null \
    data.block_size=4 \
    data.micro_batch_size_per_gpu=2 \
    model.enable_gradient_checkpointing=true \
    model.partial_pretrain=dllm_reasoning/dllm_reasoning/model/DLLM-1.5B \
    $@"

# 如果指定了日志文件，则重定向输出
if [ -n "$log_file" ]; then
    echo "运行训练，输出重定向到: $log_file"
    eval "$cmd" > "$log_file" 2>&1
else
    echo "运行训练（输出到终端）"
    eval "$cmd"
fi
