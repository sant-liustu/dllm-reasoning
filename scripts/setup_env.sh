#!/bin/bash
# Dream SFT 训练环境激活脚本
# 使用方法: source activate_dllm_env.sh

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dllm_zihan

# 验证环境
echo "=========================================="
echo "Dream SFT 训练环境已激活"
echo "=========================================="
echo "Python: $(which python)"
echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo ""
echo "关键库版本："
python -c "
import torch
import transformers
import verl
print(f'  - PyTorch: {torch.__version__}')
print(f'  - Transformers: {transformers.__version__}')
print(f'  - CUDA: {torch.version.cuda}')
print(f'  - GPU 数量: {torch.cuda.device_count()}')
print(f'  - VERL: 已安装 (verl-0.7.0.dev0)')
"
echo ""
echo "✅ 环境准备就绪！可以开始训练了"
echo "=========================================="
