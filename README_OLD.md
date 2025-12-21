# dLLM Reasoning: 迭代精炼训练框架

基于**迭代精炼（Iterative Refinement）**策略的 SFT 训练框架，用于训练扩散语言模型进行推理任务。

---

## 🚀 快速开始（5分钟上手）

### 1. 克隆仓库

```bash
git clone 
```

---

### 2. 配置环境

#### 方式 A: 创建新环境（推荐）

```bash
# 创建 conda 环境
conda create -n dllm_env python=3.10
conda activate dllm_env

# 安装依赖
cd dllm_reasoning
pip install -r requirements.txt

# 安装 VERL（必需）
cd ..
git clone https://github.com/volcengine/verl
cd verl
pip install -e .
cd ../dllm_reasoning
pip install -e .
```

---

### 3. 准备数据

#### 数据格式

训练数据需要是 **Parquet 文件**，包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `prompt` | 指令/问题 | "What is 2+2?" |
| `target` | 期望的输出 | "2+2 equals 4." |

#### 创建数据文件

```python
import pandas as pd

# 准备你的数据
data = [
    {"prompt": "What is 2+2?", "target": "2+2 equals 4."},
    {"prompt": "Explain gravity", "target": "Gravity is a fundamental force..."},
    # ... 更多数据
]

# 保存为 parquet 格式
df = pd.DataFrame(data)
df.to_parquet("train.parquet", index=False)
```

#### 放置数据文件

```bash
# 在项目根目录创建 data 文件夹
mkdir -p data

# 将数据文件放入 data 目录
cp /path/to/your/train.parquet data/
```

---

### 4. 修改配置（可选）

如果你想修改训练参数，编辑配置文件：dllm_reasoning/dllm_reasoning/config/iterative_refine.yaml


**关键配置项：**

```yaml
data:
  train_files: data/train.parquet      # 训练数据路径
  val_files: data/val.parquet          # 验证数据路径（可选）
  prompt_key: prompt                    # 你的数据中的 prompt 列名
  response_key: target                  # 你的数据中的 target 列名

model:
  partial_pretrain: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  # 预训练模型

trainer:
  total_epochs: 3                       # 训练轮数
  default_local_dir: ./checkpoints/my_exp  # 检查点保存路径
```

---

### 5. 启动训练

```bash
# 确保在根目录
cd /path/to

# 启动训练（4 GPUs）
bash dllm_reasoning/scripts/train.sh 4 ./checkpoints/my_first_exp

# 单 GPU 训练
bash dllm_reasoning/scripts/train.sh 1 ./checkpoints/my_first_exp

# 自定义参数（覆盖配置文件）
bash dllm_reasoning/scripts/train.sh 4 ./checkpoints/my_exp \
    data.train_files=data/my_data.parquet \
    model.partial_pretrain=meta-llama/Llama-3-8B \
    trainer.total_epochs=5
```

**训练启动后会看到：**

```
项目根目录: /path/to
GPU 数量: 4
保存目录: ./checkpoints/my_first_exp
运行训练（输出到终端）
...
[Epoch 1/3] Step 100/1000: loss=1.234, lr=0.00001
```

---

### 6. 推理测试

训练完成后，测试你的模型：

#### 方式 A: 使用提供的脚本

```bash
# 修改脚本中的配置
vim dllm_reasoning/scripts/inference.py

# 修改这两行：
MODEL_PATH = "checkpoints/my_first_exp/global_step_1000/huggingface"
PROMPT = "Your test question here"

# 运行推理
python dllm_reasoning/scripts/inference.py
```

#### 方式 B: 编程接口

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dllm_reasoning.inference import iterative_generate

# 加载模型
model_path = "checkpoints/my_first_exp/global_step_1000/huggingface"
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 准备输入
prompt = "What is 2+2?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# 生成
output_ids = iterative_generate(
    model=model,
    input_ids=input_ids,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_new_tokens=512,
)

# 解码输出
response = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
print(response)
```

---

## 📁 项目结构

训练完成后，你的目录结构如下：

```
PATH/                              # 项目根目录
├── dllm_reasoning/                # 训练包
│   ├── dllm_reasoning/           # 核心代码
│   │   ├── config/               # 配置文件
│   │   ├── trainer/              # 训练器
│   │   ├── losses/               # Loss 函数
│   │   ├── utils/                # 工具函数
│   │   └── inference/            # 推理模块
│   ├── scripts/                  # 启动脚本
│   │   ├── setup_env.sh         # 环境激活
│   │   ├── train.sh             # 训练脚本
│   │   └── inference.py         # 推理脚本
│   ├── setup.py                  # 包配置
│   └── requirements.txt          # 依赖列表
│
├── data/                          # 数据目录（需要创建）
│   ├── train.parquet             # 训练数据
│   └── val.parquet               # 验证数据
│
├── checkpoints/                   # 检查点目录（自动创建）
│   └── my_first_exp/
│       ├── global_step_1000/
│       │   └── huggingface/      # HuggingFace 格式模型（可直接推理）
│       └── global_step_2000/
│
└── log/                           # 日志目录（自动创建）
    └── debug.log                  # 详细调试日志
```

---

## ⚙️ 核心概念

### 迭代精炼训练流程

```
原始数据 (t0): instruction + response

↓ 加噪

加噪数据 (s0): instruction + noisy_response (部分 token 替换为 EOS)

↓ 前向传播

计算 loss: loss_s0 = CrossEntropy(model(s0), t0)

↓ 贪婪解码

精炼数据 (s1): instruction + refined_response

↓ 再次前向传播

计算 loss: loss_s1 = CrossEntropy(model(s1), t0)

↓ 聚合

total_loss = loss_s0 + loss_s1
梯度更新
```

### 与标准 SFT 的区别

- **标准 SFT**: 单轮前向传播，直接优化干净数据
- **迭代精炼**: 多轮前向传播，从噪声数据逐步恢复

---

## 🔧 常用配置

### 调整显存占用

```yaml
# 方法 1: 减小 batch size
data:
  micro_batch_size_per_gpu: 1  # 默认 2

# 方法 2: 启用梯度检查点
model:
  enable_gradient_checkpointing: true

# 方法 3: 减少迭代轮数
iterative:
  num_iterations: 1  # 默认 2
```

### 调整训练策略

```yaml
iterative:
  num_iterations: 2          # 迭代轮数
  noise_min: 0.1            # 最小噪声比例（10%）
  noise_max: 0.9            # 最大噪声比例（90%）
  loss_weights: [1.0, 1.0]  # 每轮 loss 权重

optim:
  lr: 1e-5                  # 学习率
  warmup_steps_ratio: 0.05  # Warmup 比例
  clip_grad: 1.0            # 梯度裁剪
```

---

## ❓ 常见问题

### Q1: 训练时报错 `No module named dllm_reasoning`

**原因**: 没有安装包

**解决**:
```bash
cd dllm_reasoning
pip install -e .
```

### Q2: 训练时报错 `No module named verl`

**原因**: VERL 未安装

**解决**:
```bash
git clone https://github.com/volcengine/verl
cd verl
pip install -e .
```

### Q3: 如何监控训练

训练日志保存在：
- 终端输出：实时显示训练进度
- `log/debug.log`：详细的debug信息

使用 WandB 监控（可选）:
```yaml
trainer:
  logger: ['console', 'wandb']
  project_name: my-project
```

---

## 📊 训练监控指标

训练时会输出以下指标：

- `loss_s0`: 第一轮迭代的 loss
- `loss_s1`: 第二轮迭代的 loss
- `loss_total`: 总 loss（加权聚合）
- `grad_norm`: 梯度范数
- `lr`: 当前学习率
- `noise_mean`: 平均噪声比例

---

## 🎯 进阶使用

### 断点续训

```bash
# 训练会自动从最新检查点恢复
bash dllm_reasoning/scripts/train.sh 4 ./checkpoints/my_exp

# 或在配置中指定
trainer:
  resume_mode: auto  # auto: 自动恢复, disable: 从头训练
```

### 自定义数据列名

如果你的数据列名不是 `prompt` 和 `target`：

```yaml
data:
  prompt_key: instruction  # 你的 prompt 列名
  response_key: output     # 你的 response 列名
```