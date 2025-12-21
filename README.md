# dLLM Reasoning: Interleaved SFT 训练框架

基于 **Interleaved SFT（交错式监督微调）**的训练框架，使用 FlexAttention BlockMask 进行 Next Block Prediction 训练。

---

## 🚀 快速开始

### 1. 环境配置

#### 创建 Conda 环境

```bash
# 创建环境
conda create -n dllm_env python=3.10
conda activate dllm_env

# 安装项目依赖
cd dllm_reasoning
pip install -r requirements.txt

# 安装 VERL（必需）
cd ..
git clone https://github.com/volcengine/verl
cd verl
pip install -e .

# 安装本项目
cd ../dllm_reasoning
pip install -e .
```

---

### 2. 数据准备

#### 数据格式要求

训练数据需要是 **Parquet 文件**，包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `prompt` | str 或 list | 用户问题（支持字符串或 chat 消息列表） |
| `target` | str 或 list | 模型回答（支持字符串或 chat 消息列表） |

#### 重要：`<think>` 标签处理

**如果你的 `target` 列中的回答以 `<think>` 标签开头**，数据集会自动提取 `<think>` 后面的内容作为训练目标：

```python
# 原始数据示例
{
    "prompt": [{"role": "user", "content": "What is 2+2?"}],
    "target": [{"role": "assistant", "content": "<think>\n Let me calculate...</think>The answer is 4."}]
}

# 实际训练时使用的内容：
# "Let me calculate...</think>The answer is 4."
# （自动去掉了开头的 "<think>",因为在prompt的template中会在处理完的prompt后接一个think和换行号，所以处理target的时候就手动去掉了前面的think和换行号，如果你的taget开头没有换行号或think，需注意这里的数据处理，得进行相应的适配修改
```

处理逻辑详见 [interleaved_sft_dataset.py:506-509](dllm_reasoning/trainer/interleaved_sft_dataset.py#L506-L509)

#### 创建数据文件示例

```python
import pandas as pd

# 示例 : Chat 消息格式（推荐）
data = [
    {
        "prompt": [{"role": "user", "content": "What is 2+2?"}],
        "target": [{"role": "assistant", "content": "<think>Let me think...</think>The answer is 4."}]
    }
]

# 保存为 parquet
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

### 3. 修改配置

编辑配置文件：[dllm_reasoning/dllm_reasoning/config/interleaved_sft.yaml](dllm_reasoning/dllm_reasoning/config/interleaved_sft.yaml)

**关键配置项：**

```yaml
data:
  train_files: data/train.parquet      # 训练数据路径
  val_files: null                       # 验证数据路径（null=不使用）
  prompt_key: prompt                    # Parquet 中的 prompt 列名
  response_key: target                  # Parquet 中的 response 列名
  block_size: 4                         # Block 大小（Next Block Prediction）
  max_length: 2048                      # 最大序列长度

model:
  partial_pretrain: path/to/your/model  # 预训练模型路径
  enable_gradient_checkpointing: true   # 梯度检查点（节省显存）

optim:
  lr: 1e-5                              # 学习率
  gradient_accumulation_steps: 64       # 梯度累积步数

trainer:
  total_epochs: 3                       # 训练轮数
  default_local_dir: ./checkpoints/my_exp  # 检查点保存路径
  save_checkpoint_steps: 1000           # 每 N 步保存检查点
  logger: ['console', 'wandb']          # 日志记录器
```

---

### 4. 启动训练

训练脚本：[dllm_reasoning/scripts/train_interleaved.sh](dllm_reasoning/scripts/train_interleaved.sh)

```bash
# 确保在项目根目录
cd /path/to/Dream

# 4 GPU 训练（推荐）
bash dllm_reasoning/scripts/train_interleaved.sh 4 ./checkpoints/my_exp

# 单 GPU 训练
bash dllm_reasoning/scripts/train_interleaved.sh 1 ./checkpoints/my_exp

# 后台训练（输出到日志文件）
nohup bash dllm_reasoning/scripts/train_interleaved.sh 4 ./checkpoints/my_exp training.log &

# 自定义参数（覆盖配置文件）
bash dllm_reasoning/scripts/train_interleaved.sh 4 ./checkpoints/my_exp \
    data.train_files=data/my_data.parquet \
    model.partial_pretrain=meta-llama/Llama-3-8B \
    trainer.total_epochs=5
```

**训练启动后会看到：**

```
项目根目录: /path/to/Dream
GPU 数量: 4
保存目录: ./checkpoints/my_exp
[2025-01-15 10:00:00] [INFO] 初始化分布式环境...
[2025-01-15 10:00:01] [INFO] 分布式环境初始化完成: rank=0, world_size=4
[2025-01-15 10:00:05] [INFO] 创建交错训练数据集...
[InterleavedSFTDataset] Loaded 10000 samples
[InterleavedSFTDataset] Block size: 4
...
[Epoch 1/3] Step 100/1000: loss=1.234, lr=0.00001
```

---

### 5. 推理测试

训练完成后，使用以下脚本测试你的模型。

#### 测试脚本：[dllm_reasoning/test_list/inference/ckpt_inference_by_hand.py](dllm_reasoning/test_list/inference/ckpt_inference_by_hand.py)

**修改脚本配置：**

```python
# 修改第 303-304 行
MODEL_PATH = "./checkpoints/my_exp/global_step_1000/huggingface"
DATA_PATH = "data/train.parquet"  # 用于测试的数据文件
```

**运行推理：**

```bash
python dllm_reasoning/test_list/inference/ckpt_inference_by_hand.py
```

#### 推理特性

脚本支持以下测试场景（可在脚本中注释/取消注释）：

1. **场景1：自然的自回归生成** (第 22-34 行)
   - 标准的自回归生成，不使用 block-wise 推理

2. **场景1.1：Teacher Forcing 过拟合测试** (第 36-68 行)
   - 使用 ground truth 测试模型的过拟合准确率

3. **场景2：Blockwise 生成** (第 71-134 行)
   - 使用 block-wise 推理生成文本
   - 每次生成一个 block 的 token

4. **场景2.1：Blockwise Teacher Forcing 测试** (第 136-214 行)
   - Block-wise 模式下的过拟合准确率测试

5. **场景3：配备模型自我感知的 Blockwise 生成** (第 217-299 行，**默认启用**)
   - Block-wise 生成 + 自适应停止策略
   - 如果 token 预测概率 < 0.7，停止当前 block
   - 如果检测到重复 token，停止当前 block

**输出示例：**

```
==========================================
测试Checkpoint 1000 - 三种Teacher Forcing场景
==========================================

场景3：配备模型自我感知的Blockwise生成
Block 0, Token 0: pred_token ('The')
top-5 predictions:
  Tokens: 'The', 'A', 'To', 'In', 'For'; Probabilities: 0.8234, 0.0512, 0.0234, 0.0123, 0.0098
Block 0, Token 1: pred_token ('answer')
...
✅ Blockwise生成完成
一共生成了156个token
平均每个block生成了3.12个token
生成的tokens：The answer is 4 because 2+2 equals 4.
```

---

## 📁 项目结构

```
Dream/                                   # 项目根目录
├── dllm_reasoning/                      # 训练包
│   ├── dllm_reasoning/                  # 核心代码
│   │   ├── config/                      # 配置文件
│   │   │   └── interleaved_sft.yaml    # 交错训练配置
│   │   ├── trainer/                     # 训练器
│   │   │   ├── interleaved_sft_dataset.py     # 数据集（重要！）
│   │   │   └── interleaved_sft_trainer.py     # 训练器
│   │   ├── train_interleaved_sft.py    # 主训练脚本
│   │   └── ...
│   ├── scripts/                         # 启动脚本
│   │   └── train_interleaved.sh        # 训练启动脚本
│   ├── test_list/inference/             # 推理测试
│   │   └── ckpt_inference_by_hand.py   # 推理测试脚本（重要！）
│   ├── setup.py                         # 包配置
│   └── requirements.txt                 # 依赖列表
│
├── data/                                # 数据目录（需创建）
│   └── train.parquet                    # 训练数据
│
├── checkpoints/                         # 检查点目录（自动创建）
│   └── my_exp/
│       ├── global_step_1000/
│       │   └── huggingface/             # HuggingFace 格式模型
│       └── global_step_2000/
│
└── log/                                 # 日志目录（可选）
```

---

## ⚙️ 核心概念

### Interleaved SFT 训练流程

与传统 SFT 不同，Interleaved SFT 将 response 序列分成多个 block，并通过 mask token 进行并行预测：

```
原始序列:  [P0, P1, P2] [R0, R1, R2, R3, R4, R5, R6, R7]
           ↑ Prompt      ↑ Response (block_size=4)

交错格式:  [P0, P1, P2] [M, M, M] [R0, R1, R2, R3] [M, M, M] [R4, R5, R6, R7]
                         ↑ Masks    ↑ Block 0       ↑ Masks    ↑ Block 1

预测目标:
  - P2 预测 → R0
  - M0 预测 → R1 (并行)
  - M1 预测 → R2 (并行)
  - M2 预测 → R3 (并行)
  - R0-R3 自回归预测后续 token
```

### 与标准 SFT 的区别

| 特性 | 标准 SFT | Interleaved SFT |
|------|----------|-----------------|
| 训练方式 | 单轮自回归 | Block-wise 并行预测 |
| 推理速度 | 顺序生成 | 可并行生成多个 token |
| 训练复杂度 | 简单 | 需要 FlexAttention 支持 |

---

## 🔧 常用配置调整

### 显存优化

```yaml
# 方法 1: 减小 batch size
data:
  micro_batch_size_per_gpu: 1  # 默认 2

# 方法 2: 启用梯度检查点
model:
  enable_gradient_checkpointing: true

# 方法 3: 减小 block size
data:
  block_size: 2  # 默认 4（会影响并行效率）
```

### 训练策略调整

```yaml
data:
  block_size: 4                         # Block 大小（影响并行效率）
  max_length: 2048                      # 序列最大长度

optim:
  lr: 1e-5                              # 学习率
  warmup_steps_ratio: 0.05              # Warmup 比例
  clip_grad: 1.0                        # 梯度裁剪
  gradient_accumulation_steps: 64       # 梯度累积

trainer:
  save_checkpoint_steps: 1000           # 保存频率
  max_ckpt_to_keep: 3                   # 保留检查点数量
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

### Q3: 数据中的 `<think>` 标签如何处理？

**答案**: 数据集会自动处理：

- 如果 `target` 列的内容以 `<think>` 开头，会自动去掉开头的 `<think>`
- 保留后续的所有内容（包括 `</think>` 和其他内容）
- 详见代码：[interleaved_sft_dataset.py:506-509](dllm_reasoning/trainer/interleaved_sft_dataset.py#L506-L509)

### Q4: 如何监控训练？

训练日志输出位置：
- **终端输出**: 实时显示训练进度
- **WandB**: 在配置中启用 `trainer.logger: ['console', 'wandb']`

### Q5: 断点续训

训练脚本会自动从最新检查点恢复：

```bash
# 直接运行，会自动恢复
bash dllm_reasoning/scripts/train_interleaved.sh 4 ./checkpoints/my_exp
```

配置文件中的恢复设置：
```yaml
trainer:
  resume_mode: auto  # auto: 自动恢复最新检查点 | disable: 从头训练
```

---

## 📊 训练监控指标

训练时输出的关键指标：

- `loss`: 总损失
- `grad_norm`: 梯度范数
- `lr`: 当前学习率
- `tokens_per_sec`: 训练吞吐量

---

## 🎯 进阶使用

### 自定义数据列名

如果你的数据列名不是 `prompt` 和 `target`：

```yaml
data:
  prompt_key: instruction  # 你的 prompt 列名
  response_key: output     # 你的 response 列名
```

### 使用 Tensor Parallel

对于大模型（如 70B），可以启用 Tensor Parallel：

```yaml
model:
  tensor_parallel_size: 4  # 使用 4-way TP
```

### 多节点训练

修改训练脚本的 `torchrun` 参数：

```bash
# 在每个节点上运行
torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    -m dllm_reasoning.train_interleaved_sft \
    ...
```

---

## 📝 引用

## 📄 许可证

Apache License 2.0
