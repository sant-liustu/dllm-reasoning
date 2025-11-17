# dLLM Reasoning: 迭代精炼训练框架

## 概述

这是一个基于**迭代精炼（Iterative Refinement）**策略的 SFT 训练框架，用于训练扩散语言模型（dLLM）进行推理任务。

### 核心思想

与传统的 Dream 模型不同，本框架：
1. ✅ **不修改模型架构** - 使用标准的因果 AR 模型（不需要双向注意力）
2. ✅ **迭代精炼训练** - 多轮前向传播，逐步精炼生成质量
3. ✅ **只对 response 加噪** - instruction 区域保持不变
4. ✅ **使用 EOS token 加噪** - 而非特殊的 MASK token

### 训练流程

```
原始数据 (t0):
  instruction + response

↓ 第一轮

加噪 (s0):
  instruction + noisy_response (部分 token 替换为 EOS)

↓

前向传播:
  logits_s0 = model(s0)

↓

计算 loss:
  loss_s0 = CrossEntropy(logits_s0, t0)  # 对原始 t0 计算
  【注意】只在 response 区域计算 loss

↓

贪婪解码:
  s1 = greedy_decode(logits_s0)  # 只在 response 区域解码

↓ 第二轮

前向传播:
  logits_s1 = model(s1)

↓

计算 loss:
  loss_s1 = CrossEntropy(logits_s1, t0)  # 仍然对原始 t0 计算

↓

聚合 loss:
  total_loss = loss_s0 + loss_s1

↓

梯度更新:
  total_loss.backward()
  optimizer.step()
```

---

## 目录结构

```
dllm_reasoning/
├── README.md                          # 本文件
├── train_iterative_refine.py          # 主训练脚本
│
├── config/
│   └── iterative_refine.yaml          # 训练配置文件
│
├── trainer/
│   ├── sft_dataset.py                 # 数据集类（从 Dream 复制）
│   └── iterative_refine_trainer.py    # 迭代精炼训练器主类
│
├── losses/
│   └── iterative_loss.py              # Loss 计算函数
│
├── utils/
│   └── noise_utils.py                 # 加噪、解码等工具函数
│
├── inference/                         # 🆕 推理模块
│   ├── __init__.py
│   ├── generator.py                   # 核心：iterative_generate 函数
│   └── demo.py                        # 推理演示脚本
│
└── scripts/
    ├── run_train.sh                   # 便捷启动脚本
    └── verify_label_alignment.py      # Label 对齐验证脚本
```

---

## 安装

### 1. 激活环境

```bash
conda activate dllm_zihan  # 或你创建的其他环境
```

### 2. 确保依赖已安装

关键依赖：
- ✅ PyTorch 2.5.1
- ✅ Transformers 4.57.1
- ✅ VERL 0.7.0.dev0
- ✅ Hydra 1.3.2
- ✅ TensorDict 0.10.0

如果还没安装 VERL，请参考项目根目录的 `ENVIRONMENT_SETUP.md`。

---

## 快速开始

### 1. 准备数据

数据格式：**Parquet 文件**，包含以下列：
- `instruction`: 指令文本
- `output`: 期望的输出

示例：
```python
import pandas as pd

data = [
    {"instruction": "What is 2+2?", "output": "2+2 equals 4."},
    {"instruction": "Explain gravity", "output": "Gravity is a fundamental force..."},
]

df = pd.DataFrame(data)
df.to_parquet("train.parquet", index=False)
```

### 2. 修改配置文件

编辑 `dllm_reasoning/config/iterative_refine.yaml`：

```yaml
data:
  train_files: /your/path/to/train.parquet
  val_files: /your/path/to/val.parquet
  prompt_key: instruction
  response_key: output

model:
  partial_pretrain: meta-llama/Llama-3-8B  # 你的预训练模型

trainer:
  default_local_dir: ./checkpoints/my_exp
```

### 3. 启动训练

#### 方法 A：使用便捷脚本（推荐）

```bash
cd /data/v-zihaliu/amlt-RLF-ExpConfig/Dream

bash dllm_reasoning/scripts/run_train.sh 4 ./checkpoints/my_exp \
    data.train_files=/your/path/to/train.parquet \
    model.partial_pretrain=meta-llama/Llama-3-8B
```

参数说明：
- `4`: 使用 4 个 GPU
- `./checkpoints/my_exp`: 检查点保存目录
- 后续参数：覆盖配置文件中的设置

#### 方法 B：直接使用 torchrun

```bash
cd /data/v-zihaliu/amlt-RLF-ExpConfig/Dream

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m dllm_reasoning.train_iterative_refine \
    data.train_files=/your/path/to/train.parquet \
    model.partial_pretrain=meta-llama/Llama-3-8B \
    trainer.default_local_dir=./checkpoints/my_exp
```

---

## 核心配置说明

### 迭代配置 (`iterative` 部分)

```yaml
iterative:
  num_iterations: 2          # 迭代轮数（默认 2：s0 → s1）
  noise_min: 0.1             # 最小噪声比例（10% token 被替换）
  noise_max: 0.9             # 最大噪声比例（90% token 被替换）
  loss_weights: [1.0, 1.0]   # 每轮 loss 的权重
```

**调整建议**：
- **更多轮次**：`num_iterations: 3` 或 `4`（需要更多显存）
- **更少噪声**：`noise_max: 0.5`（更温和的训练）
- **加权策略**：`loss_weights: [0.5, 1.0]`（更重视后续轮次）

### 训练配置 (`trainer` 部分)

```yaml
trainer:
  total_epochs: 3
  save_checkpoint_steps: 1000
  logger: ['console', 'wandb']  # 日志后端
```

### 优化器配置 (`optim` 部分)

```yaml
optim:
  lr: 2e-5                    # 学习率
  warmup_steps_ratio: 0.1     # warmup 比例
  clip_grad: 1.0              # 梯度裁剪
```

---

## 验证 Label 对齐

运行验证脚本确保 loss 计算正确（next token prediction）：

```bash
python dllm_reasoning/scripts/verify_label_alignment.py
```

预期输出：
```
🎉 所有测试通过！

确认：
  ✅ Label 对齐正确（next token prediction）
  ✅ Response mask 正确应用
  ✅ Shift 操作实现正确
```

---

## 与 Dream 的主要区别

| 特性 | Dream | 本框架 (dLLM Reasoning) |
|------|-------|------------------------|
| **模型架构** | 双向注意力（修改模型） | 标准因果 AR 模型（不修改） |
| **训练方式** | 单轮前向传播 | 多轮前向传播（迭代精炼） |
| **加噪位置** | 全序列 | 只对 response 区域 |
| **加噪 Token** | MASK token | EOS token |
| **Loss 计算** | 预测被 mask 的 token | 每轮都对原始 token 计算 next token prediction |
| **梯度更新** | 每轮一次 | 多轮聚合后一次 |

---

## 训练监控

### 查看日志

训练时会输出以下指标：

```
train/loss_s0: 第一轮的 loss
train/loss_s1: 第二轮的 loss
train/loss_total: 总 loss（加权聚合）
train/grad_norm: 梯度范数
train/lr: 当前学习率
train/noise_mean: 平均噪声比例
```

### 使用 WandB

配置文件中设置：
```yaml
trainer:
  logger: ['console', 'wandb']
  project_name: my-project
  experiment_name: my-exp-001
```

登录 WandB 账号：
```bash
wandb login
```

---

## 常见问题

### Q1: 如何调整显存占用？

**选项 A**：减小 batch size
```yaml
data:
  micro_batch_size_per_gpu: 2  # 从 4 改为 2
```

**选项 B**：启用梯度检查点
```yaml
model:
  enable_gradient_checkpointing: true
```

**选项 C**：减少迭代轮数
```yaml
iterative:
  num_iterations: 1  # 只用一轮（退化为标准 SFT）
```

### Q2: 如何检查 EOS token？

```bash
python << EOF
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
EOF
```

### Q3: 如何断点续训？

目前尚未实现自动断点续训。你可以手动指定检查点：

```bash
# TODO: 实现断点续训功能
```

### Q4: 训练速度慢怎么办？

**可能原因**：
1. 多轮前向传播增加了计算量（这是设计特点）
2. 梯度检查点开启（节省显存但增加计算）
3. 序列长度过长

**优化建议**：
1. 减少 `num_iterations`
2. 关闭梯度检查点（如果显存够用）
3. 使用更小的 `max_length`

---

## 代码复用说明

本框架复用了以下 Dream 和 VERL 的代码：

### 从 Dream 复用：
- ✅ `trainer/sft_dataset.py` - 数据集类
- ✅ `utils/noise_utils.py:q_sample` - 加噪函数

### 从 VERL 复用：
- ✅ `verl.utils.distributed` - 分布式初始化
- ✅ `verl.utils.fsdp_utils` - FSDP 包装工具
- ✅ `verl.utils.fs` - 文件系统工具
- ✅ `verl.utils.torch_functional` - 学习率调度器
- ✅ `verl.utils.tracking` - 训练追踪

### 自己实现：
- ✅ `trainer/iterative_refine_trainer.py` - 迭代精炼训练器
- ✅ `losses/iterative_loss.py` - 多轮 loss 计算
- ✅ `utils/noise_utils.py:greedy_decode_response` - 贪婪解码

---

---

## 推理使用

训练完成后，可以使用推理脚本进行生成。

### 快速开始

**单个 prompt 推理:**

```bash
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompt "What is 2+2?"
```

**批量推理（从文件）:**

```bash
# 创建 prompts 文件
cat > prompts.txt << EOF
What is 2+2?
Explain quantum physics in simple terms.
Write a Python function to sort a list.
EOF

# 批量推理
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompts_file prompts.txt \
    --output_file results.jsonl \
    --batch_size 4
```

**使用 chat template:**

```bash
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompt "What is 2+2?" \
    --use_chat_template
```

### 推理参数说明

**核心参数:**

- `--model_path`: 模型路径（必需）
- `--prompt`: 单个 prompt（与 `--prompts_file` 二选一）
- `--prompts_file`: prompts 文件，每行一个（与 `--prompt` 二选一）
- `--use_chat_template`: 自动应用 tokenizer 的 chat template

**生成参数:**

- `--add_eos_length`: 每块添加的 EOS 数量（默认 127）
  - 实际生成 `add_eos_length + 1` 个 token/块
- `--refine_iter`: 每块的 refine 轮数（默认 2）
- `--max_new_tokens`: 最大生成 token 数（默认 1024）
- `--max_length`: 序列最大长度（默认 8192）
- `--batch_size`: 批大小（默认 1）

**输出参数:**

- `--output_file`: 输出文件（默认 `inference_results.jsonl`）
- `--max_display`: 终端显示的结果数量（默认 5）

### 推理原理

与训练类似，推理也采用**迭代块状生成**:

```
1. 拼接 N 个 EOS token
   [prompt][eos][eos]...[eos]  (N 个)

2. 前向传播 → 得到 logits
   可以预测 N+1 个位置（利用 next token prediction）

3. 解码生成 N+1 个新 token
   [prompt][tok1][tok2]...[tok_N+1]

4. Refine M 轮（默认 2 轮）
   每轮重新前向 → 解码 → 更新

5. 检测 EOS 或达到最大长度
   - 如果新块中有 EOS → 停止
   - 如果达到 max_length → 停止
   - 否则继续下一块
```

**关键理解:**
- 训练时: 对 response 加噪 → refine → 学习恢复
- 推理时: 拼接 EOS 块 → refine → 生成高质量输出

### 编程接口

也可以在代码中直接调用推理函数:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dllm_reasoning.inference import iterative_generate

# 加载模型
model = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint").cuda()
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")

# 准备输入
prompt = "What is 2+2?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# 生成
output_ids = iterative_generate(
    model=model,
    input_ids=input_ids,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    add_eos_length=127,
    refine_iter=2,
    max_new_tokens=512,
)

# 解码
response = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
print(response)
```

---

## TODO

- [x] ~~添加推理脚本~~ ✅ 已完成
- [ ] 实现断点续训功能
- [ ] 添加验证集评估
- [ ] 支持 LoRA 微调
- [ ] 添加更多噪声调度策略（cosine、linear 等）
- [ ] 支持更多轮次的迭代（s2, s3, ...）
- [ ] 推理添加 KV Cache 优化
- [ ] 推理添加采样功能（temperature、top-p）

---

## 引用

如果你使用了本框架，请引用：

```bibtex
@misc{dllm_reasoning_2025,
  title={dLLM Reasoning: Iterative Refinement Training Framework},
  author={Your Name},
  year={2025}
}
```

---

**最后更新**: 2025-11-11
**版本**: 1.1.0 (新增推理功能)
**许可**: Apache 2.0
