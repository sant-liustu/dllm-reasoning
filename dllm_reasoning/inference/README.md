# dLLM Reasoning 推理模块

迭代块状生成的推理实现。

## 核心原理

**训练 vs 推理:**

| 阶段 | 输入 | 操作 | 输出 |
|------|------|------|------|
| **训练** | instruction + response | 对 response 加噪 → refine → 学习恢复 | Loss |
| **推理** | prompt | 拼接 EOS 块 → refine → 生成 | Generated text |

**块状生成流程:**

```
步骤1: 拼接 N 个 EOS token
  [prompt][eos][eos]...[eos]  (N 个)

步骤2: 前向传播 → 得到 logits
  利用 next token prediction，可以预测 N+1 个位置

步骤3: 解码生成 N+1 个新 token
  [prompt][tok1][tok2]...[tok_N+1]

步骤4: Refine M 轮（默认 2 轮）
  每轮: 前向 → 解码 → 更新新块

步骤5: 检测停止条件
  - EOS 检测: 新块中有 EOS → 停止
  - 长度限制: 达到 max_length → 停止
  - 继续: 否则进入下一块
```

## 快速使用

### 命令行推理

```bash
# 单个 prompt
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompt "What is 2+2?"

# 批量推理
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompts_file prompts.txt \
    --batch_size 4 \
    --output_file results.jsonl
```

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dllm_reasoning.inference import iterative_generate

# 加载模型
model = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint").cuda()
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")

# 推理
prompt = "What is 2+2?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

output_ids = iterative_generate(
    model=model,
    input_ids=input_ids,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    add_eos_length=127,   # 每块添加 127 个 EOS，生成 128 个 token
    refine_iter=2,        # 每块 refine 2 轮
    max_new_tokens=1024,
)

response = tokenizer.decode(output_ids[0, input_ids.size(1):])
print(response)
```

## 参数说明

### 核心参数

- **`add_eos_length`** (默认 127)
  - 每块添加的 EOS token 数量
  - 实际生成 `add_eos_length + 1` 个 token/块
  - 例如: `add_eos_length=127` → 生成 128 tokens/块

- **`refine_iter`** (默认 2)
  - 每块的 refine 迭代轮数
  - 更多轮次 → 更高质量，但更慢
  - 建议范围: 1-4

- **`max_new_tokens`** (默认 1024)
  - 最大生成的 token 数量
  - 总块数 = ⌈max_new_tokens / (add_eos_length + 1)⌉

- **`max_length`** (默认 8192)
  - 序列的最大长度限制 (prompt + generated)
  - 超过此长度会自动停止

### 停止条件

生成会在以下任一条件满足时停止:

1. **EOS 检测**: 新生成的块中包含 EOS token
2. **长度限制**: 序列长度达到 `max_length`
3. **Token 限制**: 生成的 token 数量达到 `max_new_tokens`

## 文件说明

- **`generator.py`**: 核心推理函数实现
  - `iterative_generate()`: 主推理函数
  - `_refine_simple()`: Refine 循环（不使用 KV cache）
  - `_has_eos()`: EOS 检测
  - `_truncate_at_eos()`: 截断到 EOS

- **`demo.py`**: 完整的推理脚本
  - 支持单个/批量推理
  - 自动应用 chat template
  - 结果保存到文件
  - 命令行参数配置

## 性能优化

当前实现是**简单版本**（不使用 KV cache），适合验证功能正确性。

**未来优化方向:**

1. **KV Cache**: 缓存 prefix 的 KV，只推理新块
   - 预期加速: 2-3x
   - 实现复杂度: 中等

2. **采样支持**: temperature、top-p、top-k
   - 增加生成多样性
   - 可复用 `utils/noise_utils.py:sample_tokens()`

3. **并行解码**: 多块并行推理（需要特殊处理）
   - 预期加速: 大幅提升
   - 实现复杂度: 高

## 与标准 AR 生成的对比

| 维度 | 标准 AR 生成 | dLLM Reasoning (本方法) |
|------|-------------|------------------------|
| **生成方式** | 逐 token | 逐块 (块大小 ≈ 128) |
| **并行度** | 低（序列依赖） | 中（块内并行） |
| **质量控制** | 单次生成 | 多轮 refine |
| **速度** | 快（单步） | 中（多轮 refine） |
| **适用场景** | 通用文本生成 | 推理任务（需要深思熟虑） |

## 常见问题

### Q1: 为什么 `add_eos_length=127` 但生成 128 个 token？

因为 next token prediction 的特性：
- 长度为 N 的序列 → 产生 N 个 logits → 可预测 N 个位置
- 添加 127 个 EOS → 序列长度 +127 → 可预测 127+1=128 个新位置

### Q2: 如何选择 `add_eos_length`？

建议根据任务特点选择:
- **短回答** (几十 tokens): `add_eos_length=31` → 32 tokens/块
- **中等回答** (百余 tokens): `add_eos_length=127` → 128 tokens/块
- **长回答** (数百 tokens): `add_eos_length=255` → 256 tokens/块

### Q3: `refine_iter` 应该设置多少？

- `refine_iter=1`: 无 refine，类似标准生成（但仍是块状）
- `refine_iter=2`: 默认值，平衡质量和速度
- `refine_iter=3-4`: 更高质量，但显著变慢

### Q4: 为什么不用 KV cache？

第一版实现专注于功能正确性，KV cache 会在后续版本添加。您可以参考 `dLLM-Var/evaluation/utils/generate_function.py` 中的实现。

---

**相关文档:**
- 主 README: [../README.md](../README.md)
- 训练文档: 见主 README 的训练部分
- 原理说明: [../../dllm-reasoning.md](../../dllm-reasoning.md)
