# 使用示例

## 基本使用

### 1. 单个 Prompt 推理

```bash
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompt "What is 2+2?"
```

**输出:**
```
=============================================================================
Generated 1 responses
=============================================================================

--- Example 1 ---
Prompt (15 tokens):
What is 2+2?

Response (42 tokens):
2+2 equals 4. This is a basic arithmetic operation...

=============================================================================
```

### 2. 批量推理

**准备 prompts 文件:**
```bash
cat > my_prompts.txt << EOF
What is 2+2?
Explain the theory of relativity in simple terms.
Write a Python function to calculate factorial.
What are the main causes of climate change?
EOF
```

**运行批量推理:**
```bash
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompts_file my_prompts.txt \
    --batch_size 2 \
    --output_file my_results.jsonl \
    --max_display 10
```

**查看结果:**
```bash
cat my_results.jsonl
```

输出格式（每行一个 JSON）:
```json
{"prompt": "What is 2+2?", "response": "2+2 equals 4...", "prompt_length": 15, "response_length": 42}
{"prompt": "Explain...", "response": "...", "prompt_length": 20, "response_length": 150}
```

### 3. 使用 Chat Template

如果您的模型是 instruction-tuned 模型（如 Llama-3-Instruct），建议使用 chat template：

```bash
python -m dllm_reasoning.inference.demo \
    --model_path meta-llama/Llama-3-8B-Instruct \
    --prompt "What is 2+2?" \
    --use_chat_template
```

这会自动将 prompt 包装为对话格式：
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

## 高级使用

### 4. 自定义生成参数

```bash
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompt "Write a long essay about AI" \
    --add_eos_length 255 \    # 更大的块 (生成 256 tokens/块)
    --refine_iter 3 \          # 更多 refine 轮次
    --max_new_tokens 2048 \    # 生成更长的文本
    --max_length 4096          # 更大的序列长度限制
```

### 5. 小块生成（快速模式）

```bash
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompt "What is 2+2?" \
    --add_eos_length 31 \      # 小块 (生成 32 tokens/块)
    --refine_iter 1 \          # 无 refine
    --max_new_tokens 128       # 短回答
```

### 6. 保存为 JSON 格式

```bash
python -m dllm_reasoning.inference.demo \
    --model_path /path/to/checkpoint \
    --prompts_file my_prompts.txt \
    --output_file results.json  # 注意：.json 而非 .jsonl
```

输出格式（单个 JSON 数组）:
```json
[
  {"prompt": "...", "response": "...", ...},
  {"prompt": "...", "response": "...", ...}
]
```

## Python API 使用

### 7. 基本 API 调用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dllm_reasoning.inference import iterative_generate

# 加载模型（只需加载一次）
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/checkpoint",
    torch_dtype=torch.bfloat16,
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")

# 单个推理
prompt = "What is 2+2?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

output_ids = iterative_generate(
    model=model,
    input_ids=input_ids,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    add_eos_length=127,
    refine_iter=2,
    max_new_tokens=512,
)

# 解码响应
response = tokenizer.decode(
    output_ids[0, input_ids.size(1):],
    skip_special_tokens=True
)
print(response)
```

### 8. 批量 API 调用

```python
prompts = [
    "What is 2+2?",
    "Explain quantum physics.",
    "Write a Python function.",
]

# Tokenize（左填充用于批量推理）
tokenizer.padding_side = "left"
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

# 批量生成
output_ids = iterative_generate(
    model=model,
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    add_eos_length=127,
    refine_iter=2,
    max_new_tokens=512,
)

# 解码
for i, (prompt, output) in enumerate(zip(prompts, output_ids)):
    # 计算 prompt 长度（排除 padding）
    prompt_len = inputs["attention_mask"][i].sum().item()
    response_ids = output[prompt_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"\nPrompt {i+1}: {prompt}")
    print(f"Response: {response}")
```

### 9. 使用 Chat Template（API）

```python
# 应用 chat template
messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 推理
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
output_ids = iterative_generate(...)
```

## 参数调优建议

### 根据任务类型选择参数

**短问答（如 GSM8K）:**
```bash
--add_eos_length 63    # 生成 64 tokens/块
--refine_iter 2        # 标准 refine
--max_new_tokens 256   # 短回答
```

**中等长度（如 MATH）:**
```bash
--add_eos_length 127   # 生成 128 tokens/块（默认）
--refine_iter 2        # 标准 refine
--max_new_tokens 512   # 中等长度
```

**长文本生成（如 HumanEval）:**
```bash
--add_eos_length 255   # 生成 256 tokens/块
--refine_iter 3        # 更多 refine
--max_new_tokens 1024  # 长回答
```

### 速度 vs 质量权衡

**快速模式（牺牲质量）:**
```bash
--add_eos_length 127
--refine_iter 1        # 最少 refine
```

**平衡模式（推荐）:**
```bash
--add_eos_length 127
--refine_iter 2        # 标准 refine
```

**高质量模式（较慢）:**
```bash
--add_eos_length 127
--refine_iter 4        # 最多 refine
```

## 常见问题解决

### Q1: 生成速度太慢

**解决方案:**
1. 减少 `refine_iter` (如 1)
2. 减少 `add_eos_length` (如 31)
3. 减少 `batch_size`
4. 使用更小的模型

### Q2: 生成质量不佳

**解决方案:**
1. 增加 `refine_iter` (如 3-4)
2. 确保使用了 chat template（如果是 instruction 模型）
3. 检查模型是否正确加载
4. 尝试不同的 `add_eos_length`

### Q3: OOM (显存不足)

**解决方案:**
1. 减少 `batch_size`
2. 减少 `max_length`
3. 减少 `add_eos_length`
4. 使用 `torch.float16` 而非 `bfloat16`

### Q4: 输出包含大量重复

**解决方案:**
1. 检查 EOS token 是否正确
2. 增加 `refine_iter`
3. 训练时可能需要调整 loss weights

## 性能基准

### 典型速度（单 GPU，A100 40GB）

| 配置 | Tokens/秒 | 说明 |
|------|----------|------|
| add_eos_length=127, refine_iter=1 | ~80 | 快速模式 |
| add_eos_length=127, refine_iter=2 | ~50 | 标准模式（推荐） |
| add_eos_length=127, refine_iter=4 | ~30 | 高质量模式 |
| add_eos_length=255, refine_iter=2 | ~45 | 大块模式 |

注：实际速度取决于模型大小、序列长度等因素。

---

**更多信息:**
- 推理原理: [README.md](README.md)
- 训练指南: [../README.md](../README.md)
- 实现细节: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
