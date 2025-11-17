# FSDP Checkpoint 直接推理指南

## 概述

这个模块提供了直接使用 FSDP checkpoint 进行推理的功能，**无需转换为 HuggingFace 格式**。

## 为什么需要这个？

FSDP checkpoint 包含分布式训练的元数据（device_mesh, DTensor 等），很难转换为标准的 HuggingFace 格式。这个脚本通过**保持 FSDP 环境**来直接加载和使用这些 checkpoint。

## 使用方法

### 基本用法

```bash
# 使用便捷脚本（推荐）
bash dllm_reasoning/inference/run_fsdp_inference.sh 4 checkpoints/openr1_test_fixed/global_step_21000

# 自定义 prompt
bash dllm_reasoning/inference/run_fsdp_inference.sh 4 checkpoints/openr1_test_fixed/global_step_21000 "Solve: 25 * 4 = ?"
```

### 高级用法

```bash
# 直接使用 Python 模块
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m dllm_reasoning.inference.fsdp_demo \
    --checkpoint_dir checkpoints/openr1_test_fixed/global_step_21000 \
    --prompt "What is the capital of France?" \
    --add_eos_length 7 \
    --refine_iter 2 \
    --max_new_tokens 512 \
    --use_chat_template

# 批量推理（从文件读取 prompts）
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m dllm_reasoning.inference.fsdp_demo \
    --checkpoint_dir checkpoints/openr1_test_fixed/global_step_21000 \
    --prompts_file my_prompts.txt \
    --output_file results.jsonl \
    --add_eos_length 7 \
    --refine_iter 2
```

## 重要参数说明

### 必需参数

- `--checkpoint_dir`: FSDP checkpoint 目录路径
- `--prompt` 或 `--prompts_file`: 单个 prompt 或 prompt 文件

### 生成参数

- `--add_eos_length`: 每个块添加的 EOS token 数量（默认: 127，建议测试时用 7）
- `--refine_iter`: 每个块的精炼迭代次数（默认: 2）
- `--max_new_tokens`: 最大生成 token 数（默认: 512）
- `--max_length`: 最大序列长度（默认: 8192）

### 其他参数

- `--use_chat_template`: 使用 tokenizer 的 chat template
- `--output_file`: 保存结果到文件（支持 .json 和 .jsonl）

## 重要注意事项

### 1. GPU 数量必须与训练时一致

**这是最关键的要求！**

```bash
# 如果训练时使用了 4 个 GPU
torchrun --nproc_per_node=4 ...  # ✅ 正确

torchrun --nproc_per_node=2 ...  # ❌ 错误！会加载失败
```

您可以从 checkpoint 中查看训练时的 GPU 数量：
```bash
cat checkpoints/openr1_test_fixed/global_step_21000/fsdp_config.json
# 输出: {"FSDP_version": 1, "world_size": 4}
```

### 2. 环境要求

- 必须使用与训练时相同的环境（dllm_zihan）
- 需要激活环境：`source activate_dllm_env.sh`

### 3. 单 GPU 模式暂不支持

由于 FSDP checkpoint 的分片特性，目前暂不支持单 GPU 推理。如果需要单 GPU 推理，请：
1. 先转换为 HuggingFace 格式（需要在训练时配置）
2. 或者使用 `demo.py` 脚本

## 工作原理

1. **初始化 FSDP 环境**：使用与训练时相同的配置（wrap_policy, mixed_precision, device_mesh 等）
2. **创建 FSDP 模型**：包装模型为 FSDP，确保配置完全匹配
3. **加载 checkpoint**：使用 VERL 的 `FSDPCheckpointManager` 加载分片权重
4. **推理**：在 FSDP 环境中进行推理，使用 `iterative_generate`

## 与普通推理的区别

| 特性 | 普通推理 (demo.py) | FSDP 推理 (fsdp_demo.py) |
|------|-------------------|------------------------|
| 模型格式 | HuggingFace | FSDP checkpoint |
| GPU 要求 | 单卡或多卡 | 必须与训练时一致 |
| 启动方式 | `python` | `torchrun` |
| 转换需求 | 需要转换 | 无需转换 |
| 速度 | 快（单卡） | 较慢（分布式开销） |

## 示例输出

```bash
$ bash dllm_reasoning/inference/run_fsdp_inference.sh 4 checkpoints/openr1_test_fixed/global_step_21000 "What is 2+2?"

==========================================
FSDP Checkpoint 直接推理
==========================================
Checkpoint: checkpoints/openr1_test_fixed/global_step_21000
World Size: 4
Prompt:     What is 2+2?
==========================================

项目根目录: /data/v-zihaliu/amlt-RLF-ExpConfig/Dream

激活 dLLM 环境...
环境已激活

开始推理...

2025-11-11 15:00:00 - INFO - Initialized distributed: rank=0/4, device=cuda:0
2025-11-11 15:00:05 - INFO - Loading config and tokenizer from checkpoints/.../huggingface
2025-11-11 15:00:10 - INFO - FSDP model initialized
2025-11-11 15:00:15 - INFO - Checkpoint loaded successfully
2025-11-11 15:00:20 - INFO - Processing 1 prompt(s)...

================================================================================
Prompt 1/1
================================================================================
2025-11-11 15:00:20 - INFO - Prompt (10 tokens): What is 2+2?...

Prompt: What is 2+2?

Response:
2+2 equals 4. This is a basic arithmetic operation where we add two numbers together.

================================================================================

==========================================
✅ 推理完成！
==========================================
```

## 故障排除

### 问题 1: "world_size 不匹配"

**错误**：加载 checkpoint 时失败
**原因**：使用的 GPU 数量与训练时不一致
**解决**：检查 `fsdp_config.json`，使用正确的 `--nproc_per_node`

### 问题 2: "device_mesh 错误"

**错误**：`AssertionError: The device mesh of a tensor should be a root mesh.`
**原因**：FSDP 配置不匹配
**解决**：确保脚本中的配置与训练时一致（已在脚本中正确配置）

### 问题 3: "内存不足"

**错误**：CUDA OOM
**原因**：模型太大，4 个 GPU 不够
**解决**：减小 `max_length` 或 `max_new_tokens`

## 未来改进

- [ ] 支持单 GPU 推理（需要实现 checkpoint 合并）
- [ ] 支持不同 GPU 数量的推理（需要实现 re-sharding）
- [ ] 添加流式输出支持
- [ ] 优化多轮对话支持

## 相关文件

- `fsdp_demo.py`: 主推理脚本
- `run_fsdp_inference.sh`: 便捷启动脚本
- `generator.py`: 底层生成函数（`iterative_generate`）
