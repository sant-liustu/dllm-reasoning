"""
简化版渐进式测试 - 用于debug
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.optim import AdamW

from dllm_reasoning.model.DLLM.modeling_dllm import DLLMForCausalLM

print("=" * 100)
print("DEBUG: 渐进式测试")
print("=" * 100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/dllm_reasoning/model/DLLM-1.5B"

print(f"\n1. 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"   ✓ Tokenizer 加载完成")

model = DLLMForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True,
).to(device)
print(f"   ✓ 模型加载成功 (设备: {device})")

print(f"\n2. 构造Level 0测试数据...")
prompt_tokens = torch.tensor([1000, 1001, 1002], device=device)
response_tokens = torch.tensor([2000, 2001, 2002, 2003], device=device)
input_ids = torch.cat([prompt_tokens, response_tokens], dim=0).unsqueeze(0)
position_ids = torch.arange(0, len(input_ids[0]), device=device).unsqueeze(0)
labels = torch.tensor([-100, -100, 2000, 2001, 2002, 2003, -100], device=device).unsqueeze(0)

print(f"   输入序列: {input_ids[0].tolist()}")
print(f"   Labels:   {labels[0].tolist()}")

print(f"\n3. 测试前向传播...")
model.train()
optimizer = AdamW(model.parameters(), lr=1e-4)

try:
    outputs = model(input_ids=input_ids, position_ids=position_ids)
    print(f"   ✓ 前向传播成功")
    print(f"   ✓ Logits shape: {outputs.logits.shape}")

    print(f"\n4. 计算loss...")
    logits = outputs.logits
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    print(f"   ✓ Loss 计算成功: {loss.item():.6f}")

    print(f"\n5. 测试反向传播...")
    optimizer.zero_grad()
    loss.backward()
    print(f"   ✓ 反向传播成功")

    optimizer.step()
    print(f"   ✓ 参数更新成功")

    print(f"\n✅ 所有步骤成功！测试通过！")

except Exception as e:
    print(f"\n❌ 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
