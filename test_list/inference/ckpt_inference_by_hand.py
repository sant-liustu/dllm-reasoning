#测试checkpoint推理功能是否正常
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np

# 将项目根目录添加到sys.path中
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def test_inference_in_train_mod(model, tokenizer, prompt_ids, full_ids=None, device='cuda', block_size=None):
    #这个测试支持有full_ids传入和没有两种情况
    #故意将模型设置为训练模式，确保推理代码不受模型模式影响
    model.train()
    # model.eval()

    # # 场景1：测试自然的自回归生成能力
    # print("\n场景1：自然自回归生成")
    # with torch.no_grad():
    #     generated_ids = model.generate(
    #         input_ids=prompt_ids.unsqueeze(0),
    #         max_new_tokens=1000,
    #         do_sample=False,
    #         eos_token_id=tokenizer.eos_token_id,
    #         pad_token_id=tokenizer.pad_token_id,
    #         use_cache=False
    #     )
    # generated_text = tokenizer.decode(generated_ids[0][prompt_ids.size(0):], skip_special_tokens=False)
    # print(f"\n生成结果：\n{generated_text}")

    # #场景1.1：自回归时Teacher Forcing - 过拟合准确率测试
    # if full_ids is not None:
    #     print("分割线---------------------------")
    #     print("Full IDs provided, 使用teacher forcing进行过拟合准确率测试")
    #     print("\n场景1.1：Teacher Forcing - 过拟合准确率测试")
    #     with torch.no_grad():
    #         outputs = model(full_ids.unsqueeze(0), use_cache=False)
    #         logits = outputs.logits[0]  # [seq_len, vocab]
    #     predictions = logits.argmax(dim=-1)

    #     # 统计准确率
    #     num_correct = 0
    #     prompt_len = prompt_ids.size(0)
    #     full_ids_len = full_ids.size(0)
    #     print(f"\n前20个token的预测详情:")
    #     for i in range(full_ids_len - prompt_len):
    #         pred_pos = prompt_len - 1 + i
    #         true_pos = prompt_len + i

    #         pred_id = predictions[pred_pos].item()
    #         true_id = full_ids[true_pos].item()

    #         is_correct = pred_id == true_id
    #         if is_correct:
    #             num_correct += 1

    #         if i < 20:  # 只显示前20个
    #             status = "✅" if is_correct else "❌"
    #             pred_text = tokenizer.decode([pred_id])
    #             true_text = tokenizer.decode([true_id])
    #             print(f"  [{i:2d}] {status} pred={pred_id:6d} '{pred_text}' | true={true_id:6d} '{true_text}'")
    #     accuracy = num_correct / (full_ids_len - prompt_len)
    #     print(f"\n过拟合准确率: {accuracy*100:.2f}%")
    
    # model.train()
    # # 场景2：测试blockwise生成能力
    # if block_size is not None:
    #     print("分割线---------------------------")
    #     print("\n场景2：Blockwise生成")
    #     eos_token_id = tokenizer.eos_token_id
    #     num_masks_to_add = block_size-1
    #     current_ids = prompt_ids.clone()
    #     prompt_len = prompt_ids.size(0)
    #     for block_idx in range(20):
    #         # 添加mask tokens
    #         mask_block = torch.full((num_masks_to_add,), eos_token_id, device=device, dtype=current_ids.dtype)
    #         input_with_masks = torch.cat([current_ids, mask_block], dim=0)

    #         # 构造block_info - 格式: (seg_type, seg_idx, seg_len)
    #         # 已经有生成的response,把它们当作prompt延伸
    #         block_info = [('mask', 0, num_masks_to_add)]
    #         effective_prompt_len = current_ids.size(0)


    #         position_ids = torch.arange(input_with_masks.size(0), device=device)

    #         # 前向传播
    #         with torch.no_grad():
    #             outputs = model(
    #                 input_with_masks.unsqueeze(0),
    #                 position_ids=position_ids.unsqueeze(0),
    #                 block_info=[block_info],
    #                 prompt_len=[effective_prompt_len],
    #                 seq_lens=[input_with_masks.size(0)],
    #                 use_cache=False
    #             )
    #             logits = outputs.logits[0]

    #         # 预测: prompt最后一个位置以及所有mask位置预测后续tokens
    #         mask_start = current_ids.size(0)
    #         mask_logits = logits[mask_start-1:]  # [num_masks_to_add+1, vocab]
    #         # print(f"\n mask_logits shape: {mask_logits.shape}")
    #         predicted_tokens = mask_logits.argmax(dim=-1)

    #         for i in range(predicted_tokens.size(0)):
    #             pred_probs = mask_logits[i].softmax(dim=-1)
    #             top5_probs, top5_ids = pred_probs.topk(5)
    #             print(f"Block {block_idx}, Token {i}: pred_token ('{tokenizer.decode([predicted_tokens[i].item()])}')")
    #             print(f"top-5 predictions:")
    #             token_list = []
    #             prob_list = []
    #             for prob, token_id in zip(top5_probs.tolist(), top5_ids.tolist()):
    #                 token_str = tokenizer.decode([token_id])
    #                 token_list.append(f"'{token_str}'")
    #                 prob_list.append(f"{prob:.4f}")
    #             print(f"  Tokens: {', '.join(token_list)}; Probabilities: {', '.join(prob_list)}")
    #         print("分割线-------------------")

    #         # 更新当前输入
    #         current_ids = torch.cat([current_ids, predicted_tokens], dim=0)


    #         #如果遇到eos token就停止生成
    #         if eos_token_id in predicted_tokens:
    #             print("遇到eos token，停止生成")
    #             break

    #     print("✅ Blockwise生成完成")
    #     print(f"生成的tokens：{tokenizer.decode(current_ids)}")

    # if block_size is not None and full_ids is not None:
    #     # 场景2.1：Blockwise生成时Teacher Forcing - 过拟合准确率测试
    #     print("分割线---------------------------")
    #     print("\n场景2.1：Blockwise生成时Teacher Forcing - 过拟合准确率测试")
    #     eos_token_id = tokenizer.eos_token_id
    #     num_masks_to_add = block_size-1
    #     current_ids = prompt_ids.clone()
    #     prompt_len = prompt_ids.size(0)
    #     full_ids_len = full_ids.size(0)

    #     num_correct = 0

    #     for block_idx in range(20):
    #         # 添加mask tokens
    #         mask_block = torch.full((num_masks_to_add,), eos_token_id, device=device, dtype=current_ids.dtype)
    #         input_with_masks = torch.cat([current_ids, mask_block], dim=0)

    #         # 构造block_info - 格式: (seg_type, seg_idx, seg_len)
    #         block_info = [('mask', 0, num_masks_to_add)]
    #         effective_prompt_len = current_ids.size(0)

    #         position_ids = torch.arange(input_with_masks.size(0), device=device)
    #         # print(f"\n position_ids: {position_ids}")
    #         # 前向传播
    #         with torch.no_grad():
    #             outputs = model(
    #                 input_with_masks.unsqueeze(0),
    #                 position_ids=position_ids.unsqueeze(0),
    #                 block_info=[block_info],
    #                 prompt_len=[effective_prompt_len],
    #                 seq_lens=[input_with_masks.size(0)],
    #                 use_cache=False
    #             )
    #             logits = outputs.logits[0]

    #         # 预测: prompt最后一个位置以及所有mask位置预测后续tokens
    #         pred_start = current_ids.size(0)
    #         pred_logits = logits[pred_start-1:]  # [num_masks_to_add+1, vocab]
    #         predicted_tokens = pred_logits.argmax(dim=-1)

    #         # 统计准确率
    #         for i in range(predicted_tokens.size(0)):
    #             pred_pos = current_ids.size(0) - 1 + i
    #             true_pos = current_ids.size(0) + i

    #             if true_pos >= full_ids_len:
    #                 break

    #             pred_id = predicted_tokens[i].item()
    #             true_id = full_ids[true_pos].item()

    #             if pred_id == true_id:
    #                 num_correct += 1



    #             if block_idx < 20:  # 只显示前20个block的token预测详情
    #                 print(f"Block {block_idx}, Token {i}: self_id={input_with_masks[pred_pos]} | pred_id={pred_id} | true_id={true_id} | {'✅' if pred_id == true_id else '❌'}")
    #                 #我这里还想观察pred_logits翻译成采样概率后，第i个token的概率分布情况
    #                 print(f'pred_token: {tokenizer.decode([pred_id])} | true_token: {tokenizer.decode([true_id])}')
    #                 pred_probs = pred_logits[i].softmax(dim=-1)
    #                 top5_probs, top5_ids = pred_probs.topk(5)
    #                 print(f"  Top-5: {top5_probs.tolist()}")  # 看最高的5个概率
    #                 print("分割线-------------------")

    #         if true_pos >= full_ids_len:
    #             break
    #         # 更新下一轮输入，因为是teacher forcing，直接用ground truth
    #         ground_truth_block = full_ids[current_ids.size(0): current_ids.size(0) + block_size]
    #         #验证一下加对了，decode出来看看
    #         # print(f"\n 之前：{tokenizer.decode(current_ids)}")
    #         # print(f" 本轮ground truth：{tokenizer.decode(ground_truth_block)}")
    #         current_ids = torch.cat([current_ids, ground_truth_block], dim=0)
    #         # print(f" 之后：{tokenizer.decode(current_ids)}")
    #         print("✅ 本block Teacher Forcing完成,分割线-------------------")
    #         print('\n\n')
    #     pred_len = min(current_ids.size(0) - prompt_len, full_ids_len - prompt_len)
    #     accuracy = num_correct / pred_len
    #     print(f"\nBlockwise生成时的过拟合准确率: {accuracy*100:.2f}%")


    # 场景3：测试配备模型自我感知的blockwise生成能力
    if block_size is not None:
        print("分割线---------------------------")
        print("\n场景3：配备模型自我感知的Blockwise生成")
        eos_token_id = tokenizer.eos_token_id
        num_masks_to_add = block_size-1
        current_ids = prompt_ids.clone()
        prompt_len = prompt_ids.size(0)
        for block_idx in range(1000):
            # 添加mask tokens
            mask_block = torch.full((num_masks_to_add,), eos_token_id, device=device, dtype=current_ids.dtype)
            input_with_masks = torch.cat([current_ids, mask_block], dim=0)

            # 构造block_info - 格式: (seg_type, seg_idx, seg_len)
            # 已经有生成的response,把它们当作prompt延伸
            block_info = [('mask', 0, num_masks_to_add)]
            effective_prompt_len = current_ids.size(0)


            position_ids = torch.arange(input_with_masks.size(0), device=device)

            # 前向传播
            with torch.no_grad():
                outputs = model(
                    input_with_masks.unsqueeze(0),
                    position_ids=position_ids.unsqueeze(0),
                    block_info=[block_info],
                    prompt_len=[effective_prompt_len],
                    seq_lens=[input_with_masks.size(0)],
                    use_cache=False
                )
                logits = outputs.logits[0]

            # 预测: prompt最后一个位置以及所有mask位置预测后续tokens
            mask_start = current_ids.size(0)
            mask_logits = logits[mask_start-1:]  # [num_masks_to_add+1, vocab]
            # print(f"\n mask_logits shape: {mask_logits.shape}")
            predicted_tokens = mask_logits.argmax(dim=-1)

            append_tokens_num = 0
            old_token_id = None
            for i in range(predicted_tokens.size(0)):
                pred_probs = mask_logits[i].softmax(dim=-1)
                top5_probs, top5_ids = pred_probs.topk(5)
                token_list = []
                prob_list = []
                #如果发现最高prob小于0.7，就不采样这个token，直接停止当前block的生成
                #第一个token是由原来的next token predict决定的，我们总是接受其，不论其概率如何
                if top5_probs[0] < 0.7 and i >= 1:
                    print(f"  最高概率 {top5_probs[0]:.4f} 小于0.7，停止当前block生成")
                    break
                if old_token_id is not None and predicted_tokens[i].item() == old_token_id and i >=1:
                    print(f"  发现重复token '{tokenizer.decode([predicted_tokens[i].item()])}'，停止当前block生成")
                    break
                old_token_id = predicted_tokens[i].item()
                append_tokens_num += 1
                print(f"Block {block_idx}, Token {i}: pred_token ('{tokenizer.decode([predicted_tokens[i].item()])}')")
                print(f"top-5 predictions:")
                for prob, token_id in zip(top5_probs.tolist(), top5_ids.tolist()):
                    token_str = tokenizer.decode([token_id])
                    token_list.append(f"'{token_str}'")
                    prob_list.append(f"{prob:.4f}")
                print(f"  Tokens: {', '.join(token_list)}; Probabilities: {', '.join(prob_list)}")
                #

            predicted_tokens = predicted_tokens[:append_tokens_num]
            
            print(f"本block实际生成了{predicted_tokens.size(0)}个token")
            # 更新当前输入
            current_ids = torch.cat([current_ids, predicted_tokens], dim=0)

            print("分割线-------------------")
            print("\n")

            #如果遇到eos token就停止生成
            if eos_token_id in predicted_tokens:
                print("遇到eos token，停止生成")
                break

        print("✅ Blockwise生成完成")
        print(f"一共生成了{current_ids.size(0) - prompt_len}个token")
        print(f"平均每个block生成了{(current_ids.size(0) - prompt_len)/ (block_idx+1):.2f}个token")
        print(f"生成的tokens：{tokenizer.decode(current_ids)}")


def main():
    MODEL_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/dllm_reasoning/checkpoints/interleaved_sft_1202/global_step_14000/huggingface"
    DATA_PATH = "/data/v-zihaliu/amlt-RLF-ExpConfig/Dream/data/openr1.parquet"

    # 从路径中提取checkpoint编号，用于打印信息
    import re
    match = re.search(r'global_step_(\d+)', MODEL_PATH)
    ckpt_num = match.group(1) if match else "unknown"

    print("="*80)
    print(f"测试Checkpoint {ckpt_num} - 三种Teacher Forcing场景")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_parquet(DATA_PATH)
    sample = df.iloc[0]

    prompt_messages = sample['prompt']
    target_messages = sample['target']

    # print("\nprompt_messages：\n")
    # print(prompt_messages)
    # print("\ntarget_messages：\n")
    # print(target_messages)

    # 处理numpy arrays
    if isinstance(prompt_messages, np.ndarray):
        prompt_messages = prompt_messages.tolist() if prompt_messages.ndim == 0 else list(prompt_messages)
    if isinstance(target_messages, np.ndarray):
        target_messages = target_messages.tolist() if target_messages.ndim == 0 else list(target_messages)

    # 提取content
    if isinstance(target_messages, (list, tuple)) and len(target_messages) > 0 and isinstance(target_messages[0], dict):
        ground_truth_content = target_messages[0].get("content", "")
    else:
        ground_truth_content = target_messages

    # print("\nground_truth_content：\n")
    # print(ground_truth_content)

    # 去掉<think>标签
    if ground_truth_content.strip().startswith('<think>'):
        think_start = ground_truth_content.find('<think>')
        ground_truth_content = ground_truth_content[think_start + 7:].lstrip()

    # print("\nground_truth_content去掉<think>标签后：\n")
    # print(ground_truth_content)

    print("✅ 数据加载完成")

    # 加载模型
    print("\n加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    print("✅ 模型加载完成")

    # Tokenize
    prompt_only_str = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )
    full_conversation_str = prompt_only_str + ground_truth_content + tokenizer.eos_token

    # print("\nprompt_only_str：\n")
    # print(prompt_only_str)
    # print("\nfull_conversation_str：\n")
    # print(full_conversation_str)

    prompt_ids = tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    full_ids = tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

    #这里full_ids如果有就可以传入，如果没有也可以不传入
    test_inference_in_train_mod(model, tokenizer, prompt_ids, full_ids, device=device, block_size=4)


if __name__ == "__main__":
    main()