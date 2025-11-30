# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Interleaved SFT Dataset for Block-Parallel Training

This module implements the interleaved training data format where:
- Original sequence is split into blocks of size k
- Each block is followed by (k-1) mask tokens for parallel prediction
- Position IDs are designed to make each mask "perceive" only preceding real tokens
- Attention mask ensures proper visibility constraints

Example (block_size=4):
  Original:  [t0, t1, t2, t3, t4, t5, t6, t7, ...]

  Interleaved input:
    [t0, t1, t2, t3] [M, M, M] [t4, t5, t6, t7] [M, M, M] ...

  Position IDs:
    [0,  1,  2,  3]  [4, 5, 6] [4,  5,  6,  7]  [8, 9, 10] ...

  The masks after block 0 "think" they are at positions 4,5,6 following t0-t3
  The masks after block 1 "think" they are at positions 8,9,10 following t0-t7
"""

from typing import List, Union, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import pandas as pd
from functools import partial

# Try to import verl utilities, fallback if not available
try:
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils import hf_tokenizer
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    def copy_local_path_from_hdfs(path, verbose=False):
        return path
    def compute_position_id_with_mask(attention_mask):
        return torch.cumsum(attention_mask, dim=-1) - 1
    def hf_tokenizer(name):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)


def create_interleaved_training_data(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert standard SFT data to interleaved format for block-parallel training.

    Args:
        input_ids: Original input token IDs [seq_len]
        loss_mask: Original loss mask (1 for response tokens, 0 for prompt) [seq_len]
        block_size: Number of tokens per block (k)
        mask_token_id: Token ID used for mask positions
        pad_token_id: Token ID used for padding

    Returns:
        interleaved_input_ids: [new_seq_len] with mask tokens inserted
        interleaved_position_ids: [new_seq_len] with proper position encoding
        interleaved_loss_mask: [new_seq_len] loss mask for the new sequence
        block_boundaries: List of (start, end) for each segment type
    """
    seq_len = input_ids.shape[0]
    device = input_ids.device

    # Find the actual sequence length (excluding padding)
    # Assuming padding is at the end with pad_token_id
    actual_len = seq_len
    for i in range(seq_len - 1, -1, -1):
        if input_ids[i] != pad_token_id:
            actual_len = i + 1
            break

    # Find prompt length (where loss_mask is 0)
    prompt_len = 0
    for i in range(actual_len):
        if loss_mask[i] == 0:
            prompt_len = i + 1
        else:
            break

    # Response part
    response_ids = input_ids[prompt_len:actual_len]
    response_len = response_ids.shape[0]

    if response_len == 0:
        # No response, return original
        position_ids = torch.arange(seq_len, device=device)
        return input_ids, position_ids, loss_mask, []

    # Number of complete blocks in response
    num_blocks = (response_len + block_size - 1) // block_size

    # Number of mask tokens per block = block_size - 1
    num_masks_per_block = block_size - 1

    # Build interleaved sequence
    interleaved_ids = []
    interleaved_pos = []
    interleaved_loss = []
    block_info = []  # Track block boundaries

    # First, add prompt (unchanged)
    interleaved_ids.append(input_ids[:prompt_len])
    interleaved_pos.append(torch.arange(prompt_len, device=device))
    interleaved_loss.append(torch.zeros(prompt_len, dtype=loss_mask.dtype, device=device))

    current_pos = prompt_len  # Current position counter for real tokens

    # Build interleaved sequence with pattern: [mask][block][mask][block]...
    # This allows parallel prediction from the very first response token
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, response_len)
        block_tokens = response_ids[block_start:block_end]
        actual_block_size = block_tokens.shape[0]

        # Add mask tokens BEFORE each block if the block is large enough
        # We add (block_size - 1) masks to predict positions 2, 3, ..., block_size within the block
        # So we only add masks if actual_block_size > num_masks_per_block
        # This ensures the mask group has valid labels
        if actual_block_size > num_masks_per_block:
            # Add (block_size - 1) mask tokens
            mask_tokens = torch.full(
                (num_masks_per_block,),
                mask_token_id,
                dtype=input_ids.dtype,
                device=device
            )

            # Position IDs for masks: start from current_pos
            mask_positions = torch.arange(
                current_pos,
                current_pos + num_masks_per_block,
                device=device
            )

            interleaved_ids.append(mask_tokens)
            interleaved_pos.append(mask_positions)
            # Loss mask: 1 for mask positions (we compute loss here)
            interleaved_loss.append(torch.ones(num_masks_per_block, dtype=loss_mask.dtype, device=device))

            block_info.append(('mask', len(interleaved_ids) - 1, num_masks_per_block))

        # Add block tokens
        # Position IDs for block tokens: masks already "used up" positions current_pos to current_pos+num_masks_per_block-1
        # So block starts at current_pos + (num_masks_per_block if we added masks else 0)
        # Actually, since mask and block predict the same tokens (with offset), they share position space
        # Wait, let me reconsider the position IDs...

        # Position IDs: The masks predict tokens at positions current_pos+1, current_pos+2, ..., current_pos+num_masks_per_block
        # The block tokens are at positions current_pos, current_pos+1, ..., current_pos+actual_block_size-1
        # So masks and block tokens overlap in position space (intentionally)
        block_positions = torch.arange(
            current_pos,
            current_pos + actual_block_size,
            device=device
        )

        interleaved_ids.append(block_tokens)
        interleaved_pos.append(block_positions)
        # Loss mask: 1 for response tokens (we want to learn to predict them)
        interleaved_loss.append(torch.ones(actual_block_size, dtype=loss_mask.dtype, device=device))

        block_info.append(('real', len(interleaved_ids) - 1, actual_block_size))

        # Update position for next block
        current_pos += actual_block_size

    # Concatenate all parts
    interleaved_input_ids = torch.cat(interleaved_ids, dim=0)
    interleaved_position_ids = torch.cat(interleaved_pos, dim=0)
    interleaved_loss_mask = torch.cat(interleaved_loss, dim=0)

    return interleaved_input_ids, interleaved_position_ids, interleaved_loss_mask, block_info


def create_interleaved_labels(
    original_input_ids: torch.Tensor,
    interleaved_input_ids: torch.Tensor,
    interleaved_loss_mask: torch.Tensor,
    block_size: int,
    prompt_len: int,
    pad_token_id: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Create labels for interleaved sequence with proper AR shift.

    For AR models, position i predicts position i+1. So:
    - Real token at position i has label = original_token[i+1]
    - Mask token at position j (in mask group after block b) has label = original_token[block_b_end + j_in_group + 1]

    Args:
        original_input_ids: Original sequence without interleaving
        interleaved_input_ids: Interleaved sequence with masks
        interleaved_loss_mask: Loss mask for interleaved sequence
        block_size: Block size used for interleaving
        prompt_len: Length of prompt (no loss computed)
        pad_token_id: Padding token ID
        ignore_index: Index to ignore in loss computation

    Returns:
        labels: [interleaved_seq_len] with proper next-token labels
    """
    device = interleaved_input_ids.device
    interleaved_len = interleaved_input_ids.shape[0]
    original_len = original_input_ids.shape[0]

    # Initialize labels with ignore_index
    labels = torch.full((interleaved_len,), ignore_index, dtype=torch.long, device=device)

    # Find actual length in original (excluding padding)
    actual_original_len = original_len
    for i in range(original_len - 1, -1, -1):
        if original_input_ids[i] != pad_token_id:
            actual_original_len = i + 1
            break

    # Response in original
    response_ids = original_input_ids[prompt_len:actual_original_len]
    response_len = response_ids.shape[0]

    if response_len == 0:
        return labels

    num_blocks = (response_len + block_size - 1) // block_size
    num_masks_per_block = block_size - 1

    # Track position in interleaved sequence
    # New structure: [prompt][masks0][block0][masks1][block1]...
    interleaved_pos = prompt_len  # Start after prompt

    for block_idx in range(num_blocks):
        block_start_in_response = block_idx * block_size
        block_end_in_response = min((block_idx + 1) * block_size, response_len)
        actual_block_size = block_end_in_response - block_start_in_response

        # Add mask tokens BEFORE the block if block is large enough
        has_masks = actual_block_size > num_masks_per_block

        if has_masks:
            # Labels for mask tokens
            # Masks predict tokens at positions block_start+1, block_start+2, block_start+3
            for i in range(num_masks_per_block):
                mask_pos_in_interleaved = interleaved_pos + i
                # Mask i predicts response token at block_start + i + 1
                target_pos_in_response = block_start_in_response + i + 1

                if target_pos_in_response < response_len:
                    labels[mask_pos_in_interleaved] = response_ids[target_pos_in_response]

            interleaved_pos += num_masks_per_block

        # Labels for real tokens in this block
        # Token at position i predicts token at position i+1
        for i in range(actual_block_size):
            token_pos_in_interleaved = interleaved_pos + i
            next_token_pos_in_response = block_start_in_response + i + 1

            if next_token_pos_in_response < response_len:
                labels[token_pos_in_interleaved] = response_ids[next_token_pos_in_response]
            # else: last token of response, label stays ignore_index

        interleaved_pos += actual_block_size

    # ✅ 修复：Prompt 的最后一个位置应该预测 response 的第一个 token
    # 这样才能实现 P[-1] → R1 的预测！
    if prompt_len > 0 and response_len > 0:
        labels[prompt_len - 1] = response_ids[0]  # P[-1] → R1

    # Set other prompt labels to ignore_index (前 prompt_len-1 个位置)
    if prompt_len > 1:
        labels[:prompt_len - 1] = ignore_index

    # Also handle loss_mask=0 positions (but preserve P[-1] if it should have a label)
    for i in range(interleaved_len):
        if interleaved_loss_mask[i] == 0 and i != prompt_len - 1:
            labels[i] = ignore_index

    return labels


def create_interleaved_attention_mask(
    seq_len: int,
    prompt_len: int,
    block_size: int,
    num_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create attention mask for interleaved training.

    Rules:
    1. Real tokens: causal within same block + see all previous blocks' real tokens
    2. Mask tokens: see current and all previous blocks' real tokens + causal within mask group
    3. Real tokens CANNOT see mask tokens
    4. Mask tokens CANNOT see future blocks' real tokens or future mask groups

    Args:
        seq_len: Total interleaved sequence length
        prompt_len: Length of prompt
        block_size: Size of each block
        num_blocks: Number of blocks in response
        device: Device for tensor

    Returns:
        attention_mask: [seq_len, seq_len] boolean mask (True = can attend)
    """
    num_masks_per_block = block_size - 1

    # Calculate segment boundaries
    # Structure: [prompt][block0][masks0][block1][masks1]...
    segments = []
    pos = 0

    # Prompt segment
    segments.append(('prompt', pos, prompt_len))
    pos += prompt_len

    # Response segments
    for block_idx in range(num_blocks):
        # Real block
        segments.append(('real', pos, block_size))
        pos += block_size

        # Mask segment (except possibly last)
        if block_idx < num_blocks - 1:
            segments.append(('mask', pos, num_masks_per_block))
            pos += num_masks_per_block

    # Create mask matrix
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    # Fill in attention patterns
    for q_seg_idx, (q_type, q_start, q_len) in enumerate(segments):
        for kv_seg_idx, (kv_type, kv_start, kv_len) in enumerate(segments):

            if q_type == 'prompt':
                # Prompt tokens: standard causal within prompt
                if kv_type == 'prompt':
                    # Causal within prompt
                    for i in range(q_len):
                        for j in range(min(i + 1, kv_len)):
                            mask[q_start + i, kv_start + j] = True

            elif q_type == 'real':
                # Real tokens in response
                if kv_type == 'prompt':
                    # Can see entire prompt
                    mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True

                elif kv_type == 'real':
                    if kv_seg_idx < q_seg_idx:
                        # Can see all previous real blocks
                        mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True
                    elif kv_seg_idx == q_seg_idx:
                        # Causal within same block
                        for i in range(q_len):
                            for j in range(min(i + 1, kv_len)):
                                mask[q_start + i, kv_start + j] = True
                    # Cannot see future real blocks

                # Real tokens CANNOT see mask tokens (kv_type == 'mask')

            elif q_type == 'mask':
                # Mask tokens
                if kv_type == 'prompt':
                    # Can see entire prompt
                    mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True

                elif kv_type == 'real':
                    # Find which real block this mask group follows
                    # Mask group at segment q_seg_idx follows real block at q_seg_idx - 1
                    mask_follows_seg_idx = q_seg_idx - 1

                    if kv_seg_idx <= mask_follows_seg_idx:
                        # Can see current and all previous real blocks
                        mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = True

                elif kv_type == 'mask':
                    if kv_seg_idx == q_seg_idx:
                        # Causal within same mask group
                        for i in range(q_len):
                            for j in range(min(i + 1, kv_len)):
                                mask[q_start + i, kv_start + j] = True
                    # Cannot see other mask groups

    return mask


class InterleavedSFTDataset(Dataset):
    """
    SFT Dataset with interleaved format for block-parallel training.
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        prompt_key: str = "prompt",
        response_key: str = "response",
        max_length: int = 1024,
        block_size: int = 4,
        truncation: str = "error",
        mask_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        """
        Args:
            parquet_files: Path(s) to parquet data files
            tokenizer: Tokenizer or path to tokenizer
            prompt_key: Column name for prompt
            response_key: Column name for response
            max_length: Maximum sequence length (before interleaving)
            block_size: Number of tokens per block
            truncation: How to handle sequences > max_length
            mask_token_id: Token ID for mask (defaults to eos_token_id)
            pad_token_id: Token ID for padding
        """
        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_length = max_length
        self.block_size = block_size

        # Mask token defaults to EOS (as mentioned in design)
        self.mask_token_id = mask_token_id if mask_token_id is not None else self.tokenizer.eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id

        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f"[InterleavedSFTDataset] Loaded {len(self.dataframe)} samples")
        print(f"[InterleavedSFTDataset] Block size: {self.block_size}")
        print(f"[InterleavedSFTDataset] Mask token ID: {self.mask_token_id}")

    def _tokenize_example(self, example) -> dict:
        """Tokenize a single example (standard SFT format)."""
        import numpy as np

        prompt = example[self.prompt_key]
        response = example[self.response_key]

        # Handle numpy arrays
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist() if prompt.ndim == 0 else list(prompt)
        if isinstance(response, np.ndarray):
            response = response.tolist() if response.ndim == 0 else list(response)

        # Determine format
        is_chat_format = isinstance(prompt, (list, tuple)) and len(prompt) > 0 and isinstance(prompt[0], dict)

        if is_chat_format:
            # Chat format
            if isinstance(response, (list, tuple)) and len(response) > 0 and isinstance(response[0], dict):
                response_content = response[0].get("content", "")
            else:
                response_content = response

            prompt_only_str = self.tokenizer.apply_chat_template(
                list(prompt), add_generation_prompt=True, tokenize=False
            )

            # Handle <think> tag
            if response_content.strip().startswith('<think>'):
                think_start = response_content.find('<think>')
                response_content = response_content[think_start + 7:].lstrip()

            full_conversation_str = prompt_only_str + response_content + self.tokenizer.eos_token

            prompt_ids = self.tokenizer(prompt_only_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            full_ids = self.tokenizer(full_conversation_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

            prompt_length = prompt_ids.shape[0]
            input_ids = full_ids
        else:
            # Simple format
            if not isinstance(prompt, str):
                prompt_chat = list(prompt)
            else:
                prompt_chat = [{"role": "user", "content": prompt}]

            if isinstance(response, (list, tuple)) and len(response) > 0 and isinstance(response[0], dict):
                response_content = response[0].get("content", "")
            elif isinstance(response, dict):
                response_content = response.get("content", "")
            else:
                response_content = response

            prompt_chat_str = self.tokenizer.apply_chat_template(
                prompt_chat, add_generation_prompt=True, tokenize=False
            )
            response_chat_str = response_content + self.tokenizer.eos_token

            prompt_ids = self.tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            response_ids = self.tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

            prompt_length = prompt_ids.shape[0]
            input_ids = torch.cat((prompt_ids, response_ids), dim=-1)

        # Truncation
        if input_ids.shape[0] > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[:self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                prompt_length = max(0, prompt_length - (input_ids.shape[0] - self.max_length))
            else:
                raise ValueError(f"Sequence length {input_ids.shape[0]} > max_length {self.max_length}")

        # Create loss mask (0 for prompt, 1 for response)
        loss_mask = torch.ones(input_ids.shape[0], dtype=torch.long)
        loss_mask[:prompt_length] = 0

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "prompt_length": prompt_length,
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        example = self.dataframe.iloc[item]
        tokenized = self._tokenize_example(example)

        input_ids = tokenized["input_ids"]
        loss_mask = tokenized["loss_mask"]
        prompt_length = tokenized["prompt_length"]

        # Create interleaved format
        interleaved_ids, interleaved_pos, interleaved_loss_mask, block_info = create_interleaved_training_data(
            input_ids=input_ids,
            loss_mask=loss_mask,
            block_size=self.block_size,
            mask_token_id=self.mask_token_id,
            pad_token_id=self.pad_token_id,
        )

        # Create labels with AR shift
        labels = create_interleaved_labels(
            original_input_ids=input_ids,
            interleaved_input_ids=interleaved_ids,
            interleaved_loss_mask=interleaved_loss_mask,
            block_size=self.block_size,
            prompt_len=prompt_length,
            pad_token_id=self.pad_token_id,
        )

        # Note: We don't create the dense 2D attention_mask here anymore
        # Instead, we return block_info and prompt_len for dynamic BlockMask construction

        return {
            "input_ids": interleaved_ids,
            "position_ids": interleaved_pos,
            "block_info": block_info,  # For dynamic FlexAttention mask construction
            "prompt_len": prompt_length,  # Needed for mask construction
            "labels": labels,
            "loss_mask": interleaved_loss_mask,
        }


def collate_interleaved_batch(batch: List[dict], pad_token_id: int, ignore_index: int = -100) -> dict:
    """
    Collate function for interleaved batches.

    Handles variable length sequences by padding.
    Now returns block_info and prompt_len for dynamic FlexAttention mask construction.
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)
    batch_size = len(batch)

    # Initialize padded tensors
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    # Collect block_info and prompt_len for each sample
    block_info_batch = []
    prompt_len_batch = []
    seq_lens = []

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        seq_lens.append(seq_len)

        input_ids[i, :seq_len] = item["input_ids"]
        position_ids[i, :seq_len] = item["position_ids"]
        labels[i, :seq_len] = item["labels"]
        loss_mask[i, :seq_len] = item["loss_mask"]

        # Store block_info and prompt_len
        # Note: block_info contains (seg_type, seg_idx, seg_len)
        # seg_idx is the index in the original interleaved_ids list, not useful after concat
        # We only need (seg_type, seg_len) for mask construction
        block_info_simplified = [(seg_type, seg_len) for seg_type, _, seg_len in item["block_info"]]
        block_info_batch.append(block_info_simplified)
        prompt_len_batch.append(item["prompt_len"])

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "block_info": block_info_batch,  # List of block_info for each sample
        "prompt_len": prompt_len_batch,  # List of prompt lengths
        "seq_lens": seq_lens,  # Actual sequence lengths (before padding)
    }
