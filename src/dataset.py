import torch
import re
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm


def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    answer_ids = torch.stack([item['answer_ids'] for item in batch])
    
    original_instruction = [item['original_instruction'] for item in batch]
    original_context = [item['original_context'] for item in batch]
    original_response = [item['original_response'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'answer_ids': answer_ids,
        'original_instruction': original_instruction,
        'original_context': original_context,
        'original_response': original_response
    }

class WikiTextDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, max_seq_len=512, max_answer_len=64, split='train'):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_answer_len = max_answer_len
        self.split = split
        # Load WikiText-103 in streaming mode
        self.dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split, streaming=True)

    def __iter__(self):
        for example in self.dataset:
            text = example['text'].strip()
            if len(text) < 50: # Skip short lines/headers
                continue
            
            # Tokenize the whole text (Disable truncation to handle splitting manually)
            tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            
            # If text is too short for a meaningful split, skip
            if len(tokens) < 10:
                continue
                
            # Split into chunks: Input (max_seq_len) + Target (max_answer_len)
            chunk_size = self.max_seq_len + self.max_answer_len

            # Sliding window or just non-overlapping chunks
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i : i + chunk_size]
                if len(chunk) < self.max_seq_len + 10: continue

                # Split: First max_seq_len tokens = Input, Next max_answer_len tokens = Target
                input_ids = chunk[:self.max_seq_len]
                target_ids = chunk[self.max_seq_len:self.max_seq_len + self.max_answer_len]

                # Skip if target is too short
                if len(target_ids) < 10:
                    continue

                # Pad and convert to tensors
                # We manually handle padding here since we are yielding tensors

                def pad_and_tensor(ids, max_len):
                    ids = ids[:max_len] # Truncate
                    mask = [1] * len(ids) + [0] * (max_len - len(ids))
                    ids = ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
                    return torch.tensor(ids), torch.tensor(mask)

                input_tensor, mask_tensor = pad_and_tensor(input_ids, self.max_seq_len)
                target_tensor, _ = pad_and_tensor(target_ids, self.max_answer_len)
                
                yield {
                    'input_ids': input_tensor,
                    'attention_mask': mask_tensor,
                    'answer_ids': target_tensor,
                    'original_instruction': "Continue the text:", # Dummy instruction
                    'original_context': self.tokenizer.decode(input_ids),
                    'original_response': self.tokenizer.decode(target_ids)
                }

    def __len__(self):
        # Approximate length for progress bars (WikiText-103 train has ~1.8M lines, but we filter)
        if self.split == 'train': return 1000000 
        return 20000

def get_dataloader(dataset_type, tokenizer, max_seq_len, max_answer_len, batch_size, split, shuffle=True, cache_dir='cached_datasets'):
    if dataset_type == 'wikitext':
        # Streaming dataset doesn't support shuffle=True in standard DataLoader in the same way
        # We rely on the dataset being naturally varied or add a shuffle buffer if needed.
        # For simplicity, we just iterate.
        dataset = WikiTextDataset(tokenizer, max_seq_len, max_answer_len, split)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Only 'wikitext' is supported.")

    return dataloader