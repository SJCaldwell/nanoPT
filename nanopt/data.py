import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_from_disk
from typing import Optional, Dict, List
import numpy as np

class FineWebEduDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer_name: str = "meta-llama/Llama-3.2-1B",
        split: str = "train",
        max_length: int = 4096,
    ):
        self.dataset = load_from_disk(dataset_path)[split]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)
        self.max_length = max_length

        print(f"Loaded {len(self.dataset)} documents for {split} split")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tokens["input_ids"].squeeze(0) # remove batch dimension
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(), #these are shifted internally, so SHOULD NOT shift here
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that packs sequences to the same length.
    Pads to the max length in the batch (not global max_length).
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    batch_input_ids = []
    batch_labels = []

    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]

        # Pad if necessary
        padding_length = max_len - input_ids.size(0)
        if padding_length > 0:
            # pad with tokenizer's pad token
            pad_id = 0
            input_ids = torch.cat([input_ids, torch.full((padding_length,), pad_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)]) # -100 is the ignore index

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
    
    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
    }

def get_dataloaders(
    dataset_path: str,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    batch_size: int = 16,
    max_length: int = 4096,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = FineWebEduDataset(
        dataset_path=dataset_path,
        tokenizer_name=tokenizer_name,
        split="train",
        max_length=max_length,
    )

    val_dataset = FineWebEduDataset(
        dataset_path=dataset_path,
        tokenizer_name=tokenizer_name,
        split="validation",
        max_length=max_length,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader