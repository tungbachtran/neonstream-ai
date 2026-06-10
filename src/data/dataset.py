"""
PyTorch Dataset cho PhoBERT
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
from loguru import logger


class VietnameseToxicDataset(Dataset):
    """
    Dataset class cho PhoBERT toxic/spam detection.
    
    PhoBERT sử dụng BPE tokenizer với vocab tiếng Việt.
    Input: văn bản đã được word-segment
    Output: input_ids, attention_mask, labels
    """

    LABEL_NAMES = {0: "clean", 1: "toxic", 2: "spam", 3: "adult"}
    NUM_LABELS = 4

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        is_training: bool = True
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training

        assert len(texts) == len(labels), "texts và labels phải cùng độ dài"
        logger.info(f"Dataset created: {len(texts)} samples, max_length={max_length}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize với PhoBERT tokenizer
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False  # PhoBERT không dùng token_type_ids
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        text_col: str = 'processed_text',
        label_col: str = 'label',
        is_training: bool = True
    ) -> 'VietnameseToxicDataset':
        """Factory method từ DataFrame"""
        texts = df[text_col].tolist()
        labels = df[label_col].tolist()
        return cls(texts, labels, tokenizer, max_length, is_training)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo DataLoaders cho train/val/test"""

    train_dataset = VietnameseToxicDataset.from_dataframe(
        train_df, tokenizer, max_length, is_training=True
    )
    val_dataset = VietnameseToxicDataset.from_dataframe(
        val_df, tokenizer, max_length, is_training=False
    )
    test_dataset = VietnameseToxicDataset.from_dataframe(
        test_df, tokenizer, max_length, is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"DataLoaders created:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader
