"""Domain Adaptation & Self-Training Pipeline for Unlabelled Test Geographies (e.g. France).

Implements:
1. Domain-Adaptive Pre-Training (DAPT) with Masked Language Modeling on test entities.
2. High-confidence pseudo-label extraction and iterative refinement.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional


class MaskedEntityDataset(Dataset):
    """Prepares unlabelled entity texts with dynamic token masking for MLM adaptation."""

    def __init__(
        self,
        token_ids_list: List[List[int]],
        mask_token_id: int,
        vocab_size: int,
        pad_token_id: int = 0,
        mask_prob: float = 0.15,
        max_len: int = 128,
    ):
        self.samples = token_ids_list
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_prob = mask_prob
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = list(self.samples[idx][: self.max_len])
        length = len(tokens)
        
        # Pad to max_len
        if length < self.max_len:
            tokens = tokens + [self.pad_token_id] * (self.max_len - length)

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.full((self.max_len,), -100, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()

        # Apply dynamic masking on valid tokens
        for pos in range(length):
            if torch.rand(1).item() < self.mask_prob:
                labels[pos] = input_ids[pos]
                prob = torch.rand(1).item()
                if prob < 0.8:
                    input_ids[pos] = self.mask_token_id
                elif prob < 0.9:
                    input_ids[pos] = torch.randint(1, self.vocab_size, (1,)).item()
                # Remaining 10% keeps original token

        return input_ids, attention_mask, labels


class PseudoLabelExtractor:
    """Extracts ultra-high-confidence candidate pairs from test predictions for self-training."""

    def __init__(self, confidence_threshold: float = 0.99, max_per_anchor: int = 1):
        self.thresh = confidence_threshold
        self.max_per_anchor = max_per_anchor

    def extract_pseudo_labels(
        self,
        test_pairs: list[tuple[str, str, float, str]],
    ) -> list[tuple[str, str, int, str]]:
        """Filter high-precision pseudo positive links from unlabelled test predictions.

        Args:
            test_pairs: List of (anchor_id, target_id, model_probability, country)

        Returns:
            List of (anchor_id, target_id, pseudo_label=1, country)
        """
        # Sort descending by probability
        sorted_pairs = sorted(test_pairs, key=lambda x: -x[2])
        
        claimed_targets = set()
        anchor_claims = {}
        pseudo_labels = []

        for anc_id, tar_id, prob, country in sorted_pairs:
            if prob < self.thresh:
                break
            if tar_id in claimed_targets:
                continue

            current_count = anchor_claims.get(anc_id, 0)
            if current_count >= self.max_per_anchor:
                continue

            claimed_targets.add(tar_id)
            anchor_claims[anc_id] = current_count + 1
            pseudo_labels.append((anc_id, tar_id, 1, country))

        return pseudo_labels
