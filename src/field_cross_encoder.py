"""Field-Disentangled Multi-Field Cross-Attention Architecture for Business Entity Resolution.

Separates Name, Address, Legal Form, and Geography into distinct interaction layers:
1. Name: ColBERT-style MaxSim token-level late interaction.
2. Address: Segment-aware alignment and sequence cross-attention.
3. Legal Form: Categorical conflict gating.
4. Country: Geographic compatibility projection.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


LEGAL_FORMS = [
    'none', 'private', 'limited', 'incorporated', 'corporation', 'company',
    'llp', 'llc', 'plc', 'sa', 'sarl', 'sas', 'gmbh', 'pvt', 'ltd', 'inc', 'corp'
]
LEGAL_MAP = {k: i for i, k in enumerate(LEGAL_FORMS)}


class MaxSimInteraction(nn.Module):
    """Computes ColBERT-style token-level maximum cosine similarity between sequence pairs."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        anchor_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        anchor_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MaxSim score vector.

        Args:
            anchor_embeds: (B, L_a, D)
            target_embeds: (B, L_t, D)
            anchor_mask: (B, L_a) boolean mask
            target_mask: (B, L_t) boolean mask

        Returns:
            (B, 3) tensor: [mean_maxsim, min_maxsim, soft_coverage]
        """
        # Normalize embeddings for cosine similarity
        a_norm = F.normalize(anchor_embeds, p=2, dim=-1)  # (B, L_a, D)
        t_norm = F.normalize(target_embeds, p=2, dim=-1)  # (B, L_t, D)

        # Pairwise token similarity matrix: (B, L_a, L_t)
        sim_matrix = torch.bmm(a_norm, t_norm.transpose(1, 2))

        # Mask invalid target tokens
        mask_t = target_mask.unsqueeze(1)  # (B, 1, L_t)
        sim_matrix = sim_matrix.masked_fill(~mask_t, -1e4)

        # For each anchor token, find the maximum similarity with any target token
        max_sim, _ = sim_matrix.max(dim=-1)  # (B, L_a)

        # Mask invalid anchor tokens
        max_sim = max_sim.masked_fill(~anchor_mask, 0.0)
        anchor_lens = anchor_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()  # (B, 1)

        mean_max = max_sim.sum(dim=-1, keepdim=True) / anchor_lens  # (B, 1)
        
        # Soft min across valid anchor tokens
        min_sim = max_sim.masked_fill(~anchor_mask, 1e4).min(dim=-1, keepdim=True)[0]
        min_sim = torch.where(min_sim > 1e3, torch.zeros_like(min_sim), min_sim)

        # Token coverage (fraction of anchor tokens with similarity > 0.75)
        high_sim_count = ((max_sim > 0.75) & anchor_mask).sum(dim=-1, keepdim=True).float()
        coverage = high_sim_count / anchor_lens

        return torch.cat([mean_max, min_sim, coverage], dim=-1)  # (B, 3)


class DirectCrossAttention(nn.Module):
    """Pure tensor-based multi-head cross-attention without external symbolic dependencies."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """Cross-attention from query to key/value.

        Args:
            query: (B, L_q, D)
            key: (B, L_k, D)
            value: (B, L_k, D)
            key_mask: (B, L_k) boolean mask (True = valid token)
        """
        B, L_q, _ = query.shape
        L_k = key.shape[1]

        q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_q, d)
        k = self.k_proj(key).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, L_k, d)
        v = self.v_proj(value).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_k, d)

        # (B, H, L_q, L_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Mask invalid key tokens
        mask = key_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_k)
        scores = scores.masked_fill(~mask, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        # (B, H, L_q, d)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.dim)
        return self.out_proj(context)


class FieldDisentangledCrossEncoder(nn.Module):
    """Field-aware neural cross-encoder with specialized field interaction sub-modules."""

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_legal_forms: int = len(LEGAL_FORMS),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.legal_embedding = nn.Embedding(num_legal_forms, 16)
        
        # Name sub-encoder & MaxSim interaction
        self.name_encoder = nn.GRU(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.name_maxsim = MaxSimInteraction(embed_dim)

        # Address sub-encoder & cross-attention
        self.addr_encoder = nn.GRU(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.addr_cross_attn = DirectCrossAttention(embed_dim, num_heads=4)

        # Legal interaction bilinear layer
        self.legal_bilinear = nn.Bilinear(16, 16, 8)

        # Feature projection & classifier
        # Inputs: Name MaxSim (3) + Addr Pooled (embed_dim*2) + Legal (8) + Country Match (1)
        input_feature_dim = 3 + (embed_dim * 2) + 8 + 1
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        anchor_name_ids: torch.Tensor,
        target_name_ids: torch.Tensor,
        anchor_addr_ids: torch.Tensor,
        target_addr_ids: torch.Tensor,
        anchor_legal: torch.Tensor,
        target_legal: torch.Tensor,
        country_match: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for field-disentangled pair scoring.

        Args:
            anchor_name_ids: (B, L_an) token IDs
            target_name_ids: (B, L_tn) token IDs
            anchor_addr_ids: (B, L_aa) token IDs
            target_addr_ids: (B, L_ta) token IDs
            anchor_legal: (B,) legal form IDs
            target_legal: (B,) legal form IDs
            country_match: (B, 1) binary float tensor

        Returns:
            (B, 1) logits tensor.
        """
        # 1. Name Encoding & Token-Level MaxSim Interaction
        a_n_mask = anchor_name_ids != 0
        t_n_mask = target_name_ids != 0
        a_n_emb = self.token_embedding(anchor_name_ids)
        t_n_emb = self.token_embedding(target_name_ids)

        a_n_out, _ = self.name_encoder(a_n_emb)
        t_n_out, _ = self.name_encoder(t_n_emb)

        name_scores = self.name_maxsim(a_n_out, t_n_out, a_n_mask, t_n_mask)  # (B, 3)

        # 2. Address Encoding & Cross-Attention Interaction
        a_a_mask = anchor_addr_ids != 0
        t_a_mask = target_addr_ids != 0
        a_a_emb = self.token_embedding(anchor_addr_ids)
        t_a_emb = self.token_embedding(target_addr_ids)

        a_a_out, _ = self.addr_encoder(a_a_emb)
        t_a_out, _ = self.addr_encoder(t_a_emb)

        # Cross attention from Anchor Address to Target Address
        attn_out = self.addr_cross_attn(
            query=a_a_out,
            key=t_a_out,
            value=t_a_out,
            key_mask=t_a_mask,
        )
        # Mask and pool address representations
        a_a_pooled = (a_a_out * a_a_mask.unsqueeze(-1)).sum(dim=1) / a_a_mask.sum(dim=1, keepdim=True).clamp(min=1)
        attn_pooled = (attn_out * a_a_mask.unsqueeze(-1)).sum(dim=1) / a_a_mask.sum(dim=1, keepdim=True).clamp(min=1)
        addr_features = torch.cat([a_a_pooled, attn_pooled], dim=-1)  # (B, embed_dim*2)

        # 3. Legal Form Bilinear Interaction
        a_l_emb = self.legal_embedding(anchor_legal)
        t_l_emb = self.legal_embedding(target_legal)
        legal_features = self.legal_bilinear(a_l_emb, t_l_emb)  # (B, 8)

        # 4. Joint Feature Fusion & Logit Computation
        combined = torch.cat([name_scores, addr_features, legal_features, country_match], dim=-1)
        logits = self.classifier(combined)  # (B, 1)
        return logits
