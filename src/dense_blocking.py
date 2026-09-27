"""Dense Vector Semantic Indexing & Hybrid Retrieval Blocker.

Combines sparse inverted indexing with dense vector cosine nearest neighbors
using Reciprocal Rank Fusion (RRF) to eliminate the 2-3% candidate recall ceiling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


class HybridDenseBlocker:
    """Hybrid candidate generation fusing sparse C++ candidates and dense vector ANN search."""

    def __init__(self, top_k_dense: int = 16, rrf_k: int = 60):
        self.top_k_dense = top_k_dense
        self.rrf_k = rrf_k

    @staticmethod
    def compute_cosine_candidates(
        anchor_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        anchor_ids: list[str],
        target_ids: list[str],
        top_k: int = 16,
        batch_size: int = 1000,
    ) -> list[tuple[str, str, int, float]]:
        """Compute top-k dense nearest neighbors using batched matrix multiplication.

        Returns list of (anchor_id, target_id, rank, cosine_sim).
        """
        # Ensure L2-normalized embeddings
        a_norms = np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)
        t_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True)
        a_normed = anchor_embeddings / np.maximum(a_norms, 1e-12)
        t_normed = target_embeddings / np.maximum(t_norms, 1e-12)

        results = []
        target_ids_arr = np.array(target_ids)

        for i in range(0, len(anchor_ids), batch_size):
            batch_a = a_normed[i : i + batch_size]
            batch_ids = anchor_ids[i : i + batch_size]

            # (Batch, Num_Targets)
            sim_matrix = np.dot(batch_a, t_normed.T)

            # Extract top-k per anchor
            # Use argpartition for fast top-k
            k = min(top_k, sim_matrix.shape[1])
            top_k_idx = np.argpartition(-sim_matrix, k - 1, axis=1)[:, :k]

            for row_idx, anc_id in enumerate(batch_ids):
                row_sims = sim_matrix[row_idx, top_k_idx[row_idx]]
                sorted_order = np.argsort(-row_sims)
                sorted_targets = target_ids_arr[top_k_idx[row_idx][sorted_order]]
                sorted_scores = row_sims[sorted_order]

                for rank, (tar_id, score) in enumerate(zip(sorted_targets, sorted_scores)):
                    results.append((anc_id, tar_id, rank + 1, float(score)))

        return results

    def fuse_sparse_and_dense(
        self,
        sparse_pairs_df: pd.DataFrame,
        dense_results: list[tuple[str, str, int, float]],
        anchor_col: str = "source1_entity_id",
        target_col: str = "candidate_entity_id",
    ) -> pd.DataFrame:
        """Fuse sparse and dense candidates using Reciprocal Rank Fusion (RRF).

        Args:
            sparse_pairs_df: DataFrame with [source1_entity_id, candidate_entity_id]
            dense_results: List of (anchor_id, target_id, dense_rank, dense_score)

        Returns:
            Deduplicated DataFrame of hybrid candidates with RRF scores.
        """
        dense_df = pd.DataFrame(
            dense_results,
            columns=[anchor_col, target_col, "dense_rank", "dense_score"],
        )

        # Compute sparse ranks by appearance order per anchor
        sparse_df = sparse_pairs_df[[anchor_col, target_col]].copy()
        sparse_df["sparse_rank"] = sparse_df.groupby(anchor_col).cumcount() + 1

        # Full outer join on (anchor, target)
        merged = pd.merge(sparse_df, dense_df, on=[anchor_col, target_col], how="outer")

        # Compute RRF score: 1 / (K + sparse_rank) + 1 / (K + dense_rank)
        sparse_rrf = np.where(
            merged["sparse_rank"].notna(),
            1.0 / (self.rrf_k + merged["sparse_rank"].fillna(1000)),
            0.0,
        )
        dense_rrf = np.where(
            merged["dense_rank"].notna(),
            1.0 / (self.rrf_k + merged["dense_rank"].fillna(1000)),
            0.0,
        )

        merged["rrf_score"] = sparse_rrf + dense_rrf
        merged["in_sparse"] = merged["sparse_rank"].notna()
        merged["in_dense"] = merged["dense_rank"].notna()

        # Sort by anchor and RRF score descending
        merged = merged.sort_values(by=[anchor_col, "rrf_score"], ascending=[True, False])
        return merged.reset_index(drop=True)
