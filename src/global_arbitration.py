"""Global arbitration for record linkage.

Enforces:
1. Target uniqueness: Each target (S2/S3 entity) is claimed by at most ONE anchor (the invariant of the ground truth: truth_pairs has 0 multi-owners).
2. Optional per-source cap: An anchor claims at most one target per source (if max_per_source=1, though note that the challenge dataset contains multiple true fragments per source for ~55% of entities).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Callable


def _default_source_of_target(target_id: str) -> str:
    """Infer 'S2' / 'S3' from the target id prefix, e.g. 'S2-380646438' -> 'S2'."""
    return target_id[:2]


def arbitrate(
    pairs: pd.DataFrame,
    prob: np.ndarray,
    anchor_col: str = "source1_entity_id",
    target_col: str = "candidate_entity_id",
    source_of: Optional[Callable[[str], str]] = None,
    max_per_source: Optional[int] = None,
) -> np.ndarray:
    """Greedy maximum-weight bipartite arbitration.

    By default (max_per_source=None), enforces target uniqueness:
        - each target is claimed by at most one anchor (the true ground-truth invariant).
        - anchors can claim multiple targets if they are legitimate distinct fragments.

    If max_per_source=1:
        - also restricts each anchor to at most one target per source.
    """
    if source_of is None:
        source_of = _default_source_of_target

    n = len(pairs)
    anchors = pairs[anchor_col].to_numpy()
    targets = pairs[target_col].to_numpy()
    sources = np.array([source_of(t) for t in targets]) if max_per_source is not None else None

    # Sort descending by probability. Stable sort preserves deterministic tie-breaking.
    order = np.argsort(-prob, kind="stable")

    claimed_anchor_source_counts: dict[tuple, int] = {}
    claimed_target: set = set()
    won = np.zeros(n, dtype=bool)

    for idx in order:
        a = anchors[idx]
        t = targets[idx]

        if t in claimed_target:
            continue

        if max_per_source is not None:
            s = sources[idx]
            akey = (a, s)
            current_count = claimed_anchor_source_counts.get(akey, 0)
            if current_count >= max_per_source:
                continue
            claimed_anchor_source_counts[akey] = current_count + 1

        claimed_target.add(t)
        won[idx] = True

    return won


def to_entity_level(
    pairs: pd.DataFrame,
    prob: np.ndarray,
    won: np.ndarray,
    anchor_col: str = "source1_entity_id",
) -> pd.DataFrame:
    """Collapse arbitrated pairs down to winning rows with probabilities."""
    winners = pairs.loc[won].copy()
    winners["prob"] = prob[won]
    return winners


def arbitrate_by_component(
    pairs: pd.DataFrame,
    prob: np.ndarray,
    anchor_col: str = "source1_entity_id",
    target_col: str = "candidate_entity_id",
    source_of: Optional[Callable[[str], str]] = None,
    max_per_source: Optional[int] = None,
) -> np.ndarray:
    """Scaling helper: arbitrate within connected components of the anchor-target graph."""
    import networkx as nx

    if source_of is None:
        source_of = _default_source_of_target

    won = np.zeros(len(pairs), dtype=bool)
    sources = pairs[target_col].map(source_of)
    
    for src_val, sub_idx in pairs.groupby(sources).groups.items():
        sub = pairs.loc[sub_idx]
        g = nx.Graph()
        g.add_edges_from(zip(sub[anchor_col], sub[target_col]))
        for component in nx.connected_components(g):
            comp_mask = sub[anchor_col].isin(component) | sub[target_col].isin(component)
            comp_pairs = sub.loc[comp_mask]
            comp_prob = prob[sub.index.get_indexer(comp_pairs.index)]
            comp_won = arbitrate(
                comp_pairs, comp_prob, anchor_col, target_col, source_of, max_per_source=max_per_source
            )
            won[pairs.index.get_indexer(comp_pairs.index)] |= comp_won

    return won
