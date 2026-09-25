"""Entity-aware metrics, including blocking misses and true singletons."""
from collections import Counter
import numpy as np
import pandas as pd
from .blocking import RULES


def fbeta(precision: float, recall: float) -> float:
    denominator = 0.25 * precision + recall
    return 1.25 * precision * recall / denominator if denominator else 0.0


def evaluate(truth: dict[str, set[str]], predictions: dict[str, set[str]],
             target_count: int, metric: str = "entity_macro") -> dict:
    if set(predictions) != set(truth):
        raise ValueError("Predictions must cover exactly the evaluated Source 1 IDs")
    if not truth:
        raise ValueError("Cannot evaluate an empty split")
    tp = fp = fn = singleton_correct = singleton_count = 0
    precisions, recalls, scores = [], [], []
    for source, actual in truth.items():
        predicted = predictions[source]
        hit = len(actual & predicted)
        tp += hit
        fp += len(predicted - actual)
        fn += len(actual - predicted)
        if not actual and not predicted:
            precision = recall = 1.0
        else:
            precision = hit / len(predicted) if predicted else 0.0
            recall = hit / len(actual) if actual else 0.0
        precisions.append(precision)
        recalls.append(recall)
        scores.append(fbeta(precision, recall))
        if not actual:
            singleton_count += 1
            singleton_correct += not predicted
    micro_p = tp / (tp + fp) if tp + fp else float(fn == 0)
    micro_r = tp / (tp + fn) if tp + fn else float(fp == 0)
    negative_pairs = len(truth) * target_count - sum(map(len, truth.values()))
    return {"precision": float(np.mean(precisions)) if metric == "entity_macro" else micro_p,
            "recall": float(np.mean(recalls)) if metric == "entity_macro" else micro_r,
            "f0.5": float(np.mean(scores)) if metric == "entity_macro" else fbeta(micro_p, micro_r),
            "micro_precision": micro_p, "micro_recall": micro_r, "micro_f0.5": fbeta(micro_p, micro_r),
            "false_positive_rate": fp / negative_pairs if negative_pairs else 0.0,
            "singleton_accuracy": singleton_correct / singleton_count if singleton_count else None,
            "singleton_false_merge_rate": 1 - singleton_correct / singleton_count if singleton_count else None,
            "singleton_count": singleton_count, "true_positive": tp, "false_positive": fp, "false_negative": fn}


def predictions_from_pairs(pairs: pd.DataFrame, probabilities, ids, threshold: float) -> dict[str, set[str]]:
    probabilities = np.asarray(probabilities, dtype=float)
    if len(pairs) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Candidate probabilities must be aligned and finite")
    result = {entity: set() for entity in ids}
    for pair, probability in zip(pairs.itertuples(index=False), probabilities):
        if probability >= threshold:
            result[pair.source1_entity_id].add(pair.candidate_entity_id)
    return result


def candidate_report(pairs: pd.DataFrame, truth: dict, target_count: int) -> dict:
    present = set(zip(pairs.source1_entity_id, pairs.candidate_entity_id))
    positives = {(source, target) for source, ids in truth.items() for target in ids}
    counts = Counter(pairs.source1_entity_id)
    possible = len(truth) * target_count
    return {"candidate_recall": len(present & positives) / len(positives) if positives else None,
            "true_matches": len(positives), "retrieved_true_matches": len(present & positives),
            "candidate_count": len(present), "average_candidates": len(present) / max(1, len(truth)),
            "candidate_reduction_ratio": 1 - len(present) / possible if possible else 1.0,
            "zero_candidate_fraction": sum(counts[x] == 0 for x in truth) / max(1, len(truth))}


def blocking_comparison(pairs, truth, anchors, targets):
    rows = []
    target_countries = dict(zip(targets.entity_id, targets.country_norm))
    country_counts = Counter(c for c in targets.country_norm if c)
    anchor_countries = dict(zip(anchors.entity_id, anchors.country_norm))
    count = sum(country_counts.get(c, 0) for c in anchor_countries.values() if c)
    true_count = sum(map(len, truth.values()))
    hits = sum(bool(anchor_countries[a]) and anchor_countries[a] == target_countries[b]
               for a, ids in truth.items() for b in ids)
    possible = len(anchors) * len(targets)
    rows.append({"strategy": "country_only_analytical", "candidate_recall": hits / true_count if true_count else None,
                 "candidate_count": count, "average_candidates": count / len(anchors),
                 "candidate_reduction_ratio": 1 - count / possible if possible else 1.0,
                 "zero_candidate_fraction": sum(not country_counts.get(c, 0) for c in anchor_countries.values()) / len(anchors)})
    for rule in RULES:
        mask = pairs.blocking_rules.map(lambda value: rule in value.split("|"))
        rows.append({"strategy": rule, **candidate_report(pairs.loc[mask], truth, len(targets))})
    rows.append({"strategy": "combined", **candidate_report(pairs, truth, len(targets))})
    return pd.DataFrame(rows)
