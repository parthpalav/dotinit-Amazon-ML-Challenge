"""Select a precision-oriented F0.5 operating point without dropping singletons."""
import numpy as np
import pandas as pd
from .evaluation import evaluate, predictions_from_pairs


def optimize_threshold(pairs, probabilities, truth, target_count, metric="entity_macro"):
    # Include a reject-all point. >= 1 can still select probability-one pairs.
    thresholds = np.concatenate([np.arange(0.10, 0.951, 0.05), [0.99, 1.0, np.nextafter(1.0, 2.0)]])
    rows = []
    for threshold in thresholds:
        predictions = predictions_from_pairs(pairs, probabilities, truth, float(threshold))
        rows.append({"threshold": float(threshold), **evaluate(truth, predictions, target_count, metric)})
    table = pd.DataFrame(rows)
    # Deterministic conservative tie-breaking: precision, then larger threshold.
    best = table.sort_values(["f0.5", "precision", "threshold"], ascending=False, kind="stable").iloc[0]
    return float(best.threshold), table
