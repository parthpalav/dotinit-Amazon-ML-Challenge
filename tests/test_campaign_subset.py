import numpy as np
import pandas as pd
from src.campaign_raw import subset
from src.experiments import evaluate


def test_explicit_anchor_split_retains_empty_candidate_rows():
    part = {
        'pairs': pd.DataFrame({'source1_entity_id': ['a'], 'label': [1]}),
        'truth_counts': {'a': 1, 'singleton': 0, 'unretrieved': 1, 'outside': 2},
    }
    selected = subset(part, np.array([True]), {'a', 'singleton', 'unretrieved'})
    assert set(selected['truth_counts']) == {'a', 'singleton', 'unretrieved'}
    result = evaluate(selected, np.array([0.9]), 0.5)
    assert np.isclose(result['f0.5'], 2 / 3)
    assert result['false_negative'] == 1
