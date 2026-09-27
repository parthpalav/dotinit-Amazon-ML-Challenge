"""Score integrity and leakage boundaries for the local improvement campaign."""
import numpy as np
import pandas as pd
import pytest

from src.local_campaign import (calibration_masks, groups, ownership_probabilities,
                                per_entity_scores, promote)
from src.local_evidence import LocalTransliterator


def test_multilingual_transliteration_preserves_distinguishing_text():
    t = LocalTransliterator()
    try:
        assert t.convert('café saint-étienne') == 'cafe saint-etienne'
        assert t.convert('श्री गणेश') == 'sri ganesa'
        assert t.convert('भारत 123') == 'bharata 123'
        assert t.convert('') == ''
        assert t.convert('abc 123') == 'abc 123'
    finally:
        t.close()


def test_complete_anchor_batches_and_singletons():
    pairs = pd.DataFrame({'anchor_rid': [1, 1, 3, 4, 4, 4, 8]})
    assert list(groups(pairs, 2)) == [(0, 3), (3, 7)]
    with pytest.raises(ValueError):
        list(groups(pairs.iloc[::-1], 2))


def test_macro_f05_accounts_for_unretrieved_truth_and_empty_anchors():
    part = {'truth_counts': {'a': 2, 'b': 0, 'c': 1, 'd': 0},
            'pairs': pd.DataFrame({'source1_entity_id': ['a', 'a', 'b'], 'label': [1, 0, 0]})}
    # One of two true links recovered; false merge for b; absent truth for c;
    # correctly empty d still contributes to the macro average.
    scores = per_entity_scores(part, np.array([.9, .2, .9]), .5)
    np.testing.assert_allclose(scores, [1.25 / 1.5, 0, 0, 1])


def test_ownership_ties_abstain_without_reordering():
    pairs = pd.DataFrame({'candidate_entity_id': ['x', 'y', 'x', 'z', 'z']})
    p = np.array([.8, .7, .9, .95, .95])
    np.testing.assert_array_equal(ownership_probabilities(pairs, p), [0., .7, .9, 0., 0.])


def test_confirmation_floor_requires_confidence_bound_not_just_point_score():
    assert not promote({'f0.5': .96, 'bootstrap_95ci': [.949, .971]}, .95)
    assert promote({'f0.5': .958, 'bootstrap_95ci': [.951, .966]}, .95)


def test_calibration_partition_keeps_whole_anchors_separate():
    anchors = [f'S1-{i}' for i in range(5000)]
    part = {'truth_counts': dict.fromkeys(anchors, 0),
            'pairs': pd.DataFrame({'source1_entity_id': np.repeat(anchors, 2)})}
    early, calibration = calibration_masks(part)
    assert np.all(early ^ calibration)
    assert calibration.sum() == 4000
    assert not set(part['pairs'].loc[early, 'source1_entity_id']) & set(part['pairs'].loc[calibration, 'source1_entity_id'])
