import numpy as np
import pandas as pd

from src.global_arbitration import arbitrate
from src.phase8_full_retrain import apply_margin_gating
from src.phase9_inference import DTYPE, select_rows, update_owners


def test_global_streaming_stack_matches_validation_with_cross_batch_ties():
    # Anchor 1 loses its best S2 candidate to a later shard. Its remaining S2
    # candidate must be compared to the post-arbitration maximum, not the old one.
    x = np.array([(1, 1, .95), (1, 2, .65), (1, 5, .55),
                  (2, 3, .8), (2, 6, .75),
                  (3, 1, .98), (3, 3, .8), (3, 4, .70),
                  (4, 5, .55), (4, 6, .80)], dtype=DTYPE)
    pairs = pd.DataFrame({'source1_entity_id': [f'S1-{a}' for a in x['anchor']],
        'candidate_entity_id': [f'S{2 if t <= 4 else 3}-{t}' for t in x['target']]})
    threshold = .5500000000000002
    p = np.where(arbitrate(pairs, x['p'], max_per_source=None), x['p'], 0.)
    expected = apply_margin_gating(pairs, p, .2) >= threshold
    best = np.full(7, -np.inf)
    owner = np.full(7, np.iinfo(np.uint64).max, np.uint64)
    update_owners(best, owner, x[:5], 0)
    update_owners(best, owner, x[5:], 5)
    actual = np.r_[select_rows(x[:5], owner, 0, 0, 2, 4, .2, threshold),
                   select_rows(x[5:], owner, 5, 2, 2, 4, .2, threshold)]
    np.testing.assert_array_equal(actual, expected)
    assert actual[1]  # .65 survives after .95 loses global ownership.
    assert actual[3] and not actual[6]  # Stable tie across shards.
    assert not actual[2]  # Exact locked threshold is above .55.
