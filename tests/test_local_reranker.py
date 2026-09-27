import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.train_local_reranker import peers_in_batches, split_entities


class LocalRerankerTests(unittest.TestCase):
    def test_entity_partitions_are_disjoint_and_complete(self):
        ids = [f'S1-{i}' for i in range(23)]
        groups = split_entities(ids)
        merged = [x for values in groups.values() for x in values]
        self.assertEqual(set(merged), set(ids))
        self.assertEqual(len(merged), len(set(merged)))
        for key, values in split_entities(ids).items():
            np.testing.assert_array_equal(groups[key], values)

    def test_peer_batches_keep_anchor_groups_and_restore_row_order(self):
        pairs = pd.DataFrame({'anchor_rid': [2, 1, 2]}, index=[8, 4, 6])
        probabilities = np.array([.1, .2, .3])

        def fake_peers(batch, p, con):
            self.assertEqual(batch.anchor_rid.tolist(), [1, 2, 2])
            return pd.DataFrame({'p': p}, index=batch.index)

        with patch('src.train_local_reranker.peer_features', side_effect=fake_peers):
            result = peers_in_batches(pairs, probabilities, None)
        self.assertEqual(result.index.tolist(), [8, 4, 6])
        np.testing.assert_array_equal(result.p.to_numpy(), probabilities)


if __name__ == '__main__':
    unittest.main()
