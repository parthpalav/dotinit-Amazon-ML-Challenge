import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.local_training_features import fit_stats, frame_signature, require_signature


class EnrichedTrainingTests(unittest.TestCase):
    def test_fit_statistics_only_fetch_fitting_anchors(self):
        pairs = pd.DataFrame({'anchor_rid': [7, 7, 9]})
        records = pd.DataFrame({'name_norm': ['alpha tools', 'alpha shop'], 'country_norm': ['us', 'us']})
        with patch('src.local_training_features.fetch_records', return_value=records) as fetch:
            stats = fit_stats(pairs, object())
        self.assertEqual(fetch.call_args.args[1], 'anchors')
        self.assertEqual(fetch.call_args.args[2].tolist(), [7, 9])
        self.assertEqual(stats['df'][('us', 'alpha')], 2)
        self.assertEqual(stats['sizes'], {'us': 2})

    def test_reordered_candidates_invalidate_signature(self):
        pairs = pd.DataFrame({'anchor_rid': [1, 1], 'target_rid': [2, 3]})
        self.assertNotEqual(frame_signature(pairs), frame_signature(pairs.iloc[::-1]))

    def test_cache_recipe_change_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'signature.json'
            require_signature(path, {'pairs': 'original'})
            with self.assertRaisesRegex(ValueError, 'changed'):
                require_signature(path, {'pairs': 'new'})
            self.assertEqual(json.loads(path.read_text()), {'pairs': 'original'})


if __name__ == '__main__':
    unittest.main()
