"""Small input-contract tests; no data audits or training runs."""
from pathlib import Path
import tempfile
import unittest

from src.train_local import check_split_ids, required_inputs


class LocalTrainingInputs(unittest.TestCase):
    def test_disjoint_entities(self):
        check_split_ids({'fit': ['S1-a'], 'calibration': ['S1-b'], 'validation': ['S1-c']})

    def test_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_split_ids({'fit': ['S1-a'], 'calibration': ['S1-b'], 'validation': ['S1-a']})

    def test_empty_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            check_split_ids({'fit': ['S1-a'], 'calibration': [], 'validation': ['S1-c']})

    def test_missing_inputs_do_not_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, 'fit.joblib'):
                required_inputs(work)
            self.assertEqual(list(work.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
