# Verification

The current real-data run uses macOS arm64, Python 3.13 and the versions in `requirements-tested.txt`.

- Original suite: **23 passed**, after installing the missing macOS OpenMP runtime. Exact output: `reports/original_tests.log`.
- Early real-data integration checks: **11 passed**, 4 stage-dependent checks deselected. Exact output: `reports/early_real_tests.log`.
- Final full-suite results are recorded in `reports/final_tests.log` after training, inference and official validation complete. Do not interpret the early subset as final validation.
- The original tests are retained. Additional real-data tests cover all six source schemas, labels, audit integrity, candidate labels/recall, missing-address features, Unicode/contact normalization, nullable preprocessing, singleton metric parity, the unmodified official validator on actual subsets, and complete final TSV coverage/uniqueness/subset/count consistency.

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -q
.venv/bin/python -m pytest tests/test_real_data.py -q
.venv/bin/python -m pytest -q > reports/final_tests.log
.venv/bin/python -m compileall -q src utils tests
.venv/bin/python -m src.real_pipeline validate --config config/real.json
```

The official validator is `../student_resource/utils/validate_submission.py`; the repository's `utils/validate_submission.py` is the separate legacy local validator. Full official results belong in `reports/official_validator.log` and `reports/official_validator_command.json`.

Synthetic fixtures are isolated under pytest temporary directories. No synthetic input is used for the real model or submissions. Optional embeddings remain disabled and untested.
