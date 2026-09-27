# Phase 9 full test inference

Run from the repository root with the local virtual environment:

```bash
.venv/bin/python -u -m src.phase9_inference --workers 4 --batch 500
```

The run verifies every pinned model/dependency hash in `config/phase9_production_lock.json`, the original candidate TSV hash from `reports/improvements/baseline_manifest.json`, and raw test source signatures. It rescored all original candidates; it does not regenerate candidates with the rebuilt native channels.

The preserved original native library provides the original retrieval keys. Posting-aware pair provenance is checked against original native lookup before scoring. The runner also checks exact 81-feature and calibrated-probability parity against the Phase 9 validation cache. Optional preloaded record dictionaries accelerate the existing veto/enhanced routines without changing feature calculations; their previous source files are preserved under `work/phase9_inference/source_archive/`.

Score checkpoints live under `work/phase9_inference/scores_500/`. A restart with identical code, inputs, model and batch size verifies the signature and resumes completed scoring shards. Export requires a fresh output directory, so a completed or interrupted export is never silently replaced. Archive an existing output directory before explicitly restarting the export stage.

Global target-only arbitration selects the highest calibrated probability for each target across **all** test anchors, with original candidate row order breaking exact ties. A second pass computes per-anchor/source margins from the globally arbitrated probabilities. Delta is 0.20 and the probability cutoff is exactly 0.5500000000000002. Candidate files contain all scored candidates, including rejected matches. Every Source 1 entity has one output row, including empty matching rows.

Outputs are `outputs/phase9_submission/matching_results.tsv` and `outputs/phase9_submission/candidate_pairs.tsv`. The earlier files under `outputs/real_submission/` remain intact.

The runner invokes:

```bash
.venv/bin/python vendor/student_resource/utils/validate_submission.py \
  --matching outputs/phase9_submission/matching_results.tsv \
  --candidate outputs/phase9_submission/candidate_pairs.tsv \
  --test-dir ../student_resource/dataset/test \
  --check-ids
```

`progress.json` reports scoring progress. `scoring_complete.json`, `export.json`, `official_validator_command.json`, `official_validator.log`, and finally `completion.json` record completed stages. Only `completion.json` with status complete and validator exit code 0 indicates success.
