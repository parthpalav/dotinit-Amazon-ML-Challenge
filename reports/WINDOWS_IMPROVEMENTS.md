# Windows improvement runbook — 2026-09-26

Run commands from `C:\Users\parth\Desktop\CS\amazon\.init-Amazon-ML-Challenge`.
Use the local `.venv\Scripts\python.exe`; do not use the older Mac environment. The environment contains CPU scientific libraries and GPU-enabled CatBoost. A C++17 compiler is needed for native retrieval (currently `C:\msys64\ucrt64\bin\g++.exe`). Large caches use read-only Python memory maps on Windows, Linux and macOS.

## What exists

- `dataset/`: supplied training/test records; all seven original input hashes verified.
- `src/real_pipeline.py`: original disk retrieval/ranking/forest pipeline.
- `work/real_v1/`: original indexes and sampled feature caches.
- `work/windows_v1/`: verified portable copies, with hard links for immutable large files. Do not modify linked data files in place.
- `src/evidence.py`: 55 additional token, number-conflict, accent-folding and corpus-frequency features, built only from supplied text.
- `src/experiments.py`: GPU CatBoost comparisons using 30,000 fitting, 5,000 calibration and 10,000 selection anchors.
- `src/confirmation.py`, `src/confirmation_evaluation.py`: 5,000 fresh anchors excluded from the above groups. Model/threshold fixed before evaluation.
- `src/rescoring.py`: checkpointed rescoring of existing candidates; original retained pairs/provenance are preserved.
- `src/finalize_improved.py`: separate TSV export with complete streaming validation; optional unique-target ownership and abstention on exact-score ties.
- `reports/improvements/`: machine-readable evidence, logs, model-selection lock and confirmation results.

## Results so far

| Model | Selection macro F0.5 | Threshold |
|---|---:|---:|
| Original calibrated forest | 0.875670 | 0.550 |
| CatBoost depth 8, original 58 features | 0.898539 | 0.650 |
| CatBoost depth 10, original 58 features | 0.899889 | 0.600 |
| CatBoost depth 8, 113 features | 0.943009 | 0.625 |
| Frozen CatBoost depth 10, 113 features | 0.943147 | 0.675 |

The original Amazon score is **0.842**, reported by the team. No new Amazon score is available. These local figures are selection results, not untouched evaluation results. The frozen model passed fresh 5,000-anchor confirmation: **0.943215 vs 0.878801** baseline; paired improvement 95% bootstrap interval **[0.058966, 0.069730]**. See `improvements/confirmation_evaluation.json`. Training has US/India labels; France generalization remains unmeasured. Country-specific unlabeled name statistics are transductive features, not external lookups.

The original submission has 111,450 targets assigned to multiple S1 records, including 296,242 excess assignments. France accounts for 197,983 excess assignments. The supplied problem states S1 is deduplicated. Unique ownership is therefore a useful separate variant, but its score impact is not established by small random validation samples.

The original raw candidate stage lost 714 of 34,676 true links on selection; the 32-candidate cap lost another 70. Most baseline losses occurred in matching. Keeping the candidate stage unchanged isolates the measured matcher improvement and avoids an expensive full ranker rerun. Further retrieval changes need independent candidate-recall evidence and regenerated caches.

## Resume in order

Do not run multiple large feature jobs concurrently on 16 GB RAM. Existing verified preparation should be reused.

```powershell
.venv/Scripts/python.exe -u -m src.confirmation
.venv/Scripts/python.exe -u -m src.evidence --splits confirmation
.venv/Scripts/python.exe -u -m src.confirmation_evaluation
```

Only after the frozen model passes confirmation, smoke-test rescoring and then resume the same directory without the limit:

```powershell
.venv/Scripts/python.exe -u -m src.rescoring --model artifacts/improvements/catboost_d10_evidence.joblib --work work/improvements/test_scores_d10_direct --workers 3 --direct-provenance --limit 1000
.venv/Scripts/python.exe -u -m src.rescoring --model artifacts/improvements/catboost_d10_evidence.joblib --work work/improvements/test_scores_d10_direct --workers 3 --direct-provenance
.venv/Scripts/python.exe -m src.finalize_improved --work work/improvements/test_scores_d10_direct --output outputs/improved_pair_threshold
.venv/Scripts/python.exe -m src.finalize_improved --work work/improvements/test_scores_d10_direct --output outputs/improved_unique_owner --unique-owner
```

A model/code/candidate signature prevents mixing score checkpoints from different runs. Output directories must be empty. Completion requires all 1,732,544 anchors; the exporter compares every scored candidate with the original TSV, validates target IDs against the supplied-data index, includes empty rows, and checks matching subsets. `validation.json` records counts and hashes. Run the supplied official validator as an additional check before submission; its full ID-set mode can require substantial RAM. No candidate-only benchmark is a complete submission.

## Preservation and verification

Baseline model and TSV files remain at `artifacts/real_model.joblib` and `outputs/real_submission/`. Their original hashes are in `improvements/baseline_manifest.json`. New models live under `artifacts/improvements/`.

All 41 full-suite tests passed in `improvements/final_tests.log`; two export regression tests passed in `improvements/export_tests.log`. The Windows migration matched all 3,187 candidates, provenance and labels for 100 baseline validation anchors. Original XGBoost snapshots are loaded only by an explicit trusted-artifact compatibility path; do not load untrusted pickle/joblib files.

RTX 4060 training takes roughly a minute per CatBoost experiment. CPU feature preparation and random disk access dominate runtime. A5000 is inaccessible; Colab T4 is not needed for this stage. Reconsider remote GPUs only if a measured multilingual reranker experiment justifies its memory and transfer costs.


## Confirmed fast inference and automatic continuation
Exact posting-aware key replay matches all 31,973 predictions on the first 1,000 test anchors, bit for bit. Native lookup took 184.8 seconds; direct replay took 21.5 seconds. The subsequent three-worker benchmark scored 3,000 more anchors in 18.4 seconds without large index mappings. See `improvements/provenance_parity.json`. Two additional posting-limit/chunk-boundary tests passed.

Resume the complete inference/export/validation sequence with:

```powershell
.venv/Scripts/python.exe -u -m src.complete_improvements
```

It runs three bounded scoring workers, uses the `test_scores_d10_direct` checkpoints, exports both variants, and runs the supplied official validator with matching-ID checking. Candidates are exhaustively checked by the streaming exporter; they are intentionally omitted from the official checker to avoid materializing 55 million strings on 16 GB RAM. This distinction is recorded in completion status. Watch `improvements/completion_status.json`, `improvements/full_rescoring.log`, and `work/improvements/test_scores_d10_direct/progress.json`. `complete` status is required before treating the new outputs as finished. Keep the machine awake while this long job runs. Full inference has not yet established a new Amazon score.
