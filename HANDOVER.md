> Historical baseline handover. Current work and live continuation status are documented in [reports/IMPROVEMENT_STATUS.md](reports/IMPROVEMENT_STATUS.md) and [the Windows runbook](reports/WINDOWS_IMPROVEMENTS.md).

# Session handover — paused 2026-09-25

## Current status

The training audit, its four workers, its multiprocessing resource tracker, and the scoped `caffeinate` process have been terminated. Process cleanup confirmed no selected PIDs remained. No further audit, training, inference, or tests were started during handover.

All source edits, configuration changes, and parity tests are saved on disk. The repository root is:

```text
/Users/parthspalav/Documents/College/Hackathons/Amazon ML Challenge/dotinit-Amazon-ML-Challenge-main
```

Completed:

- Located the real data in `../student_resource`; all seven raw TSV hashes match the previous session's inventory. Raw files are unchanged.
- Completed the data-quality scan and both training/test disk stores, base indexes, supplemental indexes, and packed-text caches.
- Trained/evaluated all four existing baselines on real candidate pairs, with disjoint fitting, calibration, and validation Source 1 samples.
- Saved the selected model, calibrator, threshold, feature engineer, retrieval ranker, and configuration together.
- Latest pre-export test run: **38 passed, 1 deselected**, including all **23 original tests**. See `reports/pre_final_tests.log`. The deselected test requires completed final submissions and official validation.
- Verified grouped/threaded retrieval scores are bit-for-bit equal to the original comparisons on the checked real candidates. Verified single-thread forest scoring preserves decisions on 10,000 real validation candidates (maximum probability difference `4.440892098500626e-16`).

Pending:

- The full training blocking audit was interrupted. Last progress observed before handover was **1,300,000 / 2,206,821 anchors**, with approximately **97.59%** recall. This is partial progress, not a completed full-audit result.
- Full test inference, final TSV exports, and the final official validator run have **not** completed.
- The final export-dependent test and completed experiment report remain pending. `src/real_report.py` requires the completed full-audit report as well as inference/validation results.

**Inference and official validation do not require completion of the training audit. Use the `infer` command below, not `run`, to avoid restarting training and the full audit.**

## Selected model and configuration

**Calibrated Random Forest; threshold 0.55.** The artifact retains the exact floating-point threshold (`0.5500000000000002`); inference reads it automatically.

Validation results (10,000 held-out Source 1 anchors; selection metrics, not a competition test score):

| Model | Macro precision | Macro recall | Macro F0.5 | Macro F1 | Candidate PR-AUC |
|---|---:|---:|---:|---:|---:|
| Exact rule | 0.153900 | 0.084190 | 0.119708 | 0.099038 | 0.136713 |
| Calibrated logistic regression | 0.858712 | 0.718621 | 0.803431 | 0.756609 | 0.917983 |
| **Calibrated random forest** | **0.908929** | **0.824108** | **0.875670** | **0.846029** | **0.957651** |
| Calibrated XGBoost | 0.901853 | 0.814129 | 0.867307 | 0.836769 | 0.953357 |

`config/real.json` currently specifies:

- Disk backend; seed 42.
- Four process workers, 100 anchors per batch, four model-training threads.
- Retrieval posting limit 240; final candidate cap 32 per anchor.
- 30,000 fitting, 5,000 calibration, and 10,000 validation anchors; every final candidate for those anchors retained.
- Fitting data: 959,871 candidate pairs, comprising 101,605 positives and 858,266 negatives.
- Held-out validation blocking recall: **97.7391%**, below the 98% target. This limit is explicitly reported; low-recall hard failure is disabled.
- New inference workers use grouped comparisons with at most two similarity-scoring threads per worker, and one forest-estimator thread per worker. Parity checks cover these computational changes.

## Saved artifacts

Paths below are relative to the repository root:

| Path | Contents |
|---|---|
| `artifacts/real_model.joblib` | **Final trained model artifact**, approximately 484 MiB; includes calibrated forest, exact threshold, 58-feature engineer/schema, retrieval ranker, and training configuration |
| `work/real_v1/candidate_ranker.joblib` | Fitting-only retrieval ranker |
| `work/real_v1/feature_engineer.joblib` | Training-fitted TF-IDF feature engineer |
| `work/real_v1/entity_selection.npz` | Deterministic disjoint entity samples |
| `work/real_v1/features/{fit,calibration,validation}.joblib` | Real candidate pairs and final features |
| `work/real_v1/train/`, `work/real_v1/test/` | Completed SQLite stores, indexes, manifests, and packed text |
| `reports/real_run_manifest.json` | Selected model/configuration, versions, features, and measured training results |
| `reports/model_comparison.tsv` | All four model results |
| `reports/calibration_comparison.tsv`, `reports/thresholds_*.tsv` | Calibration diagnostics and threshold sweeps |
| `reports/real_data_quality_report.md`, `reports/real_data_quality.json` | Fresh data-quality results and input inventory |
| `reports/real_pipeline.log` | Training and interrupted full-audit progress; wall time includes long pauses |
| `reports/pre_final_tests.log` | Latest 38 passing pre-export tests |
| `reports/grouped_similarity_benchmark.json`, `reports/threaded_similarity_benchmark.json`, `reports/inference_thread_parity.json` | Optimization/parity evidence |
| `reports/prior_run/` | Historical reports copied from the earlier session; distinguish these from current-run evidence |

The local `.venv` is installed and working. The macOS OpenMP runtime needed by XGBoost was installed. Do not delete the model or caches before inference, and do not change retrieval/preprocessing settings while reusing these artifacts.

## Execute full test inference — one command

Run this exact single shell command:

```bash
cd '/Users/parthspalav/Documents/College/Hackathons/Amazon ML Challenge/dotinit-Amazon-ML-Challenge-main' && .venv/bin/python -m src.real_pipeline infer --config config/real.json
```

This loads the saved artifact, scores candidates for **all 1,732,544 test Source 1 records** against both test reference sources (including France), applies calibration and the stored threshold, and exports:

```text
/Users/parthspalav/Documents/College/Hackathons/Amazon ML Challenge/dotinit-Amazon-ML-Challenge-main/outputs/real_submission/matching_results.tsv
/Users/parthspalav/Documents/College/Hackathons/Amazon ML Challenge/dotinit-Amazon-ML-Challenge-main/outputs/real_submission/candidate_pairs.tsv
```

There is one matching task across three sources, so these are the two required files—not three independent submissions. Each has one row per test Source 1 record. Empty matches remain empty; matches are never forced. The candidate file contains the exact final candidate set scored by the matcher.

Inference writes `.partial` files until successful completion, then renames them to the final filenames and writes `reports/test_inference.json`. This implementation does not resume an interrupted inference from partial output; rerunning inference starts its export again. Allow substantial compute time and keep the machine awake while it runs.

## Run the final official submission validator — one command

After inference completes successfully:

```bash
cd '/Users/parthspalav/Documents/College/Hackathons/Amazon ML Challenge/dotinit-Amazon-ML-Challenge-main' && .venv/bin/python -m src.real_pipeline validate --config config/real.json
```

This wrapper invokes the **unmodified supplied** `../student_resource/utils/validate_submission.py` with both output TSVs, `../student_resource/dataset/test`, and `--check-ids`. It does not use the repository's separate legacy local validator. It saves the complete output to `reports/official_validator.log` and the command, exit code, and validator SHA-256 to `reports/official_validator_command.json`. A nonzero validator exit fails the command.

**No final official PASS has been claimed or obtained yet.** Earlier validator checks were limited to small real-data subsets used by integration tests.

## Saved implementation changes

- `config/real.json`: local resource/cache paths and bounded execution settings.
- `src/real_pipeline.py`: bounded task queue, single-thread estimator inference workers, recall warnings/gating for subsequent training runs, and report generation after a complete run.
- `src/coarse_ranker.py`: equivalent grouped comparisons and optional bounded parallel scoring.
- `src/disk_blocking.py`: use at most two similarity-scoring threads, respecting the configured thread limit.
- `src/preprocessing.py`: safe contact extraction from nullable fields.
- `src/real_report.py`: measured-artifact-based final report generation, including pilot provenance and recall/runtime limitations.
- `src/main.py`: final report generation for the disk backend's complete run.
- `tests/test_real_data.py`: null handling, feature parity, recall regression, and stronger streaming final-export checks.
- `pyproject.toml`: include native C++ source files in package data.
- `README.md`, `TESTING.md`: real-data workflow and verification documentation.

No original tests were removed or weakened. Raw competition files were not modified, and no synthetic data was used for the trained real model.
