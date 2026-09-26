> **Completed campaign beyond Amazon 0.931:** see [current experiment journal](reports/CAMPAIGN_0931.md). Work is on `parth`; F: is read-only reference. Older transfer/stopped notices below describe the previous run.

> **Historical transfer checkpoint:** the old run stopped locally at 133,000 anchors and subsequently completed on the A5000. Its unique-owner output scored **0.931 on Amazon** (team-reported). The laptop has completed the newer campaign linked above.

> **Earlier improvement campaign (historical):** the original Amazon baseline was **0.842**. [Earlier findings](reports/IMPROVEMENT_STATUS.md) document the changes that led to **0.920 pair-threshold / 0.931 unique-owner** scores. The current journal supersedes its status and restart instructions.
# Business Entity Resolution

An offline Python pipeline for the problem described in the supplied **Amazon ML Challenge 2026 – Business Entity Resolution Challenge** specification. Source 1 is the deduplicated reference; each record may match zero, one, or several Source 2/3 records. The implementation does not verify competition rules or claim a competition score.

## Current contextual matching campaign

**Amazon update (2026-09-27): both main and alias scored 0.942.** Next optional diagnostic: `outputs/campaign_0942_france_baseline/matching_results.tsv`, which restores only the previous France predictions and keeps main US/India unchanged. It is fully validated but has **no Amazon score yet**.

Both new submissions are complete and officially validated:

- **Evaluated alias version (Amazon 0.942):** `outputs/campaign_0931_oof_alias_unique/matching_results.tsv` (5,768,373 matches).
- **Evaluated main version (Amazon 0.942):** `outputs/campaign_0931_oof_unique/matching_results.tsv` (5,753,928 matches).

Fresh confirmation macro F0.5 improved from **0.940297** to **0.959566**, then **0.960144** with conservative alias additions. These are local confirmation scores. On 2026-09-27, the user reported **0.942 on both new Amazon submissions**, up from **0.931**. The alias gain is not demonstrated at the reported precision. Both files cover all 1,732,544 anchors, have no duplicate target ownership, and passed full streaming integrity checks plus official matching-ID validation. All **59 regression tests passed**. The alias file preserves every main match and adds 14,445 links. Each output folder includes its corresponding `candidate_pairs.tsv`; use the expanded candidate file with the alias output.

See the [model card](reports/CAMPAIGN_MODEL_CARD.md), [experiment journal and handover](reports/CAMPAIGN_0931.md), and `reports/campaign_0931/delivery.json`. Both required model weights are preserved; a verified weights-only add-on is at `transfer/campaign-0931-models.zip` (rebuild with `python -m src.campaign_model_bundle`). This ZIP does not replace dataset/index/base-score assets.

Check/reuse the completed workflow from this repository on `parth` (validated outputs are skipped automatically):

```powershell
.\.venv\Scripts\python.exe -u -m src.campaign_run --main-workers 6 --alias-workers 2
```

Install `requirements-campaign-tested.txt` in a restored environment. This workflow reuses the complete verified base scores imported from F, plus the two model artifacts listed in the model card. F remains read-only. The older `complete_improvements` command reproduces the earlier baseline, not this campaign.

## Earlier Windows workflow (0.931 baseline)

See [the Windows runbook](reports/WINDOWS_IMPROVEMENTS.md) for reproducible commands, results, preserved baseline files, and limitations. GPU CatBoost with 113 evidence features reached **0.943147 selection F0.5**, versus **0.875670** for the original forest on the same anchors. This is not an Amazon evaluation score. A fresh 5,000-anchor confirmation achieved **0.943215**, versus **0.878801** baseline; details are in `reports/improvements/confirmation_evaluation.json`. Full improved inference was stopped by the user for transfer; see `reports/improvements/completion_status.json` for live state. All **45 tests passed** on Windows, including native retrieval, export, and exact provenance replay.

## Original competition run (historical baseline)

The supplied resources are in `../6ab10eb3b23ba_student_resource/student_resource`; raw data are in this repository's `dataset/`. `config/real.json` configures the disk backend. Run from this repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt
# macOS, if needed: brew install libomp
.venv/bin/python -m src.real_pipeline run --config config/real.json
.venv/bin/python -m pytest -q > reports/final_tests.log
.venv/bin/python -m src.real_report --config config/real.json
```

A C++17 compiler is required for compact memory-mapped retrieval. Raw TSVs remain untouched. This is one task with three sources: final `outputs/real_submission/matching_results.tsv` and `candidate_pairs.tsv` each cover every test S1, including France. The `run` command executes the **supplied** `../student_resource/utils/validate_submission.py` with both files and `--check-ids`; its exact output is recorded in `reports/official_validator.log`.

The real backend preserves the existing four classifiers, features, sigmoid calibration and singleton convention. SQLite and memory-mapped indexes replace the oversized Python retrieval indexes. Exact-name, country/name, minhash name/address, postal, address-bag, number/token and name-token-pair rules form a union. A retrieval ranker trained on fitting entities retains at most 32 candidates. Every retained candidate is scored and exported; cap/posting losses are measured, never hidden. `city_name` is inactive in this backend. The reference pool includes all S2/S3 records.

For bounded fitting, the seeded S1 split samples 30,000 fitting, 5,000 calibration and 10,000 validation anchors, retaining every candidate for those anchors. Full training blocking audit was interrupted (last recorded progress approximately 1.4 million anchors); test inference completed for every test S1. TF-IDF fits on fitting anchors plus 70,000 random training reference records. Shared reference text is a transductive aspect of the split; held-out labels are never used for fitting. Validation is a model/threshold selection set, not an untouched test estimate. The official metric is macro per-S1 F0.5, including correct singletons.

`work/real_v1` contains rebuildable caches. **Use a fresh working directory after changing data, normalization, retrieval, sampling or feature settings.** Do not mix old feature caches with a new ranker. Four workers and 100-anchor batches are configured for this 16 GB Mac; the task queue is bounded. Allow tens of GB of disk space and substantial time for full scans, indexing, audit and inference. The supplied full validator additionally materializes large ID sets.

The final `reports/real_dataset_results.md` was not generated in the original run. Existing measured results are in `reports/model_comparison.tsv`, `reports/real_run_manifest.json`, and the improvement reports linked above. Per-stage JSON/TSV artifacts and `reports/real_pipeline.log` preserve evidence. `reports/prior_run/` contains copied historical measurements and is not evidence of the current run. Synthetic data is used only by software tests.

The remaining sections document the original in-memory backend (`config/default.json`), whose retrieval and threshold grid differ from the real disk backend.

## Quick start

Python 3.10+ is required. Create an isolated environment and install the allowed ML libraries:

```bash
cd business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Dependency installation requires access to a Python package registry or a preloaded wheelhouse. Pipeline execution itself is offline. On macOS, an XGBoost import error mentioning `libomp` requires the platform OpenMP runtime; the CLI reports this dependency error instead of silently substituting another model. `requirements-tested.txt` records the versions used for this delivery's tests. Native wheels vary by platform.

Place the challenge data at these paths, or set `dataset_dir` in your config:

```text
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

Train, compare models, select a threshold, save an artifact, and infer:

```bash
python -m src.main run --config config/default.json
```

Or run stages separately:

```bash
python -m src.main train --config config/default.json
python -m src.main infer --config config/default.json
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The included `utils/validate_submission.py` is a **local schema validator** for the original in-memory backend. The real backend separately runs the supplied official validator as described above. Inference also runs these local schema checks automatically before publishing its files.

CLI path overrides are supported: `--dataset-dir`, `--output-dir`, `--model-path`. Paths resolve against the working directory. Inference always uses the blocking and feature settings stored in the model; changing retrieval requires retraining. Use `--verbose` for tracebacks. Only load trusted `.joblib` artifacts: they use pickle serialization.

## Software-test demo

Generate a fictional software-test dataset in a fresh directory:

```bash
python utils/make_demo_data.py --root demo/dataset --count 100
python -m src.main run --config config/demo.json
python3 utils/validate_submission.py \
  --matching demo/output/matching_results.tsv \
  --candidate demo/output/candidate_pairs.tsv \
  --test-dir demo/dataset/test
python -m pytest -q
```

The generator refuses to overwrite existing TSV files. Synthetic data includes multi-match entities, singletons with plausible hard negatives, missing countries, non-Latin text, and an unseen test country. It is useful for exercising code paths, not estimating challenge performance.

## Modules

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Validated, serializable configuration |
| `src/data_loader.py` | Strict TSV ingestion and ground truth |
| `src/normalization.py`, `src/preprocessing.py` | Raw text retention, normalized fields and address hints |
| `src/blocking.py` | Indexed candidate union with provenance |
| `src/features.py` | Sparse TF-IDF and pair similarities |
| `src/model.py` | Rules, logistic regression, random forest, XGBoost, calibration |
| `src/threshold.py`, `src/evaluation.py` | Entity-aware F0.5, threshold sweeps and blocking recall |
| `src/training.py` | Split isolation, experiments, artifacts |
| `src/inference.py`, `src/submission.py` | Batch scoring, TSV export and validation |
| `src/embeddings.py` | Optional local-only embedding comparison |
| `notebooks/experimentation.ipynb` | Explore saved reports without duplicating training logic |

## Data loading and validation

Every TSV is read with `pd.read_csv(path, sep="\t", ...)`. `dtype=str` preserves textual fields and leading zeros; `keep_default_na=False` preserves literal country strings such as `NA`. Empty/whitespace-only values are missing. Logs include shape, columns, missing counts, unique IDs, and distinct nonempty countries.

Required source columns are `entity_id`, `business_name`, `business_address`, and `country`. IDs must be unique within each source and start with the correct `S1-`, `S2-`, or `S3-` prefix. They cannot contain whitespace or commas. Countries are normalized strings, with no fixed vocabulary, geographic allowlist, external aliases, or US/India assumption. Country equality is only a pair feature/blocking hint; an unseen country works without retraining an encoder.

Ground truth requires `source1_entity_id` and `matched_entity_ids`, with exactly one row per training S1, including explicit empty cells for singletons. Match cells may contain comma-separated IDs or JSON string lists. Missing S1 rows are errors; they are never silently relabeled as singletons. Referenced IDs must exist in training S2/S3. Duplicate matches and assignment of the same target to multiple deduplicated S1 entities are errors.

## Normalization and address hints

Names retain their original text alongside Unicode NFKC/case-folded text. Apostrophes are removed, other punctuation becomes spaces, `&` becomes `and`, common abbreviations are expanded, and terminal legal suffixes are removed conservatively. `corporation` is retained; names such as `Limited Edition` remain intact. Unicode scripts are preserved. Examples in the request are covered by tests.

Addresses keep information while applying consistent abbreviations such as `road → rd` and `street → st`. The parser extracts leading street numbers, trailing numeric/alphanumeric postal patterns, and city/state hints from comma-separated address segments. It uses no gazetteer. **These are uncertain hints:** international formats, postal codes followed by country names, unit-first addresses, and ambiguous comma layouts may be missed or misparsed. Structured-field presence flags let the model distinguish missing evidence from a mismatch. Missing values never produce a positive exact-match feature. A postal mismatch never vetoes a candidate.

## Candidate generation and recall

The blocker builds independent inverted indexes over S2/S3 and unions the emitted pairs:

| Path | Retrieval key / decision |
| --- | --- |
| Country + name | Equal nonempty country and an informative name token |
| Country + city + name | Equal country, extracted city and informative name token |
| Postal | Equal nonempty country and postal hint |
| Name n-grams | Global 3/4/5-gram postings, minimum overlap and Jaccard similarity |
| Address | Equal country and at least two informative shared address tokens |
| Strong name | Fuzzy ratio on n-gram-retrieved names, plus normalized exact-name lookup |

Name retrieval is intentionally country-independent to recover missing or inconsistent country fields. City is optional. Generic words cannot be the sole name/address blocking key. Narrow city keys can survive when broader name postings are suppressed. No full Cartesian product or all-pairs fuzzy search is performed. Country-only comparison counts are computed analytically rather than materializing pairs.

`max_posting`, `max_document_frequency`, and `common_token_floor` suppress overly frequent index keys. This bounds each retained posting, **not the total union size**. There is no hidden top-k cut after retrieval. Suppressed keys are logged; increasing the limits can recover matches at additional memory/runtime cost. True-match recall is measured, never assumed to meet a target.

For fitting, calibration, and validation separately, reports include:

- Candidate recall = retrieved ground-truth pairs / all ground-truth pairs, including positives absent from blocking.
- Mean candidates per S1, including S1s with none.
- Reduction ratio = 1 − candidates / (`number of S1 × number of S2/S3`).
- Fraction of S1 entities with zero candidates.

Recall is `null` if a split contains no positive truth pairs. `minimum_candidate_recall` defaults to 0.98 and warns below this target; enable `fail_on_low_candidate_recall` to enforce a hard gate. `blocking_comparison.tsv` compares every rule, analytical country-only retrieval, and the combined union. The comparison measures standalone recall and candidate count, not a separately trained model per blocker.

## Features and hard negatives

Features include raw and normalized exact matches; normalized Levenshtein; token Jaccard and overlap counts; sparse character TF-IDF cosine; name 3/4/5-gram Jaccard; address character n-gram similarity; length differences/ratios; postal, city, state and street number agreement/presence; country agreement/presence; combined name/address similarity; source type; and every blocking provenance flag/count.

TF-IDF is fitted only on fitting S1 text plus the shared training S2/S3 reference corpus. Calibration/validation S1 text and test text never enter vocabulary/IDF fitting. The S2/S3 reference corpus is shared across anchor splits as in the matching task; this is not a holdout of all text belonging to a business. Caches are split-local and are not serialized into the artifact. Cosines are computed only for selected pair rows, without an all-record similarity matrix.

Every generated training candidate receives a ground-truth label. **No negative subsampling** occurs: easy, name-similar, address-similar, same-country, same-city, and multi-rule negatives retain the inference blocking distribution. Ground-truth positives missed by blocking are not injected into training or candidate exports; reports and end-to-end recall retain those misses. Class weighting addresses imbalance during model fitting; calibration is unweighted to preserve observed prevalence.

## Splits, models, and calibration

The default split is 60% S1 for fitting, 20% for calibration, and 20% for validation. Singleton status is stratified where feasible; otherwise a logged, seeded S1-level split is used. Splits are disjoint by S1 and written to `split_manifest.tsv`. Pair-level random splitting is never used. At least ten S1 records and both positive and negative fitting candidates are required; statistically useful validation typically needs substantially more data.

The pipeline compares an exact/rule baseline, standardized logistic regression, random forest, and XGBoost. XGBoost is a required dependency. Every learned model receives sigmoid calibration fitted on its disjoint calibration entities. One-class/empty calibration sets are logged and retain raw scores; calibrated probabilities are not claimed in that case. The rule baseline remains a binary rule.

Model and threshold selection use validation F0.5. The chosen fitted estimator, calibrator, threshold, vocabulary and configuration are saved together. There is no automatic full-data refit, because it would change the score distribution underlying the selected calibration/threshold. The tradeoff is that only the fitting split trains estimator parameters. For additional statistical assurance, reserve another untouched group holdout or use nested group cross-validation; the current validation score is a selection score, not an unbiased generalization estimate.

## Exact F0.5 convention

The supplied challenge README confirms **entity-macro F0.5** and the empty/empty singleton convention. For each S1, let `T` be true target IDs and `P` predicted target IDs:

```text
if both P and T are empty: precision = recall = 1
if exactly one is empty:  precision = recall = 0
otherwise: precision = |P ∩ T| / |P|; recall = |P ∩ T| / |T|
F0.5 = 1.25 × precision × recall / (0.25 × precision + recall)
```

A zero denominator gives F0.5 = 0. Reported entity precision, recall and F0.5 are the separate means across **all** validation S1 entities, including singletons and zero-candidate anchors. Mean F0.5 is not generally the F0.5 of mean precision/recall. All missed positive pairs, including blocking misses, contribute false negatives. `metric: "micro"` instead selects by pooled pair F0.5; micro metrics are always also reported. Use entity-macro for this challenge, as configured in `config/real.json`.

Thresholds are 0.10, 0.15, …, 0.95, plus 0.99, 1.0, and a floating-point value just above 1 for explicit reject-all behavior. Selection uses `probability >= threshold`; ties favor higher precision, then higher threshold. Models tie-break deterministically. Ordinary accuracy is never an optimization target.

`singleton_accuracy` is the fraction of true singleton anchors predicted empty (null when no true singleton is present). `singleton_false_merge_rate` is its complement. `false_positive_rate` uses all possible negative S1–target pairs as the denominator, so it can be small in a huge retrieval space; inspect singleton false merges and precision alongside it.

## Match decisions and exports

Inference retrieves candidates, computes features in bounded batches, and scores **every** retrieved pair. It keeps all IDs at or above the learned threshold and sorts accepted matches by descending probability then ID. It never forces the top match or enforces a one-to-one assignment. Empty candidate sets and all-rejected candidates yield empty match cells.

Both files have exactly one row per test S1 in source order:

```text
matching_results.tsv: source1_entity_id<TAB>matched_entity_ids
candidate_pairs.tsv:  source1_entity_id<TAB>candidate_entity_ids
```

Cells contain comma-separated unique S2/S3 IDs from test data. Candidate files contain the exact scored set, including rejected pairs. Every final match must occur in its candidate cell. Inference checks row coverage, IDs, uniqueness, membership, schema and scored/exported counts. Files are validated in a temporary directory before individual atomic replacement. The two renames are not a transactional pair; if the process is interrupted between them, rerun inference and validation.

## Optional local embeddings

Disabled by default. This path is only appropriate if the challenge explicitly permits pretrained weights and you have verified the local model's license. No model is included or downloaded. Install the optional library and configure:

```bash
python -m pip install -r requirements-embeddings.txt
```

```json
{
  "embedding_model_dir": "/absolute/path/to/already-available-model",
  "embedding_license": "Apache-2.0",
  "embedding_constraints_confirmed": true
}
```

Use those fields in a copy of the full configuration. MIT is also accepted. The declaration is your attestation, not automatic license verification. Loading forces offline mode, `local_files_only=True`, and `trust_remote_code=False`. Model contents are hashed; inference rejects changed weights. Name/address embedding cosines augment all traditional features. XGBoost with embeddings is eligible only if its validation F0.5 strictly exceeds traditional XGBoost, then competes with the other models. Without a permitted local model the comparison row explicitly says skipped. This optional runtime path is not end-to-end tested in this delivery because no embedding model was provided.

## Reports, reproducibility, and scale

`model_comparison.tsv` reports model, candidate recall, average candidates, precision, recall, F0.5, singleton accuracy, threshold, calibration status and model fit/calibration/selection runtime. Shared preprocessing time is excluded from model runtimes and included in the total run manifest. The manifest records configuration, package/Python versions, training-input SHA-256 hashes and metric conventions. Threshold sweeps and per-model labeled validation pair diagnostics are saved separately.

Training retains its candidate tables and dense feature matrices in memory; it is intended for datasets whose retrieved union fits RAM. Inference retains the target indexes, sparse record vectors and at most one anchor's output candidates plus scoring batches. No web-scale or memory-independent training claim is made. Monitor retrieval volume before increasing posting limits; for larger corpora, persist candidate partitions and use an external-memory learner while preserving entity splits and exact export provenance. TF-IDF vocabulary is bounded by `tfidf_max_features` per field. Defaults use one CPU thread and seeded sampling for repeatability; different library/platform versions can still produce small numeric differences.

Tests exercise normalization, missing evidence, unseen countries, candidate union/recall, conservative thresholding, full-label evaluation, split isolation, malformed ground truth, invalid submissions, and complete training/inference with exact scored-candidate export. See `TESTING.md` for this delivery's actual validation results.
