# Local model improvement run

The user reports a competition score of **0.94** and requests a target of **0.98**. There is no new competition score yet. This run cannot guarantee a hidden-test minimum. Existing submitted files and trained artifacts are preserved.

## Measured checkpoint — 2026-09-27

All planned local models have trained. The selected model is the depth-8 context XGBoost model (815 trees), using the depth-10 base model (1,160 trees) and a frozen threshold of `0.6500000000000002`.

| Stage | Macro F0.5 |
|---|---:|
| Depth-7 base selection | 0.956944 |
| Depth-10 base selection | 0.957194 |
| Depth-5 context selection | 0.962769 |
| Depth-8 context selection | 0.963580 |
| **Fresh 10,000-anchor confirmation** | **0.964032** |

Confirmation 95% bootstrap interval: **[0.961561, 0.966467]**. Precision: 0.978823; recall: 0.932221. India F0.5: 0.951101; US: 0.972296. See `local_098_v1/confirmation.json` for the complete measured result.

**The local 0.95 gate passed. The 0.98 target has not been reached on this holdout.** These are local training-derived scores, not a new competition result.

Full test inference started successfully and passed its first production shards. It remains resumable under `work/local_098_v1/test_scores`. Final TSV exports and official validation are pending until all 1,732,544 test anchors have been scored. Read `local_098_v1/status.json` for live completion status; do not interpret this launch checkpoint as a completed submission.

## What changed

- Added a CPU XGBoost training and inference workflow using the existing 30,000 fitting, 5,000 calibration and 10,000 model-selection anchors.
- Added 55 richer identity-evidence features and 46 offline multilingual/numeric/token features to the original 58 features (159 base features).
- Offline transliteration uses macOS ICU rules. It does not fetch external business data or pretrained model weights. This has its own schema and does not impersonate the pulled models' AnyAscii features.
- Train two base models and three disjoint-anchor cross-fitting folds. Train context models on the resulting out-of-fold probabilities, the base features, and candidate-to-candidate evidence. No fitting anchor receives a base prediction from a model fitted on its label.
- Early stopping and sigmoid calibration use separate complete-anchor subsets of the calibration sample.
- Freeze the selected model and threshold before evaluating 10,000 fresh anchors. Exclude the original samples and all three prior confirmation samples, reconstructed from their saved seed recipes.
- Require the fresh confirmation macro F0.5 **and its lower 95% bootstrap confidence bound** to be at least **0.95** before full test inference. This is a local check, not a guarantee of competition performance. Record separately whether local F0.5 reaches **0.98**.
- Export a separately named unique-owner submission. Every retained candidate is scored and exported. Preserve empty predictions and abstain on exact ownership ties.
- Run the supplied official validator on both final files with ID checking.

## Why this is a separate local campaign

The latest pull contains the CatBoost/context source and past reports, but its trained CatBoost/context artifacts and feature caches are absent from this Mac. CatBoost and AnyAscii are also absent, and the attempted package download failed because the execution environment could not reach the package index. XGBoost is already installed and works locally.

The pulled reports include roughly 0.956 fresh confirmation and 0.9615 model-selection results; those are historical results from other artifacts, not results of this new run.

The materialized candidate pool is `outputs/improved_unique_owner/candidate_pairs.tsv`. The files under `outputs/real_submission` are currently small Git LFS pointer files, so this campaign does not use them as data. Both materialized improved submissions remain untouched.

## Checks completed before launch

- 23 original pipeline tests passed.
- 8 targeted tests passed, covering multilingual transliteration, complete-anchor batching, macro F0.5 and singletons, ownership ties, the confirmation gate, calibration separation, and submission export.
- 96 real candidate rows reproduced the original cached features within numerical tolerance.
- Context/peer features were identical when computed together or in complete-anchor batches.
- Changing labels did not change the new text features.
- Existing native retrieval was usable locally.

No data-quality, blocking, or training audit command was run. The new work consists of feature generation, training, held-out evaluation and, conditionally, inference and official submission validation.

## Run and progress

From the repository root, the launched command is:

```bash
../../work/.venv/bin/python -u -m src.local_campaign --infer > reports/local_098_v1/run.log 2>&1
```

Progress: `reports/local_098_v1/status.json`

Live log:

```bash
tail -f reports/local_098_v1/run.log
```

After interruption, rerun the same command to reuse complete checkpoints. An exclusive OS lock prevents two simultaneous campaign runs. Changes to code or input signatures require a new campaign directory instead of silently reusing incompatible caches.

Key files:

- Configuration: `config/local_098.json`
- New models: `artifacts/local_098_v1/`
- Checkpoints: `work/local_098_v1/`
- Selection and confirmation reports: `reports/local_098_v1/`
- Conditional final output: `outputs/local_098_v1_unique/`

`not_promoted` means the measured local floor was not met; no replacement submission was generated. `complete` means inference/export/official validation finished, not that a 0.98 leaderboard score was obtained. Always inspect `reached_local_target` and the actual confirmation report.

## Remaining limitations

There are no supplied France training labels; local confirmation covers India and the US. Sampled-anchor ownership competition does not reproduce every full-test collision. The saved candidate pool still limits possible recall. None of the test labels or leaderboard outcomes are available locally. Keep the current 0.94 submission available until a new submission is evaluated by the competition.
