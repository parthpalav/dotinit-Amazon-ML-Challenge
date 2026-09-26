# Campaign beyond Amazon 0.931

## Current state
Active on `parth`, C: working repo. `F:/dotinit-Amazon-ML-Challenge` is read-only reference. User reports Amazon **0.920** pair threshold and **0.931** unique owner. A5000 unavailable; local RTX4060 8GB / 16GB system RAM. New work must be bounded-memory and resumable. Preserve the proven model and submissions; an old model is still useful as a control or ensemble member.

## Verified constraints and baseline
Official problem statement and guidelines re-read. Ground truth is provided for training only; test has no labels. No external identity lookup. Final model MIT/Apache-2.0 and <=8B parameters. Macro per-S1 F0.5 includes singletons. Five submissions/day; private score determines final outcome, so avoid public-score-only tuning.

Reference run complete and official full validation passed. Same 55,431,940 candidates; pair policy 5,626,387 links and 86,460 multiply-owned targets. Unique-owner has 5,431,304 links and zero duplicate ownership. It removed 195,083 links and improved reported Amazon score by 0.011. Local random-anchor validation had no accepted ownership conflicts; it did not measure this deployment issue.

## Priorities
1. Audit errors by country, missingness, numeric contradictions, name ambiguity, source, and candidate recall. Inspect complete scored graph from F without rerunning expensive inference.
2. Test per-entity decision rules and contextual reranking that account for candidate competition and corroborating records; compare against frozen pair baseline on training-derived validation.
3. Improve validation realism: competing S1 records and country transfer, not only isolated random anchors. Keep a newly reserved confirmation set for promoted changes.
4. Test stronger text/numeric evidence and larger/OOF fitting if low-cost context is insufficient. Neural multilingual matching only if error slices justify it.
5. Export separate versioned submission only after checks; preserve 0.931 output as fallback. Never restart old full scorer against incomplete local checkpoints by mistake.

## Handover
Latest machine state: `reports/campaign_0931/campaign.json`. Reference output metrics and completion/validator logs copied into `reports/campaign_0931/`. Training, indexes, and feature caches remain in current C repo; completed test score shards are on F. No new experiments launched at this checkpoint. Subsequent checkpoints appended below.


## Checkpoint: graph and context experiments
All 55,431,940 reference scores consolidated into `work/campaign_0931/test_scores.npy` in 60 seconds. Country audit: ownership removed 1,694 US links, 37,659 India links, and 155,730 France links (about 80% of removals). See `reports/campaign_0931/full_graph_audit.json`.

Error slices on old selection: 2,896 retained positive pairs missed; 656 have missing target addresses, 534 have non-Latin target names, and 1,243 have mutually exclusive numeric tokens. These overlap. Numeric differences must remain learned evidence, not hard rejection rules. Row order has negligible correlation with labels and is not being used as a feature.

Context-only reranking selection improved from 0.943147 to 0.945688 (exploratory). Candidate-to-candidate peer evidence further reached **0.952492** (depth 7, threshold 0.700, 662 trees; 439 FP vs champion 657). Peer features use text and base-model probabilities, never peer labels. Meta fitting uses 4,000 old calibration anchors; sigmoid calibration uses 1,000. Original validation is selection only. Fresh confirmation required before promotion. Models under `artifacts/campaign_0931`, predictions/caches under `work/campaign_0931`.

Next: test script-independent transliteration and finer numeric/token evidence; build a new confirmation set and realistic ownership competition evaluation. AnyAscii 0.3.3 is an ISC-licensed character transliteration library (https://github.com/anyascii/anyascii), not an external entity lookup or a pretrained matching model. Champion CatBoost remains the baseline and is needed to compute stacking inputs; its verified backup already exists on F.

## Checkpoint: detail evidence and frozen fresh confirmation
Full context + peer + transliteration/detail reranker selection F0.5 **0.957744** (depth 9, 602 trees, threshold .675; TP31,912 FP391). Compact reranker using saved champion probability plus context/peer/detail only: **0.956837** (depth 5, 1,199 trees, threshold .675; TP31,877 FP386). Compact avoids recomputing the expensive original 113 features for 55 million candidates. Both recipes are frozen for a fresh 5,000-anchor confirmation sample, seed 20260927, excluding the original 45,000 and prior 5,000 confirmation anchors. See `src/campaign_confirmation.py`; log `reports/campaign_0931/confirmation_v3.log`. No Amazon result for these models yet; no new submission exported.

## Reproduction and resume commands (current campaign)
Run from this C-drive repository on `parth`, with `.venv` and restored assets. New packages: `pip install -r requirements-improvements.txt`. Do not run the old `complete_improvements` to reproduce this campaign.

1. Fresh confirmation (reuses completed feature cache):
   `.\.venv\Scripts\python.exe -u -m src.campaign_confirmation`
2. Benchmark 2,000 test anchors (only after confirmation has finished to avoid memory pressure):
   `.\.venv\Scripts\python.exe -u -m src.campaign_scoring --workers 2 --limit 2000`
3. Full compact scoring, resumable with the identical command; completed 1,000-anchor shards are reused and verified:
   `.\.venv\Scripts\python.exe -u -m src.campaign_scoring --workers 2`
4. After `work/campaign_0931/test_compact/SCORING_COMPLETE.json` exists, export a new unique-owner submission:
   `.\.venv\Scripts\python.exe -u -m src.finalize_improved --work work/campaign_0931/test_compact --selection work/campaign_0931/test_compact/selection.json --output outputs/campaign_0931_compact_unique --unique-owner`

Scoring refuses full execution unless the frozen compact model has a positive lower paired-bootstrap CI on fresh confirmation. It fingerprints model, base scores, candidate pool, store manifest and feature/scoring code. Modified code requires a new work directory. Export refuses a nonempty output directory. The original candidate pool is hardlinked, not duplicated. Existing .931 output is on F and preserved. The new pipeline uses CPU for text processing and inference; GPU idleness during this stage is expected. No Colab/A5000 required.

Tests: `tests/test_campaign.py` verifies self-exclusion, no peer-label use, complete-anchor batching equivalence, and preservation of candidate coverage including empty anchors. Existing export tests still pass with the new optional selection path (5 targeted tests passed). Full official validation and measured production timing remain pending.

## Checkpoint: fresh confirmation PASSED
`confirmation_v3` excludes all previous 50,000 anchors. Champion **0.941866**, full reranker **0.957265**, compact **0.956172**. Compact TP15,942 vs15,411, FP204 vs328, FN1,399 vs1,930. Paired delta +0.014306, bootstrap 95% CI [+0.011527,+0.017134]. Full-model delta +0.015399, CI [+0.012873,+0.018191]. Both were frozen before evaluating this holdout. These remain training-derived scores, not Amazon results, and do not measure France directly.

Next experiment: cross-fit three depth-10 base models on the original 30,000 fitting anchors, each excluding one third. Use their out-of-fold predictions to train the compact reranker on 30,000 additional anchors rather than incorrectly using overconfident in-sample predictions. Original calibration contributes 4,000 meta-fit / 1,000 held anchors as before. Original validation remains the selection set. If this improves selection, reserve another fresh confirmation set before promotion. Run stages `python -m src.campaign_oof oof`, then `prepare`, then `train`. Fold predictions and 500-anchor feature blocks are resumable with input/code signatures.

## Checkpoint: deployment parity, data overlap and remote synchronization
Production compact feature reconstruction reproduces all 627 checked selection probabilities exactly (20 anchors, max error 0). Selection changes comprise 1,155 recovered true pairs and 376 removed false pairs, with 274 lost true and 105 introduced false pairs; net TP +881 and FP -271. Concrete examples and actual training owners are in `compact_error_examples.json`; they show both successes and regressions, not cherry-picked proof of universal correctness.

An exact normalized name/address/country comparison found only **one** overlapping S1 record across all train/test anchors. Direct reuse of training answers is not a meaningful shortcut. Test labels remain unavailable.

Remote `parth` had a new output/LFS-tracking commit `3f4fb90`. Merged it preserving both histories and pushed checkpoint `a91f113`. Old improved outputs added by that merge are LFS pointers locally (their real completed files remain on read-only F); do not submit a pointer file. Original local candidate TSV remains materialized and is the scoring input. All new training remains on parth; main unchanged.

The two-worker compact benchmark processed 2,000 anchors in 13.9 seconds including worker startup. This is too short for a reliable full-run ETA. A larger benchmark and memory check will choose production parallelism after model selection. OOF base folds each trained and predicted in about 49 seconds on RTX4060; all three fold predictions are saved. Meta-feature preparation is checkpointed every 500 anchors.

## Checkpoint: cross-fitted winner and retrieval hypothesis
Cross-fitted depth-9 compact reranker reached selection **0.961505**, threshold **0.6000000000000002**, 2,030 trees (TP32,359 FP421). Depth 7 reached0.961276 with fewer FP338 but lower recall. Five simple ensembles did not improve the best individual model. The depth-9 recipe is frozen in `frozen_confirmation_v4.json` for a new 5,000 anchors excluding all prior55,000.

Retrieval audit:784 true pairs absent from the old validation candidate pool;363 have non-Latin names,158 missing addresses.156 share an exact normalized name with an existing candidate predicted >=.95 by the cross-fitted model. Testing a fixed one-hop name-alias proposal: seed probability >=.95, country+name exact equality, maximum20 corpus records per alias, unchanged matcher threshold. Ground truth only evaluates proposals. This is a separate selection experiment; it will not alter the confirmed main output unless independently supported. Rules fixed here before examining confirmation_v4 results.

Full suite:52 tests passed. End-to-end completion command now `python -u -m src.campaign_finish --workers 6` after v4 confirmation. It chooses OOF only if paired CI versus compact is positive, checkpoints scoring, preserves interrupted exports, and runs official matching-ID validation. Full candidate equality/subset validation is streamed during export to avoid the supplied validator's high-memory all-candidates mapping on16GB RAM.
