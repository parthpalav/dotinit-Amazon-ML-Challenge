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
