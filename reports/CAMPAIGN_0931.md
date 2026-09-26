# Campaign beyond Amazon 0.931

## Current state
**Current execution command:** `python -u -m src.campaign_run --main-workers 6 --alias-workers 4`. It resumes the main submission first, then the independently confirmed alias-enhanced submission. Stages run sequentially to avoid RAM/cache contention. Live stage: `reports/campaign_0931/coordinator.json`. Earlier worker-count notes below are historical experiments.

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

Implementation checkpoint: the ensemble-support edit introduced an indentation error caught by a targeted recheck; corrected before production execution. Added a deployment prediction/portable-ensemble regression test. Compile-all and six targeted tests pass. Scoring now writes an explicit complete=true progress record at full completion. The earlier 2,000-anchor benchmark directory has the old code signature; production uses separate `test_final_*` directories.

## Checkpoint: second fresh confirmation PASSED; production selected
Fresh `confirmation_v4` excludes all prior55,000 anchors. Champion **0.940297**, initial compact **0.954455**, expanded OOF reranker **0.959566**. OOF versus champion: TP15,325 ->16,040; FP336 ->214; FN1,923 ->1,208. Paired delta+0.019269,95%CI[+0.016336,+0.022321]. Versus compact: delta+0.005111,95%CI[+0.002804,+0.007423]. US0.957770 ->0.967773; India0.912851 ->0.946675. France has no labeled validation and remains an explicit generalization uncertainty.

Production selects `oof_compact_d9.joblib`, SHA256 `b770fd7bea838e53f69a6f3dcdbdd3917520034d46dbfa6a3b045f0fbff20651`, threshold0.6000000000000002. Main production will retain global unique ownership and exact-tie abstention. A12,000-anchor six-worker benchmark uses `work/campaign_0931/test_final_oof`; those completed shards will be reused immediately by `src.campaign_finish --workers 6`. Follow `production_scoring.log`, `work/campaign_0931/test_final_oof/progress.json`, and `production_plan.json`. Do not edit fingerprinted scoring/feature modules during the run.

Alias selection experiment remains separate:17,835 new candidate proposals on3,247 of10,000 selection anchors. No candidate expansion has been applied to production. It must improve independent confirmation before promotion.

## Checkpoint: conservative alias expansion independently confirmed
Using the unchanged0.60 cutoff on newly proposed aliases harmed selection F0.5. A separate added-pair cutoff0.975 was selected on old validation (+63TP,+6FP;F0.5 0.961505 ->0.961902), frozen, then evaluated on v4. Confirmation: **+37 true links,0 additional false links**, macro F0.5 **0.959566 ->0.960144**, paired delta+0.000578,95%CI[+0.000327,+0.000896]. See `alias_frozen.json` and `alias_confirmation_decision.json`.

Preparing a separate alias-enhanced output. The main output stays unchanged. For full-data deployment, additions will be restricted to targets not already assigned in the main unique-owner output; this preserves every existing accepted match and avoids comparing the differently calibrated candidate populations when displacing owners. The random-anchor confirmation has no accepted ownership collisions, so this full-graph safeguard cannot be measured directly there. All newly scored candidates, including rejected ones, must be included in the expanded candidate TSV.

`src/campaign_alias_scoring.py` follows completed main scoring shards, uses one worker by default, and checkpoints every100anchors. It refuses changed model/code/recipe signatures. No recursive propagation: only main-model predictions >=.95 supply one-hop aliases. The main production is approximately400-500anchors/sec; conservative alias work runs separately and may finish later.

## Resource checkpoint: rebalance parallel jobs
The alias pilot was too slow because it loaded full target rows for IDs and recomputed peer/detail features for anchors with no additions. Optimized packed-ID reads and affected-anchor filtering reproduce all checked probabilities exactly (100 mixed anchors; see the added-pair count in parity JSON). New alias cache is `work/campaign_0931/test_alias_v2`, with500-anchor checkpoints; the original pilot cache remains preserved but unused. Alias export/owner/tie tests passed (7 targeted tests total).

Running6 main workers plus1 alias worker reduced available RAM below1GB. Rebalancing the main scorer to4 workers, preserving all completed shards. This is a resource adjustment, not a restart from zero. Current completion commands: `python -u -m src.campaign_finish --workers 4` and `python -u -m src.campaign_alias_finish --workers 1`. Both completion scripts automatically export and run official matching-ID validation. Alias additions never remove a main accepted owner. Do not start duplicate copies while these jobs are already active.

## Current resource decision: sequential main then alias
Concurrent execution reduced main throughput substantially despite the lower worker count. To get a validated main submission sooner and avoid memory/cache contention, the coordinator now runs main scoring/export/validation with6workers, then resumes the alias stage with up to4workers (bounded by available RAM). Every existing main and alias checkpoint is preserved. Single resume command: `python -u -m src.campaign_run --main-workers 6 --alias-workers 4`. It skips already validated, hash-matching outputs. Main progress and alias progress remain in their respective work directories; `coordinator.json` identifies the active stage. No more model/feature changes are planned during full inference.

## Checkpoint: concurrent Phase 8 merge and index compatibility
Merged remote `ddc16e7` into `parth` without replacing the confirmed campaign. Its best reported local F0.5 is 0.932878, versus this campaign's selection 0.961505; its wider retrieval remains an experiment. The incoming root blocking reports describe its cap-64 pool, not this campaign's frozen 55,431,940 candidates.

The incoming native supplemental key layout was incompatible with existing indexes. Preserved it as `src/native/supplement_phase8.cpp`, restored the legacy default, and require explicit `AMAZON_SUPPLEMENT_VARIANT=phase8` for that experiment. Native loading now checks the index manifest's source hash. Rebuilding supplemental packed records refuses shared links, preventing accidental truncation of existing assets; the experimental rebuild creates private packed records and uses Windows-compatible hardlinks only for immutable inputs. It has NOT been run. Targeted integration tests: 20 passed. Existing legacy native hash and fresh confirmation feature-cache signature still match exactly. No running inference feature/model files changed.

At this checkpoint main scoring passed 922,000 / 1,732,544 anchors at about 463 anchors/sec; sequential coordinator continues automatically into alias scoring, export and validation. Resume with the command at the top of this report. New outputs are not ready until their production plan says `ready_for_amazon_evaluation`.

Full post-merge verification: 56 tests passed in 58.53 seconds (reports/campaign_0931/integration_tests.log).


Model handover: `transfer/campaign-0931-models.zip` contains both required model weights, frozen recipes, dependencies and license/model card. Every extracted byte hash was verified. This is a weights-only add-on, NOT a replacement for the original dataset/index assets or complete base score cache. ZIP manifest/hash: `reports/campaign_0931/model_bundle.json`. Both old and new models remain in place because inference requires both.


## Checkpoint: full main inference complete
All 1,732,544 anchors / 55,431,940 original candidates have been scored. `work/campaign_0931/test_final_oof/SCORING_COMPLETE.json` exists. The final resumed segment processed 1,188,544 anchors in 2,522 seconds (about 471/sec). Main export and official matching-ID validation are now active; the sequential coordinator will then resume alias inference automatically. Do not submit a partially written output; check the corresponding production plan for `ready_for_amazon_evaluation`.


## Checkpoint: main submission READY
`outputs/campaign_0931_oof_unique/matching_results.tsv` is complete: 1,732,544 rows, 5,753,928 matches, 108,973 empty rows, zero duplicate target ownership. SHA256 `65a1648f0e07d230dfb488328987f416ce76da55e7c2f3292e5bbc50a74fcc94`. Full streaming candidate equality/subset/coverage checks passed; official matching validator with `--check-ids` passed. Its optional candidate-file warning is expected: the separate full streaming checks covered all 55,431,940 candidates without materializing the huge candidate file in the official validator. Export took 134.8 seconds.

Full output comparison: +427,927 links and -105,303 links versus the preserved 0.931 output; this is not a labeled test gain. France count changed 869,115 -> 874,265; US 2,091,035 -> 2,218,776; India 2,471,154 -> 2,660,887. Report: `reports/campaign_0931/main_output_audit.json`. User was asked for the Amazon score of this ready main file while alias inference continues. Coordinator selected 3 alias workers based on available RAM.


## Checkpoint: global tie safeguard and runtime profile
A complete main-score audit found only seven targets with tied top scores above the matching threshold (all seven also above .975). Alias export now requires an otherwise eligible new winner to strictly exceed any prior main score, so an alias cannot turn a tied main abstention into a lower/equal-scoring assignment. Existing accepted main owners remain untouched. Four exporter edge-case tests passed. Rechecking the fresh confirmation found zero high-score ties and zero affected alias decisions: the confirmed .960144 result is unchanged. This correctness safeguard is not claimed as a measured Amazon gain. Reports: `main_tie_audit.json`, `alias_tie_tests.log`, `alias_tie_confirmation.json`. No inference feature/model code changed or checkpoints invalidated.

One completed 500-anchor alias batch reproduced its saved scores exactly under profiling. Most time was database reads and peer/detail text features; model prediction was not the dominant cost. Alias throughput improved after initial warm-up; 486,500 anchors completed at this checkpoint. The ready main output is committed/pushed as `b575be8`; alias inference continues under the same coordinator.
