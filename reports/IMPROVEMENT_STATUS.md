# Improvement campaign — 2026-09-26

## Current status
**Stopped by user for transfer to the A5000 workstation.** No inference/training processes remain running on the laptop. Saved: 133,000 anchors, 1,330 complete shards, 4,255,387 pair scores. Transfer archive and verification are complete. No ML job is running.

Fresh confirmation: **0.943215 vs 0.878801 baseline**. Amazon score remains **0.842**, no new submission evaluated. Baseline artifacts and outputs remain untouched. Resume instructions: [A5000_HANDOVER.md](A5000_HANDOVER.md). Asset bundle: `transfer/amazon-ml-assets.zip` (9.97 GB), verified in `improvements/transfer_archive_verification.json`. Code/data are transferred separately.

## Verified baseline
- Calibrated random forest, threshold 0.55; local selection macro F0.5 0.875670 on 10,000 S1 anchors.
- Fit: 30,000 S1; calibration: 5,000 S1. 58 features; no embeddings or GPU in real backend.
- Test: 1,732,544 S1; 55,431,940 candidates; 5,747,090 links; 101,827 empty predictions.
- Fresh output audit: 111,450 targets have multiple S1 owners; 296,242 excess links affect 154,526 anchors. France accounts for 197,983 excess links.
- Validation misses: 714 raw retrieval + 70 cap/ranker + 5,095 matcher. 2,308 false positives. Singleton false merges: 137/528.

## Prioritized work
1. Windows-compatible memory mapping/compiler/loading/RSS; portable verified cache migration; correct local paths.
2. Preserve baseline and reproduce cached validation. Separate tuning from an untouched confirmation holdout.
3. Train stronger GPU tabular matcher and evidence features. Evaluate each change and country slices.
4. Target ownership arbitration with confidence/abstention. Validate with realistic competing S1 owners, not only a tiny isolated anchor sample.
5. Multilingual retrieval/matching if error analysis supports cost; no external business data.
6. Export only evaluated improvements to a new output directory. Validate, document, and package.

## Known limitations
- Legacy reports mix paused audit and completed inference. Full training blocking audit and final test report are absent.
- Original macOS caches include absolute paths and source hashes sensitive to CRLF. Never bypass validation blindly.
- GPU choice: local RTX 4060 has 8 GB VRAM. A5000 is on an inaccessible separate computer. Colab T4 is available as a possible later option, but is not needed for the current model.
- No leaderboard claim beyond user's baseline. Local improvements do not guarantee public/private score gains.

## Handover
Resume with `.venv/Scripts/python.exe -u -m src.complete_improvements` from the repository root after ensuring no existing continuation is running. See WINDOWS_IMPROVEMENTS.md. Do not launch duplicate continuations against the same checkpoints.
Read this file before HANDOVER.md, which describes the earlier paused run.
## Checkpoint: first measured improvement
- Windows original regression suite: 23 passed.
- Isolated .venv installed; RTX 4060 GPU training works.
- Legacy forest probabilities reproduced sufficiently to match saved macro F0.5 exactly.
- CatBoost GPU depth 8, 1800 trees, same 58 features: selection macro F0.5 **0.898538913** vs forest **0.875670250**. FP 1376 vs 2308; singleton accuracy 85.227% vs 74.053%. Threshold 0.65.
- Artifact: artifacts/improvements/catboost_d8_baseline_features.joblib; evidence: reports/improvements/catboost_d8_baseline_features.json.
- NOT promoted: fresh confirmation holdout and realistic ownership evaluation still required.
- Mac XGBoost pickle failed native deserialization; src/artifact_compat.py extracts the standard Model payload explicitly. Candidate parity verification is in progress.
- Run command: .venv/Scripts/python.exe -u -m src.experiments --depth 8 --iterations 1800

## Checkpoint: Windows migration verified
- All seven raw input SHA256 hashes match the previous run.
- Portable C++ DLLs compile and load on Windows. The index file format and retrieval keys are unchanged.
- Exact candidate IDs, provenance and labels matched on 100 validation anchors (3187 pairs) after XGBoost recovery.
- Windows copy: config/windows.json, work/windows_v1, artifacts/real_model_windows.joblib. Large immutable data files use hard links; baseline model/output files unchanged.
- Deeper CatBoost (depth 10) with old features reached **0.899888874** on selection validation.
- Currently generating richer evidence features and a reserved 5000-anchor confirmation cache. See reports/improvements/evidence_preparation.log and confirmation_preparation.log.

## Memory checkpoint
- 16 GB system RAM became saturated when feature enrichment and two confirmation workers ran concurrently. Confirmation preparation was stopped cleanly before cache export, and configured for one worker. Run after enrichment, not concurrently. Baseline files are unaffected.
- Native Unicode-path / empty-index Windows test passed. Numeric-evidence test fixture corrected to supply normalized text, matching the production function contract.

## Additional Windows fixes
- Explicit UTF-8 added to Path text reads/writes throughout source and tests. The Mac default encoding had hidden failures in the Unicode audit report and final-output test.
- Repeated tests are targeted at discovered failures; no assertions removed.

## Checkpoint: feature preparation complete
- Added 55 evidence features; caches complete for fitting, calibration and selection validation. Inputs are exclusively supplied records.
- Testing depth-8 and depth-10 CatBoost on 113 total features.
- Last full-suite run: 40 passed, 1 failed due to the audit report's implicit Windows encoding. Explicit UTF-8 fix is now applied; targeted regression follows.
- A5000 is on an inaccessible separate computer. Continue on local RTX 4060; Colab T4 is an option for a later memory-heavy neural stage.
- Pruning-only validation: best tested 0.88823; even oracle pruning of the baseline cannot exceed 0.92984 on this sample. Full candidate rescoring is needed to recover missed matches.


## Selection checkpoint: evidence matcher
Depth 8 achieved 0.943009; depth 10 achieved 0.943147 macro F0.5 on the existing selection set. Depth 10 and threshold 0.675 are frozen in improvements/frozen_selection.json before fresh confirmation evaluation. This is not an Amazon score. Preparing 5,000 reserved anchors with one worker to respect 16 GB RAM. A5000 is inaccessible; local RTX 4060 training is sufficient for these experiments.


## Verification and inference preparation
All 41 full-suite tests passed on Windows (improvements/final_tests.log). Two new export tests also passed, covering ownership ties, empty anchors, and candidate alignment. Windows default workers reduced to one for 16 GB RAM. Added src/rescoring.py and src/finalize_improved.py; full rescoring requires a matching frozen model with a positive confirmation confidence interval. These new inference stages still need a real-data smoke test before a full run. Baseline is untouched.


## Fresh confirmation: passed
Frozen depth-10 evidence matcher reached **0.943214619** macro F0.5 versus **0.878800622** baseline on 5,000 reserved S1 anchors. Paired improvement +0.064414, bootstrap 95% interval [0.058966, 0.069730]. False positives 1136 -> 337; true positives 14756 -> 15772. US 0.914519 -> 0.959672; India 0.825312 -> 0.918569. France remains unmeasured. No threshold or model adjustment was made using confirmation.

Promoted for full test rescoring, pending inference smoke test. Original Amazon score remains 0.842; no improved TSV is complete yet. Ownership selection diagnostic removed zero accepted pairs among 10,000 sampled anchors, so it provides no evidence for full-corpus ownership changes. Export both variants separately if needed; fixed pair-threshold is the measured policy.


## Fast inference verified
New exact provenance replay avoids 4 GB of index mappings per worker. All 31,973 predictions across 1,000 test anchors exactly match the original path. Runtime 184.8 s -> 21.5 s. A three-worker continuation scored another 3,000 anchors in 18.4 s. Two provenance regression tests passed. Ready for full continuation via `python -m src.complete_improvements`; runtime expected to be hours, checkpointed every 100 anchors. Completion and failures are written automatically to `improvements/completion_status.json` and this journal.


## Final pre-inference verification
All **45 tests passed** in 33.97 seconds. SHA256 verification confirms the original model and both original submission TSVs remain byte-for-byte unchanged. Full inference/export/validation continuation is being launched as a hidden background process; keep the computer awake. The new submission is not complete until completion_status.json says complete.

- Automated continuation: {"stage": "full_rescoring", "state": "running", "updated_utc": "2026-09-26T07:46:00.590236+00:00"}


## Runtime check requested by user
The 4060 was used for training, but full submission generation uses CPU feature extraction and CPU prediction. GPU snapshot: 0% utilization, 0 MiB allocated, 49 C. The job advanced from 67,700 to 72,900 anchors during inspection. RAM was 91.8% used with substantial Windows page reads, so disk/memory pressure is a real bottleneck. No interruption or reconfiguration performed. See improvements/runtime_diagnostic.json.


## Workstation transfer ready
`transfer/amazon-ml-assets.zip`: 9,972,950,552 bytes compressed; 22,184,574,176 unique restored bytes, 38,837,927,359 logical bytes with hard-linked copies. 1,465 files, 25 aliases. Companion SHA256 file provided. Seven targeted transfer/provenance/export tests passed after portability changes. No ML job is running; resume only on the destination as requested.


## Final transfer verification
All 1,465 asset entries passed decompression/SHA256/size/alias checks. Staged Git source signatures match the saved scoring checkpoint, and all required inference assets are present. Git main is being published; an offline Git bundle is included as a backup. Resume only on the workstation using the A5000 handover.
