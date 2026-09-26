| State | Candidate pool / actual provenance | Macro F0.5 |
|---|---|---:|
| True Control: 81-Feature Model, Original Candidate Pool (Cap 32) | Original; full 30,000-anchor fit; arbitration + δ=0.20 | 0.9336216172545162 |
| Phase 7 historical reference (brief calls this “Partial Retrain”) | Rebuilt cap 64; original v3 estimator recalibrated on rebuilt calibration; arbitration + δ=0.20 | 0.9287367252062976 |
| Phase 8 Full Retrain | Rebuilt cap 64; full 30,000-anchor fit; arbitration + δ=0.25 | 0.9328782659065414 |
| Phase 4 idealized simulation — reference only | Hypothetical hybrid; 72-feature model and synthetic score changes | 0.9372308853017451 |

**Verdict: the blocking rebuild is a net loss on this validation comparison. Revert to the original cap-32 pool for the recommended production baseline.** Rebuilt minus control is exactly `-0.0007433513479747633` using the stored double-precision scores; the decision does not compare rounded five-decimal values. The control wins by 0.0007433513479747633 (about 0.0743 percentage points).

The two full-fit models use the same 81-feature order, CatBoost settings, disjoint fit/calibration/validation anchor sets, validation truth, sigmoid calibration method, threshold grid, target-only arbitration, and six-delta sweep. Their candidate pools and corresponding training/calibration distributions differ. This is an operational validation result, not a claim of statistical significance or a fresh test-set estimate: thresholds and deltas were selected on validation for both, as required to match Phase 8.

**Final locked recommendation.** The canonical manifest is [config/phase9_production_lock.json](../../config/phase9_production_lock.json).

- Candidate pool: `work/real_v1`; cap 32; posting limit 240; preserved original `work/real_v1/train/supplement_index.bin`; no rebuilt phonetic or relaxed-address retrieval channels.
- Trained model: `artifacts/improvements/phase9_true_control_20260926T152905058159Z.joblib`. Newly trained end-to-end; 58 baseline + 14 veto + 9 enhanced features; original calibration split. Best iteration 1199 (1,200 trees).
- Post-processing: sigmoid probabilities → target-only arbitration (`max_per_source=None`, stable pair-order ties) → margin gating δ=0.20 within each anchor/source group **after arbitration** → probability ≥ 0.55. The exact threshold-grid float is `0.5500000000000002` and is preserved in the manifest.
- Enhanced feature computation supplies no preliminary probabilities, matching Phase 8; the two preliminary-margin feature columns therefore remain constant. The original feature engineer, candidate ranker, model and preserved supplement index are pinned by SHA-256 in the lock.
- This locks the recommendation and artifacts. No test-set inference or deployment was performed. When generating future candidates, use the preserved original indexes; regenerating them with the changed C++ channel implementation would not reproduce the control.

The selected control gives macro precision 0.9638301587301586, macro recall 0.8713378174603175, TP=30,109, FP=702, FN=4,567. Phase 8 gives TP=30,465, FP=859, FN=4,211. The rebuilt operating point recovers 356 more TPs but adds 157 FPs; the per-anchor precision/recall tradeoff lowers macro F0.5.

**True-control training provenance.**

`TRAINING DATA PROVENANCE: split=fit, anchors=30000, pairs=959871, positives=101605, candidate_recall=101605/104019=0.9767927013334102, fit.joblib mtime=2026-09-25T10:43:03.034890+00:00, source_index=work/real_v1/train/supplement_index.bin (original cap 32, posting limit 240, no phonetic/relaxed-address channels)`

| Original split | Anchors | Candidate pairs | Retained true / total true | Feature file mtime (UTC) |
|---|---:|---:|---:|---|
| fit | 30,000 | 959,871 | 101,605 / 104,019 | 2026-09-25T10:43:03.034890+00:00 |
| calibration | 5,000 | 159,974 | 16,982 / 17,421 | 2026-09-25T10:40:50.603457+00:00 |
| validation | 10,000 | 319,956 | 33,892 / 34,676 | 2026-09-25T10:40:25.594445+00:00 |

Cached supplemental features existed for all three original splits and were reused after checking shape, row index, feature count and exact recomputation of 514 deterministic sampled rows per split. This is sampled cache-content verification, not a full recomputation. Feature/model file hashes and byte sizes are recorded in the audit. Fit, calibration and validation anchors were checked for disjointness; each pool has identical anchor IDs and truth counts in every corresponding split.

**Complete true-control evaluation.**

| Stack | Threshold | Macro F0.5 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| raw | 0.675 | 0.9301889032678169 | 30335 | 884 | 4341 |
| target_arbitration | 0.675 | 0.9301889032678169 | 30335 | 884 | 4341 |
| margin_delta_0.05 | 0.550 | 0.92460055158757 | 28214 | 388 | 6462 |
| margin_delta_0.1 | 0.550 | 0.9311119631783249 | 29268 | 485 | 5408 |
| margin_delta_0.15 | 0.550 | 0.9329652235771562 | 29758 | 595 | 4918 |
| margin_delta_0.2 | 0.550 | 0.9336216172545162 | 30109 | 702 | 4567 |
| margin_delta_0.25 | 0.550 | 0.9332837681360503 | 30364 | 823 | 4312 |
| margin_delta_0.3 | 0.550 | 0.9329200771443266 | 30565 | 928 | 4111 |

**Historical corrections and verification.**

The four requested rows cannot honestly all be described as the same 81-feature training experiment. The saved Phase 7 report assigns 0.92874 to `enhanced_model_v3_recalibrated_rebuilt`; its separate calibration-only retrain scored 0.92463, and that retrained artifact is absent. Recalibrating the saved v3 estimator reproduces 0.9287367252062976 from raw validation pairs, with the same TP/FP/FN. The six-delta resweep also chooses δ=0.20. The historical row is retained with corrected attribution; no missing retrain artifact or exact retrain score is fabricated.

The Phase 4 code uses the 72-feature veto model, randomly allocates 535 presumed additional TPs and 25 FPs, then adds a hard-coded 0.0004 arbitration improvement to macro F0.5. Replaying that arithmetic reproduces 0.9372308853017451. It is neither a scored real expanded pool nor an 81-feature margin-gated result. This reference is not an achievable configuration or a target for further search.

Phase 8 was rescored from its saved model and rebuilt raw validation features; its best score reproduces exactly. Every reported operating point was independently checked by grouping selected pairs per anchor and recomputing scalar F0.5 plus integer TP/FP/FN. The saved control model passes reload prediction parity. The underlying full-grid threshold tables and per-anchor best scores are archived alongside the probability arrays.

The blocking work retains diagnostic value: it exposed a real missing-channel implementation gap and raised retained validation matches from 33,892 to 34,276 out of 34,676. That recall improvement does not overcome the observed precision cost with this model. This concludes the blocking diagnostic arc; no further feature or architecture investigation was started.

**Honest ceiling assessment.**

The current best matched 81-feature configuration achieves **0.9336216172545162**. If its 30,109 recovered TPs stay fixed and all FPs are removed, exact macro F0.5 is only **0.9536440397831156**. Precision cleanup alone therefore cannot reach 0.99; materially higher true-match recall is necessary.

For F0.5=0.99, even with precision at its ideal maximum, recall must be at least 0.99×0.25/(1.25−0.99) = **0.9519230769230769** (95.1923%). This is also a necessary mean-recall bound for macro F0.5 by concavity, not a sufficient operating condition. Mean per-anchor F0.5 is not F0.5 computed from pair recall or from averaged precision/recall.

The brief's stronger impossibility claim is not supported by the raw data. Selecting all retained true pairs and zero false pairs yields exact candidate-pool macro ceilings of **0.9919579527401471** for original cap 32 and **0.995766672715781** for rebuilt cap 64. Both exceed 0.99. Phase 8's saved pair-level ceiling was likewise above 0.99; it was not the correct macro ceiling. These oracle values do not demonstrate practical attainability, but they prevent claiming that external data is mathematically required.

The prior **0.938–0.945** range remains an unvalidated competitive aspiration, not a demonstrated attainable range. The fair result does not justify raising it or claiming it has been reached. The defensible current baseline is 0.9336216172545162; stop iterating on blocking.

Full audit: [evaluation.json](phase9/20260926T152905058159Z/evaluation.json), [additional verification](phase9/20260926T152905058159Z/additional_verification.json), [run log](phase9/20260926T152905058159Z/run.log). Reproduce the experiment with `.venv/bin/python -u -m src.phase9_true_control`; it creates a new timestamped run and model without replacing prior artifacts.
