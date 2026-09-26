# Rethink after Amazon 0.942

Main 0.942; alias 0.942; France-only rollback 0.941 (user-reported overall scores). Preserve all artifacts. Country rollback is not supported. The 0.942 model remains the champion.

Next work: audit latest-model false positives/negatives against supplied training ground truth, inspect true competing owners, and measure whether sparse random-anchor validation misses global ownership errors. Avoid another unvalidated country mix or arbitrary leaderboard threshold sweep. Retraining or a new architecture should follow a measured failure mode. Any promotion needs independent confirmation and resumable inference. Work remains on parth; F is read-only.

## Active experiments, 2026-09-27

Official student README confirms macro per-S1 F0.5 including singletons, public/private test split, France absent from training, and **no test ground truth supplied**. All seven dataset files were inventoried. Public score equality at three decimals does not establish identical predictions or exact scores.

`latest_error_audit.json`: v4 has 214 FP, 791 in-pool FN and 417 missing true candidates. Of the FP, 104 have another true S1 owner, and zero of those owners appear in the small holdout. This establishes a validation blind spot, not the direction of its bias. Independent pair probability calibration is reasonably aligned on that holdout.

New `src/raw_evidence.py` retains legal-form tokens, one-to-one residual token alignment, short edits, slash/hyphen number groups and units. All are learned features, never unconditional rejection rules. Existing feature/scoring modules and champions are unchanged.

`python -u -m src.campaign_raw prepare` completed: 52,661 selection + 26,164 v3 + 26,503 v4 ambiguous pairs, main probability gate [0.005, 0.9999). Per-5k-row atomic caches in work/campaign_0942. These old validation sets are now development data. `python -u -m src.campaign_raw train` fits depth 5/7 CatBoost on the RTX 4060 (task_type GPU, device 0). Anchor split: 14k fit / 3k calibration / 3k selection, seed 20260929. Main-model probabilities are out of sample on all 20k anchors. A new untouched v5 is mandatory before promotion.

Research: Ditto (https://arxiv.org/abs/2004.00584) supports fine-tuned transformer pair classifiers. Reverse target-to-S1 retrieval / competing-owner context is the other experiment to prioritize. A public competition repository claims high validation scores, but its claims are not independently verified and are not evidence of Amazon leaderboard performance. No external identity data or other team's predictions are used.

Space: about 31 GB free on C; no deletion or model moves needed. F reference remains unchanged. New artifacts are separate from campaign_0931. Pending work is not yet pushed.

### Development results and active validation

Full raw correction d5: development F0.5 **0.965185**, vs current model **0.957738**, TP 9838 vs 9809, FP 100 vs 147. Threshold 0.70. Minimal raw correction d7 (only raw features + saved main probability): **0.964904**, TP 9823, FP 91, threshold 0.725. Both frozen before looking at new v5 outcomes. Minimal model is preferable operationally if independently supported because it avoids recomputing the 79-feature contextual model on 55 million test pairs. These are NOT Amazon scores.

`python -u -m src.campaign_confirm_raw` is running v5, 5k fresh anchors excluding all prior 60k. Original feature preparation was not batch-resumable and suffered paging; replaced only in this new runner with atomic 100-anchor base batches. Stops/restarts preserve completed batches. Current progress: reports/campaign_0942/raw_confirmation_v5.log. After it finishes, run `python -u -m src.campaign_confirm_minimal run`; its recipe is already frozen in minimal_frozen_v5.json. Do not retune on v5.

Profiler py-spy confirmed time spent in memory-mapped candidate lookup and PackedText.get, not stalled GPU training. Added separate src/native/sorted_lookup.cpp and src/sorted_lookup.py: identical legacy posting limits/rule masks, sorted query order for disk locality. Parity tests pass for both 24- and 64-key schemas, duplicated postings and multiple caps. Original production/index feature files remain unchanged. New v5 runner uses this lookup only; original champion artifacts are preserved. Five targeted tests pass (3 raw evidence, 2 native parity).

Reverse reference indexing is implemented in src/campaign_reverse.py but its first run was stopped to avoid RAM contention with v5. Resume `python -u -m src.campaign_reverse index`, then `... prepare`. It uses unlabelled source names/addresses only; no trained competitor scores, which could introduce in-sample advantage for background training owners. Candidate and feature caches are separate. Reverse evidence is experimental and not promoted.

Conditional production scorer is implemented: `python -u -m src.campaign_raw_scoring --workers 2`. It refuses full inference unless minimal_confirmation_v5.json supports promotion. Atomic 1000-anchor shards and per-shard input/output hashes allow resume. `--limit 2000` is a timing pilot only; it cannot mark full scoring complete. Once full scoring completes, export using existing src.finalize_improved with work/campaign_0942/test_minimal/selection.json, unique ownership, and a new output folder; then run official validator. No new full-test output exists yet.
