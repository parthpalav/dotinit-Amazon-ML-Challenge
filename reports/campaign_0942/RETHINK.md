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

### Fresh confirmation completed; production started (~03:27 IST)

v5 full correction: 0.95993880 -> **0.96725927**, TP 15973 -> 16052, FP 233 -> 156; paired 95% delta CI [0.005366, 0.009409]. Minimal correction: **0.96731724**, TP 16011, FP 143; paired 97.5% delta CI [0.005298, 0.009672]. Both hypotheses were frozen before v5 outcomes. Prefer minimal correction for production due to equivalent measured quality and much lower inference cost. Main 0.942 Amazon score remains the external champion until the new file is evaluated.

Full minimal scoring is RUNNING with two workers. Log: raw_production.log. Work: work/campaign_0942/test_minimal. Initial speed ~1,100 anchors/sec, approximately 26 minutes remaining at 32k anchors; this is measured throughput, subject to disk/compute contention. Safe resume command above. New output intended: outputs/campaign_0942_raw_unique.

Reverse source index build resumed and is sorting its last partitions. It is entirely unlabelled. Next run its `prepare` stage to measure competing owner evidence.

Neural experiment started: src/campaign_neural.py, Microsoft Multilingual-MiniLM-L12-H384, MIT, ~117M parameters (21M transformer + 96M embeddings), suitable for 8GB GPU. Model card: https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384. PyTorch 2.8.0 CUDA 12.6 is downloading; transformers 4.57.6 and sentencepiece 0.2.1 installed. Tokenization/model download stage running in neural_prepare.log. No neural training has started yet. Training command: `python -u -m src.campaign_neural train --batch 32 --epochs 3`; optimizer/model/RNG resume checkpoint every 500 steps. Original multilingual embeddings remain frozen. Same 14k/3k/3k development split; a neural ensemble will require its own untouched confirmation before promotion.

Code/audit checkpoint 4908622 was pushed to origin/parth. Later result logs and neural code still need the next checkpoint push. Generic pretrained weights are downloaded; no business records are sent to any external API. PDFs were re-read: max five submissions per day and deadline 27 Sep 2026 23:59 IST; preserve scarce submissions for meaningful variants.

### Runtime / handover update (~03:53 IST)

Checkpoint **2a89720** was pushed. Fresh-model result files are now in Git. `transfer/campaign-0942-models.zip` is a verified weights-only backup of the three required models, with MIT model license and a model card. The previous two models remain dependencies and must not be discarded. This ZIP is not a full data/index backup.

SQLite-based raw scoring completed 375k anchors before switching to a compact, lossless raw-text store. `src.raw_packed` streamed the identical raw names/addresses into memory-mapped fields. `src.campaign_raw_packed --parity` verified **1,200 exact feature and probability matches** across four widely separated test shards. Existing verified shards were hard-linked into the new run, with their original signature recorded in import.json. No model, gate, threshold, or candidate change occurred.

**ACTIVE production work is now `work/campaign_0942/test_minimal_packed`**, log `reports/campaign_0942/raw_packed_production.log`. Do not run both storage backends concurrently. Resume with:

```
python -u -m src.campaign_raw_packed --workers 2
```

After the current scoring process ends, finish/export/officially validate with:

```
python -u -m src.campaign_raw_finish --packed --workers 2
```

The finish command can itself resume scoring; **do not launch it while another scorer is running**. Output will be `outputs/campaign_0942_raw_unique`. Full export is not ready yet. New storage/algorithm files have LF rules in .gitattributes so frozen byte hashes survive Windows checkout.

Neural training is actually using RTX 4060: measured 89% GPU utilization, 2.2GB VRAM. PyTorch reports CUDA device explicitly. First epoch calibration logloss 0.13643; two passes finished/nearing completion, three configured. Main checkpoint `work/campaign_0942/neural_checkpoint.pt` saves model/optimizer/scheduler/RNG every 500 steps. Resume original training command with batch32/epochs3. After training: `python -u -m src.campaign_neural infer --batch 32`, then `python -u -m src.campaign_neural_select`. These test logistic and small-tree fusion on disjoint development anchors. Neither neural model nor fusion is promoted yet.

Reverse preparation's first broad key set produced excessive I/O. That partial cache is preserved but unused. The current restricted competitor search uses exact name/address, name LSH, and name+street-number keys; disables generic single-token/postal/address-LSH postings; caps postings at120 and retains top4 alternatives by text strength. It is background-competitor evidence, not the forward candidate blocker. New caches: `competitors_name_address_*`; old `competitors_*` partials must not be mixed. The run was paused during dependency installation disk contention, then resumed. After prepare completes: `python -u -m src.campaign_raw train --reverse`.

v5 is now consumed for diagnostic ceiling calculations; any new neural/reverse promotion requires a fresh holdout. Current-pool v5 oracle = 0.9924558 with 396 omitted true pairs. This is a local bound, not a test-score promise. Raw model feature importance confirms legal distinction evidence is useful, but never a hard rule.
