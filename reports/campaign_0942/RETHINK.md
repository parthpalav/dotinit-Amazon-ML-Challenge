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


### New submission ready; multilingual model trained (~04:10 IST)

`outputs/campaign_0942_raw_unique/matching_results.tsv` is READY: 1,732,544 anchors, 5,685,947 matches, 108,690 empty; official matching-ID validator PASS and bounded full candidate equality/subset/unique-owner checks PASS. Matching SHA256 b261646dd222fbc210aea72b13c90c28dc3880b1de15ecb2e4727f0413891936. The 55,431,940-pair candidate file is unchanged. Amazon score unknown; 0.942 remains the evaluated champion. Raw production fully completed in `work/campaign_0942/test_minimal_packed`; no need to resume it.

Neural training completed all three epochs on RTX4060. Best checkpoint epoch1, calibration BCE 0.10716694; epoch2 was slightly worse. Best model: artifacts/campaign_0942/neural_minilm. Development predictions and resumable training checkpoint preserved. Corrected full-3,000-anchor selection: raw 0.964904, neural_stack_d3 0.973332, joint_stack_d3 0.974975. Thresholds frozen at 0.675 and 0.725 respectively. These are development scores only.

Found/fixed a development metric bug: neural/joint selection IDs were derived from gated rows, omitting 46 anchors without ambiguous pairs. Both selectors now reconstruct the complete original anchor split, including those anchors; subset helper supports explicit IDs, including anchors with no candidates. Re-ran selectors. Previously reported neural/joint numbers are superseded. Raw v5 confirmation and the ready raw submission are unaffected.

Restricted reverse search recovers only 20 of 417 missing v4 true pairs, leaving397. Its value is competing-owner context, not broad recall expansion. This diagnostic queries known positives to measure coverage only; no label-selected candidates are deployed.

Fresh v6 confirmation RUNNING via `python -X utf8 -u -m src.campaign_confirm_v6`, log confirmation_v6.log. Five thousand anchors exclude all earlier 65k; seed20260930. Models/tokenizer and recipes are frozen before accessing outcomes. Paired bootstrap: neural promotion needs positive 97.5% delta interval vs raw; joint needs positive 97.5% vs raw AND95% vs neural. Atomic 100-anchor base and prediction checkpoints. Do not retune on v6. No neural/joint test submission exists yet. Next: finish v6, implement bounded/resumable full test inference for supported model, export and validate. New production serialization must preserve actual country; the raw-only packed adapter intentionally has blank country and cannot be used unchanged for neural input.

Checkpoint bded698 is on origin/parth; subsequent ready output and neural results await the next checkpoint. Old models are still required dependencies. No files removed or moved; F reference untouched.


### Fresh v6 passed; neural full-test run started (~04:24 IST)

Frozen fresh v6: raw F0.5 0.96621632 (TP16256/FP126/FN1203), neural 0.97373510 (TP16482/FP119/FN977), joint 0.97460373 (TP16481/FP97/FN978). Neural delta +0.0075188 with 97.5%CI [0.0050327,0.0101339]. Joint vs raw also positive; joint vs neural95%CI [-0.00036385,0.00210446], so the preregistered rule selects **neural_stack_d3**. Joint remains experimental. Threshold0.675, batch64, bf16, full192-token padding. Confirmation file: confirmation_v6.json. v6 is now consumed; no retuning on it.

Full pipeline RUNNING: `python -X utf8 -u -m src.campaign_ensemble_run --workers 2`. This reuses completed neural pilot shards, scores all test candidates using the frozen gate, exports to outputs/campaign_0942_neural_unique, runs bounded full integrity checks and official ID validator. It does NOT automatically deploy the joint model because joint was not selected. Log ensemble_run.log; neural_production.log; state ensemble_production.json; detailed work/campaign_0942/test_neural/progress.json. Per1000-anchor score and neural-component caches have input/output hashes and can resume. Do not launch duplicate GPU scorers. Expected GPU inference roughly1-3hours from the warm pilot, subject to sustained throughput; measure progress after several shards. Initial load time distorts early estimates.

`src.campaign_test_records` provides full packed raw+normalized records with actual country preserved, unlike raw-only adapter. All fields matched SQLite exactly on4,004 sampled/end-point records. `src.campaign_neural_test` uses one CPU preparation thread and GPUdevice0, bounded memory. `src.campaign_joint_test` is available for future supported joint inference, but is not running and requires a reverse_test index. Source1 remains one-to-many; only target ownership is unique.

Performance experiment src.neural_batched tested length sorting/batch256. Across11,750 unlabelled test pairs it changed0 neural-fusion threshold decisions but introduced small probability changes (max0.004817, mean~3e-6). It is NOT used for production, preserving exact confirmation inference settings. v5 gate audit: upper-gate1804pairs all true; lower-gate131956pairs contained27true links; all143raw FP within the corrected gate. No expensive gate expansion justified.

`transfer/campaign-0942-ensemble-models.zip` verified; includes old dependencies, both fusion/reverse models, fine-tuned neural weights/config/tokenizer. ZIP paths normalized to forward slashes (fixed a Windows archive verification error). Dataset/index/score caches remain separate. Source/reports/ready raw output checkpoint e28cbfb successfully pushed. Later neural code and confirmation await next checkpoint. F reference remains untouched; no models deleted.


### Sustained runtime update (~04:27 IST)

Neural inference has passed19,000anchors, with100,834 ambiguous pairs processed in136.5s this run (plus2,000 pilot anchors reused). Current estimate ~3.8hours remaining for scoring; earlier1-3hour pilot estimate was optimistic. Allow export/official validation afterwards. GPU observed98%utilization,832MiB VRAM,76C; profiler shows the main thread receiving GPU predictions and preparation worker idle, so compute is active and not stalled. The automatic end-to-end process remains running and will export/validate without a manual command. Do not close the computer or start another scorer. Resume after interruption with the same campaign_ensemble_run command. It preserves completed score shards and verifies hashes before reusing them.


### Verification checkpoint (~04:29 IST)

Full regression run:66 passed,1 failed solely because the Windows test child wrote cp1252 while its UTF8 parent decoded stdout. Fixed the test subprocess to request `-X utf8` and explicit UTF8 decoding; targeted rerun passed (including rejection of invalid S1-as-target IDs). Logs regression_tests.log and validator_test_recheck.log retain both evidence and repair. Actual production already explicitly runs child Python in UTF8 and raw official validation passed. All67 tests are therefore verified across the full run plus the repaired-test rerun.

Neural production continues (34k anchors at04:28, sustained estimate~3.3hours then); detailed progress is updated every1000anchors. C has20.5GB free; F1149GB free. No offloading/deletion needed. Verified ensemble ZIP rebuilt with the fresh result and updated model card. The current pipeline is fully scripted through output validation, so no manual step is required after scoring. Amazon evaluation still requires uploading the resulting matching_results.tsv; no new Amazon score has been reported.
