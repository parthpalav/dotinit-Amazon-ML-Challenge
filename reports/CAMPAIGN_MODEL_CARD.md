# Confirmed contextual matching model

## Scope and artifacts
The selected pipeline resolves provided S2/S3 records against deduplicated S1. It uses only competition records/labels and generic text-processing libraries. No external business lookup, geocoding, or test ground truth is used. Training labels exist only for US and India; France is an unseen test country.

The pipeline needs both `artifacts/improvements/catboost_d10_evidence.joblib` and `artifacts/campaign_0931/oof_compact_d9.joblib`. The first supplies calibrated pair probabilities; the second combines them with candidate context. The earlier model is an input, not an obsolete file to delete. No pretrained neural network is used. Learned model weights are supplied under the MIT license in `CAMPAIGN_MODEL_LICENSE.txt`; third-party libraries keep their own licenses.

## Retrieval and features
The existing disk-backed union of 24 base and 64 supplemental keys retrieves candidates, and a supervised coarse ranker retains at most32 per S1. The fixed test pool has55,431,940 pairs across1,732,544 S1 records. Candidate-pool recall on the old10,000-anchor selection sample is33,892/34,676=97.739%, so retrieval still imposes a ceiling; missed pairs are included in reported false negatives.

The base matcher uses113 text, address, numeric, TF-IDF and corpus-frequency features. The compact reranker uses79 features: calibrated base probability, within-anchor score context, corroboration from up to six predicted candidate peers, and additional transliterated/phonetic/token/numeric comparisons. Peer selection uses predictions and text only, never peer labels. Original scripts remain available; transliteration is an additional view. Numbers are evidence, not universal hard-rejection rules. Generic AnyAscii transliteration is entirely offline.

## Training and selection
Base champion: CatBoost depth10,2,306 trees,30,000 fitting S1 anchors; sigmoid calibration on5,000 separate anchors. To expand reranker fitting without leaking in-sample base predictions, three additional depth10 models each exclude one third of those30,000 anchors and predict the excluded third. These out-of-fold probabilities generate contextual features. The reranker fits those30,000 anchors plus4,000 original calibration anchors; the remaining1,000 calibrate its probabilities and support early stopping.

Selected reranker: CatBoost depth9,2,030 trees, learning rate0.04, L2=8. Original10,000-anchor validation is explicitly a selection set, used for architecture and threshold choices. Selected threshold is0.6000000000000002. Simple probability ensembles did not outperform the selected single reranker. Both models are far below the8-billion-parameter limit.

## Independent checks
A fresh5,000-anchor confirmation excludes all55,000 anchors previously used for fitting, calibration, selection or earlier confirmation. All choices were frozen before its evaluation.

| Model | Macro F0.5 | TP | FP | FN |
|---|---:|---:|---:|---:|
| Previous113-feature champion |0.940297|15,325|336|1,923|
| Initial4,000-anchor compact reranker |0.954455|15,806|197|1,442|
| Selected34,000-anchor cross-fitted reranker |**0.959566**|**16,040**|214|**1,208**|

Paired bootstrap delta versus champion:+0.019269,95%CI[+0.016336,+0.022321]. Versus initial compact:+0.005111,CI[+0.002804,+0.007423]. US improved0.957770 ->0.967773; India0.912851 ->0.946675. These are training-derived confirmation scores, not Amazon scores. No labeled France estimate is available. Random-anchor validation understates competition among different S1 records for the same target.

## Global decisions and reproducibility
Every target is assigned only to its highest-scoring S1 if the score clears the threshold. Exact-score ties abstain. This respects deduplicated S1 semantics. The previous pipeline's Amazon result improved from0.920 to0.931 with unique ownership; approximately80% of removed duplicate assignments were in France.

`python -u -m src.campaign_finish --workers 6` resumes scoring, exports a new submission, and runs official matching-ID validation. Progress and model/code/data fingerprints are under `work/campaign_0931/test_final_oof`; stage status and logs are under `reports/campaign_0931`. CPU text processing/inference dominates the full run; RTX4060 was used for fitting. Every1,000-anchor scoring shard is reusable after interruption. Export verifies all candidate rows, IDs, probability ranges, matching subsets, empty anchors and unique ownership in bounded memory. The official validator additionally checks the matching file with `--check-ids`.

A separate one-hop alias expansion was also confirmed: a stricter0.975 cutoff for new pairs recovered37 true links with no additional false links on the fresh holdout (F0.5 0.960144; paired gain95%CI[0.000327,0.000896]). Its output preserves every accepted main owner and exports the expanded scored candidate pool. An added winner must also strictly exceed any tied main score; this safeguard changes zero decisions on the fresh confirmation set. `python -u -m src.campaign_run --main-workers 6 --alias-workers 4` completes both pipelines sequentially to fit16GB RAM. The preserved Amazon0.931 result is the fallback until a new Amazon evaluation is obtained.
