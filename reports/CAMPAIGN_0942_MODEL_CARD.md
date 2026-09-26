# Raw-text correction model

The evaluated Amazon champion remains the previous 0.942 pipeline until a new submission is scored. This campaign adds a learned correction to its pair probabilities; it does not assume local scores equal Amazon scores.

## Inputs and method

The original candidate graph and the two existing models are retained. The depth-10 evidence model produces base probabilities; the cross-fitted depth-9 contextual model produces the current main probabilities. A depth-7 CatBoost correction uses the main probability and raw name/address evidence: legal tokens retained before normalization, one-to-one token alignment, short edits, and numeric groups/units. Legal or numeric disagreements are not hard vetoes.

The correction applies only for main probabilities in [0.005, 0.9999); predictions outside that gate remain unchanged. The correction has 1,353 trees and a frozen pair threshold of 0.7250000000000003. For export, each target belongs only to its highest-scoring source above threshold; exact highest-score ties abstain. Every original candidate remains represented in candidate_pairs.tsv.

## Training and validation

The main model was not trained on the labels of the 20,000 anchors used to develop this correction. These were the original 10k selection set and the old v3/v4 5k sets. They are now development data, not independent confirmation. Seed 20260929 separates 14k fitting anchors, 3k early-stopping/sigmoid-calibration anchors, and 3k architecture/threshold-selection anchors. CatBoost training explicitly used GPU device 0, the RTX 4060.

A further 5,000 anchors (v5), excluding all prior 60,000 anchors, were reserved. Both full-context and minimal correction hypotheses were frozen before viewing v5 outcomes. The minimal version was retained because it has equivalent measured quality and substantially lower inference cost.

| Fresh v5 metric | Previous main | Minimal correction |
|---|---:|---:|
| Macro F0.5 | 0.9599388 | 0.9673172 |
| True matches | 15,973 | 16,011 |
| False matches | 233 | 143 |
| Missed matches | 1,176 | 1,138 |
| Singleton accuracy | 0.95423 | 0.97535 |

Paired improvement: +0.0073784; 97.5% bootstrap interval [0.0052981, 0.0096715]. The original candidate-pool oracle on this holdout is 0.9924558. That is a diagnostic bound, not an achievable/test-score claim.

## Limitations

France is absent from the supplied training labels. Random-anchor holdouts also do not by themselves reproduce full-graph ownership competition. Test labels are not supplied. Three-decimal Amazon scores can conceal small differences. The previous alias submission changed only 14,445 links and did not demonstrate a public-score gain. The France rollback scored 0.941 and was rejected.

## Reproduce / resume

Use branch `parth` and the tested base environment. Existing dataset, indexes, and main-model score shards are required. Keep both earlier model files: the correction is an additional stage, not a replacement for their input scores.

1. `python -u -m src.campaign_raw prepare`
2. `python -u -m src.campaign_raw train --minimal`
3. Use the frozen confirmation artifacts and `src.campaign_confirm_raw` / `src.campaign_confirm_minimal` to reproduce v5.
4. `python -u -m src.campaign_raw_finish --workers 2` resumes original SQLite-backed correction, exports, and officially validates.

The optional packed raw-text backend changes only storage, with exact feature/prediction parity checks. Its separate work directory and cache-import provenance are documented in `reports/campaign_0942/RETHINK.md`. No output is ready until production_plan.json reports `ready_for_amazon_evaluation`.

Model license: the trained correction follows the campaign's existing MIT model-license declaration; CatBoost is Apache 2.0. No external business identity lookup or augmentation was used. The separate multilingual MiniLM experiment uses MIT-licensed generic pretrained weights and is not part of this submission unless independently validated and explicitly documented later.


## Confirmed multilingual ensemble (2026-09-27)

A fine-tuned microsoft/Multilingual-MiniLM-L12-H384 (MIT,117,654,530parameters) classifies raw name/address/country record pairs. XLMRobertaTokenizer, max192tokens; frozen embeddings; three epochs on the same14k training anchors, AdamW4e-5, batch32, bfloat16 on RTX4060. Epoch1 is best by3k calibration BCE0.10716694. The pair serialization preserves names, legal endings, addresses and country. A depth3/350-tree CatBoost fuses logits from original contextual model, raw correction and neural classifier, trained on3k calibration anchors. Threshold0.675 was chosen on the separate3k development anchors.

New untouched v6 excludes all earlier65k anchors. Raw0.9662163 -> neural0.9737351; paired97.5% deltaCI [0.0050327,0.0101339]. TP16256->16482, FP126->119, FN1203->977. A four-input global-owner alternative scored0.9746037 but its advantage over neural was not statistically established, so it is not promoted. Full inference uses the same batch64/bfloat16/padded192-token settings as confirmation. Unique-owner export remains unchanged. Global labels are not features. No test truth exists in the supplied resources.

Resume all confirmed neural work using `python -X utf8 -u -m src.campaign_ensemble_run --workers 2`. Required environment: base requirements plus requirements-neural.txt. Model ZIP: transfer/campaign-0942-ensemble-models.zip (weights only). Expected output outputs/campaign_0942_neural_unique is not ready until ensemble_production.json marks ready_for_amazon_evaluation. Raw output is already ready.
