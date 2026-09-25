"""Render the real experiment report from completed, measured artifacts only."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .config import Config


def table(frame, columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in frame[columns].itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return lines


def generate(config):
    reports=Path(config.reports_dir)
    def read(name):return json.loads((reports/name).read_text())
    run=read('real_run_manifest.json');quality=read('real_data_quality.json')
    audit=read('full_training_blocking.json');inference=read('test_inference.json')
    validation=read('official_validator_command.json');sampling=read('entity_sampling.json')
    if validation['exit_code']!=0:raise ValueError('Official validation did not pass')
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        if not (Path(config.output_dir)/name).is_file():raise FileNotFoundError(name)
    lines=['# Real dataset results','',
        'All model fitting and final predictions use the supplied competition records. No external business lookup or synthetic final data is used. The README specifies one matching task across three sources, with two output files covering every test Source 1 entity.','',
        '## Dataset and canonical mapping','',
        '| Real column | Internal use |','|---|---|',
        '| entity_id | Preserved string ID; S1 anchor or S2/S3 target; never a numeric model feature |',
        '| business_name | Preserved raw name; name_norm adds Unicode NFKC, casefolding, punctuation/abbreviation/suffix normalization |',
        '| business_address | Preserved raw address; address_norm plus uncertain city/state/postal/street-number hints |',
        '| country | Preserved open-set label; country_norm equality, no fixed-country one-hot encoding |',
        '| source1_entity_id, matched_entity_ids | Training-only anchor-to-target labels; empty list explicitly means singleton |','',
        'No dedicated contact or structured address columns exist. Explicit phone/email/URL text is extracted conservatively from the original strings; no external enrichment occurs.','',
        '| Source | Train rows / unique IDs | Test rows / unique IDs | Train missing addresses | Test missing addresses |',
        '|---|---:|---:|---:|---:|']
    for source in (1,2,3):
        a=quality[f'dataset/train/train_source{source}.tsv'];b=quality[f'dataset/test/test_source{source}.tsv']
        lines.append(f"| {source} | {a['counts']['rows']:,} / {a['counts']['unique_ids']:,} | {b['counts']['rows']:,} / {b['counts']['unique_ids']:,} | {a['missing']['business_address']:,} | {b['missing']['business_address']:,} |")
    gt=quality['dataset/train/train_ground_truth.tsv']['counts']
    lines+=['',f"Ground truth: {gt['positive_pairs']:,} positive links and {gt['singletons']:,} singleton anchors. Negatives are the complement of positive labels in each retrieved candidate set; no explicit negative-pair file is supplied. Exact raw fingerprint repeats are potential duplicate records, not confirmed entity identities. Full missingness, Unicode, contact hints, duplicates and train/test overlap are in `real_data_quality_report.md` and `real_data_quality.json`.",'',
        '## Sampling and leakage controls','',
        'A fixed-seed permutation assigns disjoint Source 1 groups to fitting, calibration and validation. All final candidates for each sampled anchor are retained. Positive pairs cannot cross these anchor groups because ground truth enforces unique target ownership. Shared reference records can appear as negative candidates across groups. TF-IDF uses fitting anchors and 70,000 random training reference records; this shares some reference text with held-out entities, but no held-out labels. Validation is used for model, probability-variant and threshold selection; these are selection scores, not an untouched generalization estimate. France occurs only in test, so validation does not measure that distribution shift.','']
    lines+=table(pd.DataFrame([{'split':k,**v} for k,v in sampling['groups'].items()]),['split','assigned_entities','sampled_entities'])
    lines+=['','Candidate label distribution:','']
    lines+=table(pd.DataFrame([{'split':k,**v} for k,v in run['training_candidate_labels'].items()]),['split','positive','negative'])
    c=audit['counts']
    lines+=['','## Blocking','',
        f"Disk-backed SQLite records and memory-mapped native indexes avoid a Cartesian product. Complementary exact-name, country/name-token, name/address character minhash, address-token-bag, country/postal, name/number, address/number and name-token-pair keys are unioned. Keys above {config.retrieval_posting_limit} postings are suppressed (exact name/address allow five times this limit). The fitting-only XGBoost retrieval ranker retains at most {config.max_candidates_per_anchor} candidates per anchor. This cap is explicit and its recall loss is included below. The legacy city-name flag is inactive in the disk backend.",'',
        f"Full training audit: {c['anchors']:,} anchors; {c['true_pairs']:,} true pairs; {c['raw_candidates']:,} raw candidate pairs; {c['candidate_pairs']:,} scored-stage candidates; {c['retained_true']:,} retained true pairs. Raw recall {audit['raw_candidate_recall']:.6%}; final recall {audit['candidate_recall']:.6%}; reduction ratio {audit['candidate_reduction_ratio']:.9%}; mean {audit['average_candidates']:.4f}, maximum {c['max_candidates']} final candidates per anchor.",'',
        'Per-rule figures overlap because retrieval is a union. The full audit includes fitting entities used by the ranker; consult validation recall for held-out retrieval performance.','']
    rules=[]
    for rule,count in audit['rule_candidates'].items():
        rules.append({'rule':rule,'raw_candidate_pairs':count,'raw_true_pairs':audit['raw_rule_true'][rule],
                      'raw_recall':audit['raw_rule_true'][rule]/c['true_pairs'],'retained_true_pairs':audit['final_rule_true'][rule]})
    lines+=table(pd.DataFrame(rules),['rule','raw_candidate_pairs','raw_true_pairs','raw_recall','retained_true_pairs'])
    pilot=read('retrieval_ranker_pilot.json')
    lines+=['','The current retrieval-ranker pilot uses 1,000 held-out fitting anchors (no calibration/validation labels). Its top-k comparison is:','']
    lines+=table(pd.DataFrame([{'candidate_cap':int(k),'retained_true_pairs':pilot[f'retained_top_{k}'],
                              'recall':v} for k,v in pilot['recall'].items()]),
                 ['candidate_cap','retained_true_pairs','recall'])
    historical=reports/'prior_run/posting_limit_pilot.json'
    if historical.exists():
        history=json.loads(historical.read_text())
        lines+=['','Historical posting-limit pilot retained from the previous work (same raw TSV hashes; the 120/500 settings were not rerun here). The 240-posting result was reproduced by the current pilot. The chosen 240/32 configuration trades some recall for substantially fewer raw comparisons; the held-out validation recall remains below the 98% target.','']
        lines+=table(pd.DataFrame([{'posting_limit':r['posting_limit'],
                                   'raw_candidates':r['counts']['raw_candidates'],
                                   'final_candidates':r['counts']['candidate_pairs'],
                                   'final_recall':r['candidate_recall']} for r in history]),
                     ['posting_limit','raw_candidates','final_candidates','final_recall'])
    lines+=['','## Features','',', '.join(run['features'])+'.','',
        '## Four-model comparison','',
        'Precision, recall, F0.5 and F1 are per-anchor macro averages including correct empty/empty singletons as 1. PR-AUC and ROC-AUC are candidate-pair diagnostics and exclude positives lost during retrieval; macro recall/F0.5 include those losses.','']
    comparison=pd.read_csv(reports/'model_comparison.tsv',sep='\t')
    lines+=table(comparison,['model','probability_variant','precision','recall','f0.5','f1','pr_auc','roc_auc','threshold'])
    lines+=['','Candidate confusion matrices use [[TN, FP], [FN, TP]]; full-space false negatives additionally include blocking misses:','']
    lines+=table(comparison,['model','confusion_matrix','predicted_positive','predicted_negative_candidates','false_negative','runtime_seconds'])
    lines+=['','## Calibration and threshold','',
        'Sigmoid calibrators are fitted only on the disjoint calibration group. Raw versus calibrated selection uses validation macro F0.5, then precision and Brier score. Exact-rule probabilities are unchanged. Lower Brier/log loss is better.','']
    calibration=pd.read_csv(reports/'calibration_comparison.tsv',sep='\t')
    lines+=table(calibration,['model','probability_variant','brier','log_loss','threshold','f0.5'])
    lines+=['',f"Selected model: **{run['selected_model']}**; threshold **{run['selected_threshold']:.8g}**; validation macro F0.5 **{run['validation']['f0.5']:.6f}**. Selection is based only on training-derived validation data. Full threshold sweeps are saved as `thresholds_*_*.tsv`. No test-based threshold adjustment or uncalibrated refit occurs.",'',
        '## Final inference and singletons','']
    c=inference['counts']
    lines += [f"All {c['anchors']:,} test Source 1 anchors are exported once in source order against both test reference sources, including France. {c['scored_pairs']:,} candidates were scored; {c['predicted_matches']:,} accepted and {c['predicted_nonmatches']:,} rejected. Empty matching lists: {c['predicted_singletons']:,} ({c['predicted_singletons']/c['anchors']:.4%}). Candidate counts: zero={c['no_candidate']:,}, exactly one={c['one_candidate']:,}, multiple={c['multiple_candidates']:,}. No match is forced.",'']
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        path=(Path(config.output_dir)/name).resolve();lines.append(f'- `{path}` ({path.stat().st_size:,} bytes)')
    lines+=['','## Official validation','',f"Unmodified supplied validator SHA-256: `{validation['validator_sha256']}`. Exit code: **{validation['exit_code']}**. Both TSVs were checked together with `--check-ids`.",'','```text',(reports/'official_validator.log').read_text().rstrip(),'```','',
        '## Runtime and memory','',f"Training stage: {run['runtime_seconds']:.1f} seconds; full blocking audit: {audit['seconds']:.1f} seconds; inference: {inference['seconds']:.1f} seconds. Reported high-water RSS includes mapped pages and is per process, not aggregate physical memory. Parent training peak: {run['peak_parent_rss_mb']:.1f} MiB; inference worker peak: {inference['worker_peak_rss_mb']:.1f} MiB. Native index builds, quality scan and validation have additional runtime. Elapsed timings include long pauses visible in the log and are not CPU-time benchmarks.",'',
        '## Tests','']
    for filename in ('original_tests.log','final_tests.log'):
        path=reports/filename
        if path.exists():lines+=['```text',path.read_text().rstrip(),'```','']
    lines+=['## Reproduction','',
        'Run from the repository root with the supplied resource directory at `../student_resource`. Python 3.13, a C++17 compiler and macOS OpenMP are used. Keep raw data unchanged. A new working directory is required after changing retrieval, normalization, sampling or feature settings; do not reuse stale caches.','',
        '```bash','python3 -m venv .venv','.venv/bin/python -m pip install -r requirements-tested.txt',
        '# macOS, if OpenMP is absent: brew install libomp',
        '.venv/bin/python -m src.real_pipeline run --config config/real.json',
        '.venv/bin/python -m pytest -q > reports/final_tests.log',
        '.venv/bin/python -m src.real_report --config config/real.json','```','',
        'The `run` command performs quality → index build → four-model fitting/calibration/selection → full training retrieval audit → full test inference → supplied official validator. Stage commands `quality`, `prepare`, `train`, `audit`, `infer`, `validate` support inspection and reruns. Raw files are never changed. Saved feature caches and model artifacts contain only real data.']
    destination=reports/'real_dataset_results.md';destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/real.json')
    print(generate(Config.load(parser.parse_args().config)))
