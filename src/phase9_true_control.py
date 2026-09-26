"""Final candidate-pool comparison. Each invocation writes a new archival run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.enhanced_features import compute_enhanced_features
from src.veto_features import compute_veto_features
from src.model import CalibratedMatcher
from src.global_arbitration import arbitrate
from src.phase8_full_retrain import apply_margin_gating
from src.real_metrics import entity_scores, tune


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size,
                mtime=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                sha256=h.hexdigest())


def load_split(base, split, names, audit):
    path = Path(base) / 'features' / f'{split}.joblib'
    data = joblib.load(path)
    pairs = data['pairs']
    original = base == 'work/real_v1'
    paths = ([Path('work/improvements/veto_cache') / f'{split}_veto.joblib',
              Path('work/improvements/enhanced_cache') / f'{split}_enhanced.joblib']
             if original else [Path(base) / 'supplemental_cache' / f'{split}_{k}.joblib'
                               for k in ['veto', 'enh']])
    supplemental = [joblib.load(p) for p in paths]
    assert [len(x.columns) for x in supplemental] == [14, 9]
    assert all(len(x) == len(pairs) for x in supplemental)
    assert all(x.index.equals(pairs.index) for x in supplemental)
    # Legacy caches have no pair-key manifest: check deterministic samples against records.
    sample = np.unique(np.r_[0, len(pairs)-1,
                            np.random.default_rng(42).choice(len(pairs), 512, replace=False)])
    with sqlite3.connect('file:work/real_v1/train/records.sqlite?mode=ro', uri=True) as con:
        for cached, fn in zip(supplemental, [compute_veto_features, compute_enhanced_features]):
            fresh = fn(pairs.iloc[sample], con)
            np.testing.assert_allclose(cached.iloc[sample].to_numpy(), fresh.to_numpy(), rtol=0, atol=0)
    X = pd.concat([data['features'].reset_index(drop=True)] +
                  [x.reset_index(drop=True) for x in supplemental], axis=1)[names]
    assert X.shape == (len(pairs), 81) and np.isfinite(X.to_numpy()).all()
    c = data['stats']['counts']
    assert len(data['truth_counts']) == c['anchors'] == pairs.source1_entity_id.nunique()
    assert len(pairs) == c['candidate_pairs']
    assert int(pairs.label.sum()) == c['retained_true']
    assert sum(data['truth_counts'].values()) == c['true_pairs']
    assert not pairs.duplicated(['source1_entity_id', 'candidate_entity_id']).any()
    audit[f'{base}/{split}'] = dict(**fingerprint(path), anchors=c['anchors'], pairs=len(pairs),
        positives=int(pairs.label.sum()), true_pairs=c['true_pairs'],
        candidate_recall=int(pairs.label.sum()) / c['true_pairs'],
        max_candidates=int(pairs.groupby('source1_entity_id').size().max()),
        supplemental=[fingerprint(p) for p in paths],
        supplemental_verified_sample_rows=len(sample), cache_check='shape, index and exact sampled recomputation')
    print('Verified', base, split, X.shape, flush=True)
    return data, X


def independently_check(data, probabilities, threshold, scores):
    selected = data['pairs'].loc[probabilities >= threshold]
    grouped = selected.groupby('source1_entity_id').label.agg(['sum', 'size'])
    tp_total = fp_total = fn_total = 0
    f = []
    for anchor, actual in data['truth_counts'].items():
        tp, predicted = grouped.loc[anchor] if anchor in grouped.index else (0, 0)
        tp_total += int(tp)
        fp_total += int(predicted - tp)
        fn_total += int(actual - tp)
        f.append(1.0 if actual == predicted == 0 else 5 * tp / (actual + 4 * predicted))
    assert (tp_total, fp_total, fn_total) == tuple(scores[k] for k in
        ['true_positive', 'false_positive', 'false_negative'])
    assert abs(float(np.mean(f)) - scores['f0.5']) < 1e-14
    return np.asarray(f)


def evaluate(data, p, folder, label):
    pairs = data['pairs']
    truth = data['truth_counts']
    pos = {v: i for i, v in enumerate(truth)}
    ix = pairs.source1_entity_id.map(pos).to_numpy(dtype=np.int32)
    counts = np.array(list(truth.values()))
    y = pairs.label.to_numpy()
    arb = np.where(arbitrate(pairs, p, max_per_source=None), p, 0.)
    variants = [('raw', p), ('target_arbitration', arb)] + [
        (f'margin_delta_{d}', apply_margin_gating(pairs, arb, d))
        for d in [.05, .10, .15, .20, .25, .30]]
    results = {}
    per_anchor = {}
    for name, probabilities in variants:
        threshold, table = tune(ix, y, probabilities, counts, 10320219)
        scores = entity_scores(ix, y, probabilities, counts, threshold, 10320219)
        per_anchor[name] = independently_check(data, probabilities, threshold, scores)
        results[name] = dict(threshold=threshold, **scores)
        table.to_csv(folder / f'{label}_{name}_thresholds.tsv', sep='\t', index=False)
        print(label, name, json.dumps(results[name]), flush=True)
    best = max(results, key=lambda k: results[k]['f0.5'])
    np.save(folder / f'{label}_probabilities.npy', p)
    pd.DataFrame({'anchor': list(truth), 'f05': per_anchor[best]}).to_csv(
        folder / f'{label}_best_per_anchor.tsv', sep='\t', index=False)
    oracle = entity_scores(ix, y, y.astype(float), counts, .5, 10320219)
    independently_check(data, y.astype(float), .5, oracle)
    return dict(states=results, best_config=best, best=results[best],
                oracle_all_retained_true_only=oracle)


def main():
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = Path('reports/improvements/phase9') / run
    folder.mkdir(parents=True, exist_ok=False)
    model_path = Path('artifacts/improvements') / f'phase9_true_control_{run}.joblib'
    audit = {}
    prior = joblib.load('artifacts/improvements/enhanced_model_v3.joblib')
    names = prior.estimator.feature_names_
    datasets = {s: load_split('work/real_v1', s, names, audit)
                for s in ['fit', 'calibration', 'validation']}
    anchors = [set(d[0]['truth_counts']) for d in datasets.values()]
    assert all(not anchors[i] & anchors[j] for i in range(3) for j in range(i))
    fit, xf = datasets['fit']; cal, xc = datasets['calibration']; val, xv = datasets['validation']
    assert len(fit['truth_counts']) == 30000
    cb = CatBoostClassifier(iterations=1200, depth=8, learning_rate=.06, l2_leaf_reg=5,
        loss_function='Logloss', eval_metric='Logloss', random_seed=42, thread_count=6,
        allow_writing_files=False, verbose=100)
    start = time.time()
    cb.fit(xf, fit['pairs'].label.to_numpy(), eval_set=(xc, cal['pairs'].label.to_numpy()),
           early_stopping_rounds=100)
    elapsed = time.time() - start
    model = CalibratedMatcher(cb).calibrate(xc, cal['pairs'].label.to_numpy())
    joblib.dump(model, model_path)
    control = evaluate(val, model.predict(xv), folder, 'true_control')
    # Reload saved model to verify serialization parity.
    np.testing.assert_array_equal(joblib.load(model_path).predict(xv.iloc[:512]), model.predict(xv.iloc[:512]))
    del xf, xc, xv, datasets
    rebuilt, xr = load_split('work/cap64_rebuilt', 'validation', names, audit)
    assert val['truth_counts'] == rebuilt['truth_counts']
    phase8_path = 'artifacts/improvements/phase8_full_fit_retrain_model.joblib'
    phase8_model = joblib.load(phase8_path)
    assert phase8_model.estimator.feature_names_ == names
    phase8 = evaluate(rebuilt, phase8_model.predict(xr), folder, 'phase8')
    old8 = json.loads(Path('reports/improvements/phase8_full_retrain_evaluation.json').read_text())
    assert abs(phase8['best']['f0.5'] - old8['comparison']['phase8_best_f05']) < 1e-14
    # The Phase 7 report attributes 0.92874 to v3 recalibration, not the partial retrain.
    rc, xrc = load_split('work/cap64_rebuilt', 'calibration', names, audit)
    prior.calibrate(xrc, rc['pairs'].label.to_numpy())
    phase7 = evaluate(rebuilt, prior.predict(xr), folder, 'phase7_recalibrated')
    phase7_model_path = folder / 'phase7_recalibrated_reconstruction.joblib'
    joblib.dump(prior, phase7_model_path)
    quoted7 = phase7['states']['margin_delta_0.2']
    assert round(quoted7['f0.5'], 5) == .92874
    # Preserve and qualify the original hypothetical result; it is a 72-feature simulation.
    historic4 = json.loads(Path('reports/improvements/veto_model_validation_v2.json').read_text())
    sim4 = historic4['state_4_combined']
    assert round(sim4['f0.5'], 5) == .93723
    gain = phase8['best']['f0.5'] - control['best']['f0.5']
    rebuilt_wins = gain > 0
    winner = phase8 if rebuilt_wins else control
    selected_model = Path(phase8_path) if rebuilt_wins else model_path
    base = 'work/cap64_rebuilt' if rebuilt_wins else 'work/real_v1'
    config = dict(status='final_recommended_production_baseline', run=run,
        candidate_pool=base, max_candidates_per_anchor=64 if rebuilt_wins else 32,
        retrieval_posting_limit=500 if rebuilt_wins else 240,
        phonetic_relaxed_address_channels=rebuilt_wins,
        existing_index=str(Path(base) / 'train/supplement_index.bin'),
        model=fingerprint(selected_model), features=names,
        feature_engineer='work/real_v1/feature_engineer.joblib',
        candidate_ranker='work/real_v1/candidate_ranker.joblib',
        supplemental_functions=['src.veto_features.compute_veto_features',
                                'src.enhanced_features.compute_enhanced_features'],
        enhanced_preliminary_probabilities=None,
        target_arbitration=dict(enabled=winner['best_config'] != 'raw', max_per_source=None,
                                tie_break='stable input pair order'),
        margin_delta=float(winner['best_config'].split('_')[-1]) if winner['best_config'].startswith('margin') else None,
        margin_order='after target arbitration; group by anchor and candidate source',
        threshold=winner['best']['threshold'], comparison='probability >= threshold',
        validation_f05=winner['best']['f0.5'],
        selection_caveat='Threshold and delta selected on validation, matching Phase 8; not a fresh held-out estimate.',
        scope='Recommendation lock; no test inference or deployment performed. Use preserved index, not regeneration with changed C++ channels.')
    (folder / 'production_config.json').write_text(json.dumps(config, indent=2) + '\n')
    result = dict(run=run, title='True Control: 81-Feature Model, Original Candidate Pool (Cap 32)',
        provenance=audit, model=fingerprint(model_path), training_seconds=elapsed,
        parameters=cb.get_params(), best_iteration=cb.get_best_iteration(),
        true_control=control, phase8=phase8, phase7_recalibrated=phase7,
        phase7_historical_quoted_stack=quoted7,
        phase7_label_correction='0.92874 is v3 recalibrated on rebuilt calibration; reported calibration-only retrain scored 0.92463. That retrain artifact is absent.',
        phase4_reference=sim4,
        phase4_caveat='72-feature veto model plus simulated TPs/FPs and hard-coded arbitration score offsets; not an 81-feature production configuration or margin-gated comparison.',
        rebuilt_minus_control=gain, winner='rebuilt' if rebuilt_wins else 'original',
        production_config=config,
        ceiling_caveat='Use exact mean per-anchor oracle F0.5, not pair recall substituted into macro formula. Recall alone does not prove 0.99 impossible if oracle exceeds 0.99.',
        mean_recall_necessary_for_099_at_perfect_precision=.99*.25/(1.25-.99),
        realistic_range=[.938, .945], realistic_range_status='Prior heuristic aspiration, not a measured or guaranteed attainable range.',
        independent_metric_checks='All operating points cross-checked by pandas grouping and per-anchor scalar F0.5; TP/FP/FN exact.')
    (folder / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('FINAL_REPORT', folder, 'WINNER', result['winner'], 'REBUILT_MINUS_CONTROL', repr(gain), flush=True)


if __name__ == '__main__':
    main()
