"""Phase 8.1: Full fit-split retrain on rebuilt C++ index, then evaluate.
Phase 8.2: Honest apples-to-apples comparison.
"""
import json, time, sqlite3, os
from pathlib import Path
import joblib, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from src.model import CalibratedMatcher
from src.real_metrics import tune, entity_scores
from src.global_arbitration import arbitrate

def apply_margin_gating(pairs_df, probs, delta):
    df = pd.DataFrame({
        'source1_entity_id': pairs_df['source1_entity_id'].to_numpy(),
        'candidate_source': pairs_df['candidate_entity_id'].str[:2].to_numpy(),
        'prob': probs
    })
    max_p = df.groupby(['source1_entity_id', 'candidate_source'])['prob'].transform('max')
    gated_probs = np.where((max_p - df['prob']) <= delta, probs, 0.0)
    return gated_probs

def main():
    base = Path('work/cap64_rebuilt')
    feat_dir = base / 'features'
    supp_dir = base / 'supplemental_cache'

    # ── Phase 8.0 verification ──────────────────────────────────────────
    fit_path = feat_dir / 'fit.joblib'
    if not fit_path.exists():
        print(f"FATAL: {fit_path} does not exist. Run fit extraction first.")
        return

    fit_mtime = os.path.getmtime(fit_path)
    fit_mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fit_mtime))

    print(f"Loading datasets from {base}...")
    fit_data = joblib.load(fit_path)
    val_data = joblib.load(feat_dir / 'validation.joblib')
    cal_data = joblib.load(feat_dir / 'calibration.joblib')

    print(f"Loading supplemental features...")
    fit_veto = joblib.load(supp_dir / 'fit_veto.joblib')
    fit_enh = joblib.load(supp_dir / 'fit_enh.joblib')
    val_veto = joblib.load(supp_dir / 'validation_veto.joblib')
    val_enh = joblib.load(supp_dir / 'validation_enh.joblib')
    cal_veto = joblib.load(supp_dir / 'calibration_veto.joblib')
    cal_enh = joblib.load(supp_dir / 'calibration_enh.joblib')

    # Load prior model to get the canonical 81-feature order
    matcher_v3 = joblib.load('artifacts/improvements/enhanced_model_v3.joblib')
    expected_features = matcher_v3.estimator.feature_names_

    # Assemble feature matrices
    X_fit = pd.concat([fit_data['features'].reset_index(drop=True),
                       fit_veto.reset_index(drop=True),
                       fit_enh.reset_index(drop=True)], axis=1)[expected_features]
    X_val = pd.concat([val_data['features'].reset_index(drop=True),
                       val_veto.reset_index(drop=True),
                       val_enh.reset_index(drop=True)], axis=1)[expected_features]
    X_cal = pd.concat([cal_data['features'].reset_index(drop=True),
                       cal_veto.reset_index(drop=True),
                       cal_enh.reset_index(drop=True)], axis=1)[expected_features]

    y_fit = fit_data['pairs'].label.to_numpy()
    y_val = val_data['pairs'].label.to_numpy()
    y_cal = cal_data['pairs'].label.to_numpy()

    # ── Explicit provenance ─────────────────────────────────────────────
    fit_pairs = len(y_fit)
    fit_pos = int(y_fit.sum())
    fit_anchors = fit_data['stats']['counts']['anchors']
    fit_recall_num = fit_data['stats']['counts']['retained_true']
    fit_recall_den = fit_data['stats']['counts']['true_pairs']
    fit_recall = fit_recall_num / fit_recall_den

    provenance = (
        f"TRAINING DATA PROVENANCE: split=fit, anchors={fit_anchors}, "
        f"pairs={fit_pairs}, positives={fit_pos} ({fit_pos/fit_pairs*100:.2f}%), "
        f"candidate_recall={fit_recall_num}/{fit_recall_den}={fit_recall:.4f}, "
        f"fit.joblib mtime={fit_mtime_str}, "
        f"source_index=work/cap64_rebuilt/train/supplement_index.bin (rebuilt C++ with Indic Soundex + compound address keys)"
    )

    print(f"\n{'='*80}")
    print(provenance)
    print(f"{'='*80}\n")

    print(f"X_fit shape: {X_fit.shape}, Positives: {fit_pos}/{fit_pairs} ({fit_pos/fit_pairs*100:.4f}%)")
    print(f"X_cal shape: {X_cal.shape}, Positives: {y_cal.sum()}/{len(y_cal)} ({y_cal.mean()*100:.4f}%)")
    print(f"X_val shape: {X_val.shape}, Positives: {y_val.sum()}/{len(y_val)} ({y_val.mean()*100:.4f}%)")

    # ── Phase 8.1: Full fit-split retrain ──────────────────────────────
    print("\nTraining CatBoost on FULL FIT SPLIT (depth=8, iterations=1200, lr=0.06)...")
    t0 = time.time()
    cb = CatBoostClassifier(
        iterations=1200, depth=8, learning_rate=0.06, l2_leaf_reg=5,
        loss_function='Logloss', eval_metric='Logloss', random_seed=42,
        thread_count=6, allow_writing_files=False, verbose=100
    )
    cb.fit(X_fit, y_fit, eval_set=(X_cal, y_cal), early_stopping_rounds=100)
    train_time = time.time() - t0
    best_iter = cb.get_best_iteration()
    print(f"CatBoost training completed in {train_time:.1f}s, best iteration: {best_iter}")

    # Calibrate on rebuilt calibration split
    matcher = CalibratedMatcher(cb)
    matcher.calibrate(X_cal, y_cal)

    # Save model
    model_path = Path('artifacts/improvements/phase8_full_fit_retrain_model.joblib')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(matcher, model_path)
    print(f"Saved retrained model to: {model_path}")

    # Score validation
    print("\nScoring rebuilt validation split...")
    p_val = matcher.predict(X_val)

    pairs = val_data['pairs']
    truth = val_data['truth_counts']
    pos = {v: i for i, v in enumerate(truth)}
    indices = np.array([pos[x] for x in pairs.source1_entity_id], dtype=np.int32)
    counts = np.array(list(truth.values()), dtype=np.int32)
    target_count = 10320219

    results = {}

    # State 1: Raw full-fit retrained
    th_raw, _ = tune(indices, y_val, p_val, counts, target_count)
    sc_raw = entity_scores(indices, y_val, p_val, counts, th_raw, target_count)
    label_raw = f"Full Fit-Split Retrain ({fit_anchors} anchors, {fit_pairs} pairs) — Raw"
    print(f"\n1. {label_raw}:")
    print(f"   Thresh={th_raw:.4f} | Prec={sc_raw['precision']:.5f} | Rec={sc_raw['recall']:.5f} | F0.5={sc_raw['f0.5']:.5f} | TP={sc_raw['true_positive']} | FP={sc_raw['false_positive']} | FN={sc_raw['false_negative']}")
    results['state_1_raw'] = {'label': label_raw, 'threshold': th_raw, **sc_raw}

    # State 2: Target Arbitration
    won_tgt = arbitrate(pairs, p_val, max_per_source=None)
    p_tgt = np.where(won_tgt, p_val, 0.0)
    th_tgt, _ = tune(indices, y_val, p_tgt, counts, target_count)
    sc_tgt = entity_scores(indices, y_val, p_tgt, counts, th_tgt, target_count)
    label_tgt = f"Full Fit-Split Retrain ({fit_anchors} anchors, {fit_pairs} pairs) — + Target Arbitration"
    print(f"\n2. {label_tgt}:")
    print(f"   Thresh={th_tgt:.4f} | Prec={sc_tgt['precision']:.5f} | Rec={sc_tgt['recall']:.5f} | F0.5={sc_tgt['f0.5']:.5f} | TP={sc_tgt['true_positive']} | FP={sc_tgt['false_positive']} | FN={sc_tgt['false_negative']}")
    results['state_2_arbitrated'] = {'label': label_tgt, 'threshold': th_tgt, **sc_tgt}

    # State 3: Margin Gating Sweep
    print(f"\n3. Full Fit-Split Retrain ({fit_anchors} anchors, {fit_pairs} pairs) — Margin Gating Sweep on Arbitration:")
    margin_results = {}
    for d in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        p_gated = apply_margin_gating(pairs, p_tgt, d)
        th_g, _ = tune(indices, y_val, p_gated, counts, target_count)
        sc_g = entity_scores(indices, y_val, p_gated, counts, th_g, target_count)
        margin_results[str(d)] = {'threshold': th_g, **sc_g}
        print(f"   Delta={d:<4} | Thresh={th_g:.4f} | Prec={sc_g['precision']:.5f} | Rec={sc_g['recall']:.5f} | F0.5={sc_g['f0.5']:.5f} | TP={sc_g['true_positive']} | FP={sc_g['false_positive']} | FN={sc_g['false_negative']}")
    results['state_3_margin_sweep'] = margin_results

    # ── Phase 8.2: Comparison ──────────────────────────────────────────
    benchmark_f05 = 0.93723
    phase7_partial_best = 0.92874  # Phase 7 best with margin gating (cal-only retrain)

    # Find best config
    best_f05 = max(sc_raw['f0.5'], sc_tgt['f0.5'], max(v['f0.5'] for v in margin_results.values()))
    best_config = "raw"
    if sc_tgt['f0.5'] == best_f05:
        best_config = "target_arbitration"
    for d, v in margin_results.items():
        if v['f0.5'] == best_f05:
            best_config = f"margin_delta_{d}"

    print(f"\n{'='*80}")
    print("PHASE 8.2 — HONEST APPLES-TO-APPLES COMPARISON")
    print(f"{'='*80}")
    print(f"Benchmark (Phase 4/5, un-retrained model, Cap 32 blocking): F0.5 = {benchmark_f05:.5f}")
    print(f"Phase 7 partial retrain (Cal-only 5k anchors, Cap 64 rebuilt): F0.5 = {phase7_partial_best:.5f}")
    print(f"Phase 8 FULL fit-split retrain (30k anchors, Cap 64 rebuilt): F0.5 = {best_f05:.5f} ({best_config})")
    print(f"")

    if best_f05 > benchmark_f05:
        verdict = "OUTCOME A: Full retrain RECOVERS past benchmark. The earlier regression was a training-data-starvation artifact, now resolved."
        delta = best_f05 - benchmark_f05
        print(f"VERDICT: {verdict}")
        print(f"Improvement over benchmark: +{delta:.5f}")
    elif best_f05 > phase7_partial_best and best_f05 <= benchmark_f05:
        verdict = "OUTCOME B: Full retrain improves over Phase 7 partial numbers but still doesn't clear benchmark. The wider candidate pool has a real structural precision cost under F0.5."
        delta_vs_p7 = best_f05 - phase7_partial_best
        gap_to_bench = benchmark_f05 - best_f05
        print(f"VERDICT: {verdict}")
        print(f"Improvement over Phase 7 partial: +{delta_vs_p7:.5f}")
        print(f"Gap to benchmark: -{gap_to_bench:.5f}")
    else:
        verdict = "OUTCOME C: Full retrain doesn't meaningfully change anything. The cal-split stand-in wasn't the bottleneck."
        print(f"VERDICT: {verdict}")

    results['comparison'] = {
        'benchmark_f05': benchmark_f05,
        'phase7_partial_best_f05': phase7_partial_best,
        'phase8_best_f05': best_f05,
        'phase8_best_config': best_config,
        'verdict': verdict
    }

    # ── Phase 8.4: Honest ceiling ──────────────────────────────────────
    val_recall = val_data['stats']['candidate_recall']
    val_retained = val_data['stats']['counts']['retained_true']
    val_total = val_data['stats']['counts']['true_pairs']
    val_singletons = val_data['stats']['counts']['true_singletons']

    # Max F0.5 = at perfect precision (FP=0) with current recall ceiling
    # For entities: recall = sum of per-entity recall / N anchors
    # With zero FP, precision = 1.0, so F0.5 = (1.25 * 1.0 * recall) / (0.25 * 1.0 + recall)
    max_possible_recall = val_retained / val_total  # This is pair recall, not macro recall
    # Macro recall ceiling needs to account for entity-level aggregation
    # A simpler upper-bound: if every retained true pair is correctly predicted
    # then macro recall = macro_recall_ceiling, and with FP=0, precision=1.0
    max_f05_at_zero_fp = (1.25 * 1.0 * max_possible_recall) / (0.25 * 1.0 + max_possible_recall)

    print(f"\n{'='*80}")
    print("PHASE 8.4 — HONEST CEILING (REAL PRODUCTION RECALL)")
    print(f"{'='*80}")
    print(f"Validation candidate recall (production): {val_retained}/{val_total} = {val_recall:.4f} ({val_recall*100:.2f}%)")
    print(f"Candidate recall ceiling places upper bound on macro recall.")
    print(f"At perfect precision (FP=0), maximum pair-level F0.5 = {max_f05_at_zero_fp:.5f}")
    print(f"This is a mathematical UPPER BOUND, not an achievable operating point.")
    print(f"Gate threshold: 0.99")
    gate_reachable = max_f05_at_zero_fp >= 0.99
    print(f"Is 0.99 reachable at this recall ceiling? {'YES' if gate_reachable else 'NO'}")
    if not gate_reachable:
        needed_recall = (0.99 * 0.25) / (1.25 - 0.99)
        print(f"To reach F0.5=0.99 even at perfect precision, need pair recall >= {needed_recall:.4f} ({needed_recall*100:.2f}%)")

    results['ceiling'] = {
        'production_candidate_recall': val_recall,
        'retained_true': val_retained,
        'total_true': val_total,
        'max_f05_at_zero_fp': max_f05_at_zero_fp,
        'gate': 0.99,
        'gate_reachable': gate_reachable
    }

    results['provenance'] = provenance
    results['training_time_seconds'] = train_time
    results['best_iteration'] = best_iter

    out_path = Path('reports/improvements/phase8_full_retrain_evaluation.json')
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved Phase 8 report to: {out_path}")

if __name__ == '__main__':
    main()
