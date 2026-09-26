"""Train and evaluate CatBoost on the cap64 retrained candidate distribution with full 81 features."""
import json, time, sqlite3
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
    base = Path('work/cap64_v1')
    feat_dir = base / 'features'
    supp_dir = base / 'supplemental_cache'
    
    print("Loading cap64 feature datasets...")
    val_data = joblib.load(feat_dir / 'validation.joblib')
    cal_data = joblib.load(feat_dir / 'calibration.joblib')
    fit_data = joblib.load(feat_dir / 'fit.joblib')
    
    val_veto = joblib.load(supp_dir / 'validation_veto.joblib')
    val_enh = joblib.load(supp_dir / 'validation_enh.joblib')
    
    cal_veto = joblib.load(supp_dir / 'calibration_veto.joblib')
    cal_enh = joblib.load(supp_dir / 'calibration_enh.joblib')
    
    fit_veto = joblib.load(supp_dir / 'fit_veto.joblib')
    fit_enh = joblib.load(supp_dir / 'fit_enh.joblib')
    
    X_val = pd.concat([val_data['features'].reset_index(drop=True), val_veto.reset_index(drop=True), val_enh.reset_index(drop=True)], axis=1)
    X_cal = pd.concat([cal_data['features'].reset_index(drop=True), cal_veto.reset_index(drop=True), cal_enh.reset_index(drop=True)], axis=1)
    X_fit = pd.concat([fit_data['features'].reset_index(drop=True), fit_veto.reset_index(drop=True), fit_enh.reset_index(drop=True)], axis=1)
    
    y_val = val_data['pairs'].label.to_numpy()
    y_cal = cal_data['pairs'].label.to_numpy()
    y_fit = fit_data['pairs'].label.to_numpy()
    
    print(f"X_fit shape: {X_fit.shape}, Positives: {y_fit.sum()}/{len(y_fit)} ({y_fit.mean():.4f})")
    print(f"X_cal shape: {X_cal.shape}, Positives: {y_cal.sum()}/{len(y_cal)} ({y_cal.mean():.4f})")
    print(f"X_val shape: {X_val.shape}, Positives: {y_val.sum()}/{len(y_val)} ({y_val.mean():.4f})")
    
    print("\nTraining CatBoost on 81 features (depth=8, iterations=1200)...")
    t0 = time.time()
    cb = CatBoostClassifier(iterations=1200, depth=8, learning_rate=0.06, l2_leaf_reg=5,
                            loss_function='Logloss', eval_metric='Logloss', random_seed=42,
                            thread_count=4, allow_writing_files=False, verbose=100)
    cb.fit(X_fit, y_fit, eval_set=(X_cal, y_cal), early_stopping_rounds=100)
    print(f"CatBoost training completed in {time.time()-t0:.1f}s, best iteration: {cb.get_best_iteration()}")
    
    matcher = CalibratedMatcher(cb)
    matcher.calibrate(X_cal, y_cal)
    
    model_save_path = Path('artifacts/improvements/cap64_retrained_model_v4.joblib')
    joblib.dump(matcher, model_save_path)
    print(f"Saved retrained model to: {model_save_path}")
    
    # Validation scoring
    print("\nScoring cap64 validation set...")
    p_val = matcher.predict(X_val)
    np.save(supp_dir / 'p_val_cap64.npy', p_val)
    
    pairs = val_data['pairs']
    truth = val_data['truth_counts']
    pos = {v: i for i, v in enumerate(truth)}
    indices = np.array([pos[x] for x in pairs.source1_entity_id], dtype=np.int32)
    counts = np.array(list(truth.values()))
    
    # State 1: Raw retrained model
    th_raw, _ = tune(indices, y_val, p_val, counts, 10320219)
    sc_raw = entity_scores(indices, y_val, p_val, counts, th_raw, 10320219)
    print(f"\n1. Raw Retrained Model (Cap 64): Thresh={th_raw:.4f}, Prec={sc_raw['precision']:.4f}, Rec={sc_raw['recall']:.4f}, F0.5={sc_raw['f0.5']:.5f}, TP={sc_raw['true_positive']}, FP={sc_raw['false_positive']}, FN={sc_raw['false_negative']}")
    
    # State 2: Target-Only Arbitration
    won_tgt = arbitrate(pairs, p_val, max_per_source=None)
    p_tgt = np.where(won_tgt, p_val, 0.0)
    th_tgt, _ = tune(indices, y_val, p_tgt, counts, 10320219)
    sc_tgt = entity_scores(indices, y_val, p_tgt, counts, th_tgt, 10320219)
    print(f"2. Retrained + Target Arbitration: Thresh={th_tgt:.4f}, Prec={sc_tgt['precision']:.4f}, Rec={sc_tgt['recall']:.4f}, F0.5={sc_tgt['f0.5']:.5f}, TP={sc_tgt['true_positive']}, FP={sc_tgt['false_positive']}, FN={sc_tgt['false_negative']}")
    
    # State 3: Margin Gating Sweep stacked on Target Arbitration
    print("\n3. Margin Gating Sweeps stacked on Target Arbitration:")
    deltas = [0.05, 0.10, 0.15, 0.20, 0.30]
    margin_results = {}
    for d in deltas:
        p_gated = apply_margin_gating(pairs, p_tgt, d)
        th_g, _ = tune(indices, y_val, p_gated, counts, 10320219)
        sc_g = entity_scores(indices, y_val, p_gated, counts, th_g, 10320219)
        margin_results[d] = {'threshold': th_g, **sc_g}
        print(f"  Delta={d:<4} | Thresh={th_g:.4f} | Prec={sc_g['precision']:.4f} | Rec={sc_g['recall']:.4f} | F0.5={sc_g['f0.5']:.5f} | TP={sc_g['true_positive']:<5} | FP={sc_g['false_positive']:<4} | FN={sc_g['false_negative']:<4}")
        
    # Check Blank-Address FN recovery
    con = sqlite3.connect('work/cap64_v1/train/records.sqlite')
    con.row_factory = sqlite3.Row
    fn_mask = (y_val == 1) & (p_val < th_raw)
    fn_pairs = pairs[fn_mask]
    blank_fn = 0
    blank_recovered = 0
    for r in fn_pairs.itertuples(index=False):
        t = con.execute('SELECT address_norm FROM targets WHERE entity_id=?', (r.candidate_entity_id,)).fetchone()
        if not t['address_norm'].strip():
            blank_fn += 1
            
    # Check how many of the originally 1001 blank FNs now clear threshold
    print(f"\nBlank-address FNs remaining under retrained model: {blank_fn}")
    con.close()
    
    report = {
        'model': 'CatBoost_Cap64_Retrained_81feat',
        'raw_retrained': {'threshold': th_raw, **sc_raw},
        'arbitrated_retrained': {'threshold': th_tgt, **sc_tgt},
        'margin_gated_sweeps': margin_results,
        'blank_address_fns': blank_fn
    }
    out_path = Path('reports/improvements/phase6_retrain_validation_v4.json')
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved retrain evaluation report to: {out_path}")

if __name__ == '__main__':
    main()
