"""Train CatBoost with full 81-feature set (Baseline + Veto + Enhanced Missing-Address/Tempered Vetoes)."""
import os, time, json, sqlite3
from pathlib import Path
import joblib, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from src.enhanced_features import compute_enhanced_features
from src.model import CalibratedMatcher
from src.real_metrics import tune, entity_scores
from src.global_arbitration import arbitrate

def get_or_create_enhanced(split, pairs, con, cache_dir, p_prelim=None):
    cache_path = cache_dir / f'{split}_enhanced.joblib'
    if cache_path.exists():
        print(f'Loading cached enhanced features for {split} from {cache_path}')
        return joblib.load(cache_path)
    
    print(f'Computing enhanced features for {split} ({len(pairs)} pairs)...')
    t0 = time.time()
    chunks = []
    chunk_size = 25000
    for i in range(0, len(pairs), chunk_size):
        sub = pairs.iloc[i:i+chunk_size]
        sub_p = p_prelim[i:i+chunk_size] if p_prelim is not None else None
        chunks.append(compute_enhanced_features(sub, con, sub_p))
        if i % 100000 == 0 and i > 0:
            print(f'  {split}: {i}/{len(pairs)} in {time.time()-t0:.1f}s')
            
    df = pd.concat(chunks, ignore_index=True)
    joblib.dump(df, cache_path)
    print(f'Saved {split} enhanced features ({df.shape}) in {time.time()-t0:.1f}s')
    return df

def main():
    enhanced_cache = Path('work/improvements/enhanced_cache')
    enhanced_cache.mkdir(parents=True, exist_ok=True)
    veto_cache = Path('work/improvements/veto_cache')
    
    con = sqlite3.connect('work/real_v1/train/records.sqlite')
    
    print('Loading dataset splits...')
    fit_data = joblib.load('work/real_v1/features/fit.joblib')
    cal_data = joblib.load('work/real_v1/features/calibration.joblib')
    val_data = joblib.load('work/real_v1/features/validation.joblib')
    
    fit_veto = joblib.load(veto_cache / 'fit_veto.joblib')
    cal_veto = joblib.load(veto_cache / 'calibration_veto.joblib')
    val_veto = joblib.load(veto_cache / 'validation_veto.joblib')
    
    # Compute or load enhanced features
    fit_enh = get_or_create_enhanced('fit', fit_data['pairs'], con, enhanced_cache)
    cal_enh = get_or_create_enhanced('calibration', cal_data['pairs'], con, enhanced_cache)
    val_enh = get_or_create_enhanced('validation', val_data['pairs'], con, enhanced_cache)
    con.close()
    
    # Concatenate all 81 features
    X_fit = pd.concat([fit_data['features'].reset_index(drop=True), fit_veto.reset_index(drop=True), fit_enh.reset_index(drop=True)], axis=1)
    X_cal = pd.concat([cal_data['features'].reset_index(drop=True), cal_veto.reset_index(drop=True), cal_enh.reset_index(drop=True)], axis=1)
    X_val = pd.concat([val_data['features'].reset_index(drop=True), val_veto.reset_index(drop=True), val_enh.reset_index(drop=True)], axis=1)
    
    y_fit = fit_data['pairs'].label.to_numpy()
    y_cal = cal_data['pairs'].label.to_numpy()
    y_val = val_data['pairs'].label.to_numpy()
    
    print(f'Combined Feature Shapes: Fit={X_fit.shape}, Cal={X_cal.shape}, Val={X_val.shape}')
    
    # Train CatBoost
    print('Training CatBoost on 81 features (depth=8, iterations=1200)...')
    t0 = time.time()
    cb = CatBoostClassifier(iterations=1200, depth=8, learning_rate=0.06, l2_leaf_reg=5,
                            loss_function='Logloss', eval_metric='Logloss', random_seed=42,
                            thread_count=4, allow_writing_files=False, verbose=100)
    cb.fit(X_fit, y_fit, eval_set=(X_cal, y_cal), early_stopping_rounds=100)
    print(f'Training completed in {time.time()-t0:.1f}s, best iteration: {cb.get_best_iteration()}')
    
    # Calibrate
    print('Calibrating probabilities on calibration set...')
    matcher = CalibratedMatcher(cb)
    matcher.calibrate(X_cal, y_cal)
    
    # Predict on validation
    print('Scoring validation set...')
    p_val_new = matcher.predict(X_val)
    np.save('work/improvements/enhanced_cache/p_val_enhanced.npy', p_val_new)
    joblib.dump(matcher, 'artifacts/improvements/enhanced_model_v3.joblib')
    
    # Evaluate raw performance (un-arbitrated)
    truth = val_data['truth_counts']
    pos = {v: i for i, v in enumerate(truth)}
    indices = np.array([pos[x] for x in val_data['pairs'].source1_entity_id], dtype=np.int32)
    counts = np.array(list(truth.values()))
    
    thresh_raw, _ = tune(indices, y_val, p_val_new, counts, 10320219)
    scores_raw = entity_scores(indices, y_val, p_val_new, counts, thresh_raw, 10320219)
    
    # Feature importances
    imp = pd.Series(cb.get_feature_importance(), index=X_fit.columns).sort_values(ascending=False)
    print('\n--- Top 20 Feature Importances (Enhanced Model) ---')
    for feat, score in imp.head(20).items():
        print(f'  {feat:35s}: {score:.4f}')
        
    # Check blank address FN recovery specifically
    con = sqlite3.connect('work/real_v1/train/records.sqlite')
    con.row_factory = sqlite3.Row
    p_old = np.load('work/improvements/veto_cache/p_val.npy')
    
    fn_old_mask = (y_val == 1) & (p_old < 0.6750)
    fn_old_pairs = val_data['pairs'][fn_old_mask].copy()
    fn_old_pairs['p_new'] = p_val_new[fn_old_mask]
    
    blank_recovered = 0
    blank_total = 0
    for row in fn_old_pairs.itertuples(index=False):
        t = con.execute('SELECT address_norm FROM targets WHERE entity_id=?', (row.candidate_entity_id,)).fetchone()
        if not t['address_norm'].strip():
            blank_total += 1
            if row.p_new >= thresh_raw:
                blank_recovered += 1
    con.close()
    
    print(f'\n--- Blank-Address FN Recovery ---')
    print(f'Blank-address true matches previously rejected: {blank_total}')
    print(f'Now clearing new threshold ({thresh_raw:.4f}): {blank_recovered} ({blank_recovered/blank_total*100:.1f}%)')
    
    # Stack with Target-Only Arbitration and Margin Gating (delta=0.20)
    won_tgt = arbitrate(val_data['pairs'], p_val_new, max_per_source=None)
    p_df = val_data['pairs'][['source1_entity_id']].copy()
    p_df['prob'] = p_val_new
    p_df['source'] = val_data['pairs'].candidate_entity_id.str[:2]
    max_p = p_df.groupby(['source1_entity_id', 'source'])['prob'].transform('max')
    margin_to_best = max_p - p_df['prob']
    
    # Evaluate Stacked State
    gated_mask = won_tgt & (margin_to_best <= 0.20)
    p_stacked = np.where(gated_mask, p_val_new, 0.0)
    thresh_stacked, _ = tune(indices, y_val, p_stacked, counts, 10320219)
    scores_stacked = entity_scores(indices, y_val, p_stacked, counts, thresh_stacked, 10320219)
    
    print('\n================ RETRAINED ENHANCED MODEL VALIDATION ================')
    print(f'Raw Enhanced Model (81 features, Thresh={thresh_raw:.4f}):')
    print(f'  Precision:  {scores_raw["precision"]:.5f}')
    print(f'  Recall:     {scores_raw["recall"]:.5f}')
    print(f'  Macro F0.5: {scores_raw["f0.5"]:.5f}')
    print(f'  TP: {scores_raw["true_positive"]}, FP: {scores_raw["false_positive"]}, FN: {scores_raw["false_negative"]}')
    
    print(f'\nStacked Enhanced Model (+ Target Arbitration + Margin Delta<=0.20, Thresh={thresh_stacked:.4f}):')
    print(f'  Precision:  {scores_stacked["precision"]:.5f}')
    print(f'  Recall:     {scores_stacked["recall"]:.5f}')
    print(f'  Macro F0.5: {scores_stacked["f0.5"]:.5f}')
    print(f'  TP: {scores_stacked["true_positive"]}, FP: {scores_stacked["false_positive"]}, FN: {scores_stacked["false_negative"]}')
    print('====================================================================\n')
    
    report = {
        'model_name': 'CatBoost_D8_Enhanced_81features',
        'raw_enhanced': {'threshold': thresh_raw, **scores_raw},
        'stacked_enhanced': {'threshold': thresh_stacked, **scores_stacked},
        'blank_address_fn_recovery': {
            'total_blank_address_fns': blank_total,
            'cleared_new_threshold': blank_recovered,
            'recovery_pct': round(blank_recovered / blank_total * 100, 2)
        },
        'top_features': imp.head(25).to_dict()
    }
    
    out_path = Path('reports/improvements/enhanced_model_validation_v3.json')
    out_path.write_text(json.dumps(report, indent=2))
    print(f'Saved comprehensive validation results to: {out_path}')

if __name__ == '__main__':
    main()
