"""Generate precision-focused veto features, train CatBoost, and optimize Macro F0.5."""
import os, time, json, sqlite3
from pathlib import Path
import joblib, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from src.veto_features import compute_veto_features
from src.model import CalibratedMatcher
from src.real_metrics import tune, entity_scores

def get_or_create_veto(split, pairs, con, cache_dir):
    cache_path = cache_dir / f'{split}_veto.joblib'
    if cache_path.exists():
        print(f'Loading cached veto features for {split} from {cache_path}')
        return joblib.load(cache_path)
    
    print(f'Computing veto features for {split} ({len(pairs)} pairs)...')
    t0 = time.time()
    chunks = []
    chunk_size = 20000
    for i in range(0, len(pairs), chunk_size):
        sub = pairs.iloc[i:i+chunk_size]
        chunks.append(compute_veto_features(sub, con))
        if i % 100000 == 0 and i > 0:
            print(f'  {split}: {i}/{len(pairs)} in {time.time()-t0:.1f}s')
            
    df = pd.concat(chunks, ignore_index=True)
    joblib.dump(df, cache_path)
    print(f'Saved {split} veto features ({df.shape}) in {time.time()-t0:.1f}s')
    return df

def main():
    cache_dir = Path('work/improvements/veto_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    con = sqlite3.connect('work/real_v1/train/records.sqlite')
    
    # Load raw feature parts
    print('Loading dataset splits...')
    fit_data = joblib.load('work/real_v1/features/fit.joblib')
    cal_data = joblib.load('work/real_v1/features/calibration.joblib')
    val_data = joblib.load('work/real_v1/features/validation.joblib')
    
    # Compute or load veto features
    fit_veto = get_or_create_veto('fit', fit_data['pairs'], con, cache_dir)
    cal_veto = get_or_create_veto('calibration', cal_data['pairs'], con, cache_dir)
    val_veto = get_or_create_veto('validation', val_data['pairs'], con, cache_dir)
    con.close()
    
    # Concatenate features
    X_fit = pd.concat([fit_data['features'].reset_index(drop=True), fit_veto.reset_index(drop=True)], axis=1)
    X_cal = pd.concat([cal_data['features'].reset_index(drop=True), cal_veto.reset_index(drop=True)], axis=1)
    X_val = pd.concat([val_data['features'].reset_index(drop=True), val_veto.reset_index(drop=True)], axis=1)
    
    y_fit = fit_data['pairs'].label.to_numpy()
    y_cal = cal_data['pairs'].label.to_numpy()
    y_val = val_data['pairs'].label.to_numpy()
    
    print(f'Feature shapes: Fit={X_fit.shape}, Cal={X_cal.shape}, Val={X_val.shape}')
    
    # Train CatBoost
    print('Training CatBoost with precision veto features...')
    t0 = time.time()
    cb = CatBoostClassifier(iterations=1200, depth=8, learning_rate=0.06, l2_leaf_reg=5,
                            loss_function='Logloss', eval_metric='Logloss', random_seed=42,
                            thread_count=4, allow_writing_files=False, verbose=100)
    cb.fit(X_fit, y_fit, eval_set=(X_cal, y_cal), early_stopping_rounds=100)
    print(f'CatBoost fit completed in {time.time()-t0:.1f}s, best iteration: {cb.get_best_iteration()}')
    
    # Feature importances
    imp = pd.Series(cb.get_feature_importance(), index=X_fit.columns).sort_values(ascending=False)
    print('\n--- Top 20 Feature Importances ---')
    for feat, score in imp.head(20).items():
        print(f'  {feat:35s}: {score:.4f}')
        
    # Calibrate
    print('\nCalibrating probabilities...')
    matcher = CalibratedMatcher(cb)
    matcher.calibrate(X_cal, y_cal)
    
    # Predict on validation
    print('Scoring validation set...')
    p_val = matcher.predict(X_val)
    
    # Tune threshold on validation for Macro F0.5
    truth = val_data['truth_counts']
    pos = {v: i for i, v in enumerate(truth)}
    indices = np.array([pos[x] for x in val_data['pairs'].source1_entity_id], dtype=np.int32)
    counts = np.array(list(truth.values()))
    
    thresh, table = tune(indices, y_val, p_val, counts, 10320219)
    scores = entity_scores(indices, y_val, p_val, counts, thresh, 10320219)
    
    print('\n================ VALIDATION RESULTS ================')
    print(f'  Optimal Threshold: {thresh:.4f}')
    print(f'  Precision:         {scores["precision"]:.6f}')
    print(f'  Recall:            {scores["recall"]:.6f}')
    print(f'  Macro F0.5:        {scores["f0.5"]:.6f}')
    print(f'  Macro F1:          {scores["f1"]:.6f}')
    print(f'  True Positives:    {scores["true_positive"]}')
    print(f'  False Positives:   {scores["false_positive"]}')
    print(f'  False Negatives:   {scores["false_negative"]}')
    print('====================================================\n')
    
    # Save results
    result_record = {
        'model': 'CatBoost_D8_Veto',
        'threshold': thresh,
        'metrics': scores,
        'top_features': imp.head(25).to_dict()
    }
    with open('reports/improvements/veto_model_validation.json', 'w') as f:
        json.dump(result_record, f, indent=2)
    print('Results saved to reports/improvements/veto_model_validation.json')

if __name__ == '__main__':
    main()
