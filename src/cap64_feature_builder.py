"""Extract 14 veto features and 9 enhanced features for cap64 splits."""
import os, time, sqlite3
from pathlib import Path
import joblib, numpy as np, pandas as pd
from src.veto_features import compute_veto_features
from src.enhanced_features import compute_enhanced_features

def build_supplemental_features_for_split(split_name, base_dir='work/cap64_v1'):
    base = Path(base_dir)
    feat_dir = base / 'features'
    cache_dir = base / 'supplemental_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    split_file = feat_dir / f'{split_name}.joblib'
    if not split_file.exists():
        print(f"File not found: {split_file}")
        return
        
    data = joblib.load(split_file)
    pairs = data['pairs']
    n_pairs = len(pairs)
    print(f"\nProcessing {split_name} ({n_pairs} pairs)...")
    
    veto_cache_file = cache_dir / f'{split_name}_veto.joblib'
    enh_cache_file = cache_dir / f'{split_name}_enh.joblib'
    
    con = sqlite3.connect(f'{base_dir}/train/records.sqlite')
    con.row_factory = sqlite3.Row
    
    # 1. Veto features
    if veto_cache_file.exists():
        print(f"Loading cached veto features from {veto_cache_file}")
        veto_df = joblib.load(veto_cache_file)
    else:
        print(f"Computing 14 veto features for {split_name}...")
        t0 = time.time()
        chunks = []
        chunk_size = 25000
        for i in range(0, n_pairs, chunk_size):
            sub = pairs.iloc[i:i+chunk_size]
            chunks.append(compute_veto_features(sub, con))
            if i % 100000 == 0 and i > 0:
                print(f"  veto {split_name}: {i}/{n_pairs} in {time.time()-t0:.1f}s")
        veto_df = pd.concat(chunks, ignore_index=True)
        joblib.dump(veto_df, veto_cache_file)
        print(f"Saved {split_name} veto features ({veto_df.shape}) in {time.time()-t0:.1f}s")
        
    # 2. Enhanced features
    if enh_cache_file.exists():
        print(f"Loading cached enhanced features from {enh_cache_file}")
        enh_df = joblib.load(enh_cache_file)
    else:
        print(f"Computing 9 enhanced features for {split_name}...")
        t0 = time.time()
        chunks = []
        chunk_size = 25000
        for i in range(0, n_pairs, chunk_size):
            sub = pairs.iloc[i:i+chunk_size]
            chunks.append(compute_enhanced_features(sub, con))
            if i % 100000 == 0 and i > 0:
                print(f"  enh {split_name}: {i}/{n_pairs} in {time.time()-t0:.1f}s")
        enh_df = pd.concat(chunks, ignore_index=True)
        joblib.dump(enh_df, enh_cache_file)
        print(f"Saved {split_name} enhanced features ({enh_df.shape}) in {time.time()-t0:.1f}s")
        
    con.close()
    return veto_df, enh_df

if __name__ == '__main__':
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'calibration'
    base_dir = sys.argv[2] if len(sys.argv) > 2 else 'work/cap64_v1'
    build_supplemental_features_for_split(split, base_dir)
