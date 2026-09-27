"""Extract 14 veto features and 9 enhanced features for a split using multiprocessing."""
import os, time, sqlite3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import joblib, numpy as np, pandas as pd
from src.veto_features import compute_veto_features
from src.enhanced_features import compute_enhanced_features

def _veto_worker(args):
    chunk_df, db_path = args
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    res = compute_veto_features(chunk_df, con)
    con.close()
    return res

def _enh_worker(args):
    chunk_df, db_path = args
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    res = compute_enhanced_features(chunk_df, con)
    con.close()
    return res

def build_supplemental_parallel(split_name='fit', base_dir='work/cap64_rebuilt', n_workers=6, chunk_size=20000):
    base = Path(base_dir)
    feat_dir = base / 'features'
    cache_dir = base / 'supplemental_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_file = feat_dir / f'{split_name}.joblib'
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    print(f"Loading {split_name} pairs from {split_file}...")
    t0 = time.time()
    data = joblib.load(split_file)
    pairs = data['pairs']
    n_pairs = len(pairs)
    print(f"Loaded {n_pairs} pairs in {time.time()-t0:.1f}s")

    db_path = str(base / 'train/records.sqlite')
    veto_cache_file = cache_dir / f'{split_name}_veto.joblib'
    enh_cache_file = cache_dir / f'{split_name}_enh.joblib'

    chunks = [pairs.iloc[i:i+chunk_size] for i in range(0, n_pairs, chunk_size)]
    print(f"Split {n_pairs} pairs into {len(chunks)} chunks of size {chunk_size}")

    # 1. Veto features
    if veto_cache_file.exists():
        print(f"Veto cache exists at {veto_cache_file}, loading...")
        veto_df = joblib.load(veto_cache_file)
    else:
        print(f"\nComputing 14 veto features using {n_workers} parallel workers...")
        t_v0 = time.time()
        work_args = [(c, db_path) for c in chunks]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_veto_worker, work_args))
        veto_df = pd.concat(results, ignore_index=True)
        joblib.dump(veto_df, veto_cache_file)
        print(f"Computed & saved veto features ({veto_df.shape}) in {time.time()-t_v0:.1f}s to {veto_cache_file}")

    # 2. Enhanced features
    if enh_cache_file.exists():
        print(f"Enhanced cache exists at {enh_cache_file}, loading...")
        enh_df = joblib.load(enh_cache_file)
    else:
        print(f"\nComputing 9 enhanced features using {n_workers} parallel workers...")
        t_e0 = time.time()
        work_args = [(c, db_path) for c in chunks]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_enh_worker, work_args))
        enh_df = pd.concat(results, ignore_index=True)
        joblib.dump(enh_df, enh_cache_file)
        print(f"Computed & saved enhanced features ({enh_df.shape}) in {time.time()-t_e0:.1f}s to {enh_cache_file}")

    print(f"\nDone! Veto: {veto_df.shape}, Enhanced: {enh_df.shape}. Total time: {time.time()-t0:.1f}s")
    return veto_df, enh_df

if __name__ == '__main__':
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    base_dir = sys.argv[2] if len(sys.argv) > 2 else 'work/cap64_rebuilt'
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    build_supplemental_parallel(split, base_dir, n_workers=workers)
