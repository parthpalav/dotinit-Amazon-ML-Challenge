"""Build production supplement index with Indic Soundex and relaxed address channels."""
import os, time, json, sqlite3, shutil
from dataclasses import replace
from pathlib import Path
import joblib, numpy as np, pandas as pd
from src.config import Config
from src.supplement import build_supplement
from src.disk_blocking import DiskBlocker
from src.disk_store import connect, fetch_records

def main():
    os.environ['AMAZON_SUPPLEMENT_VARIANT']='phase8'
    target_dir = Path('work/cap64_rebuilt')
    target_train = target_dir / 'train'
    target_train.mkdir(parents=True, exist_ok=True)
    (target_dir / 'native').mkdir(parents=True, exist_ok=True)
    
    real_train = Path('work/real_v1/train').resolve()
    real_v1 = Path('work/real_v1').resolve()
    
    # Symlink base SQLite and text stores from real_v1 (zero extra disk space)
    # Only immutable inputs are shared. Packed text/offsets must be private:
    # build_supplement opens those outputs for writing.
    files_to_link = ['records.sqlite', 'blocking_index.bin', 'store_manifest.json']
    for fname in files_to_link:
        src = real_train / fname
        dst = target_train / fname
        if not dst.exists() and not dst.is_symlink():
            try:os.link(src, dst)
            except OSError:shutil.copy2(src, dst)
            print(f"Linked {fname}")
            
    for item in ['candidate_ranker.joblib', 'feature_engineer.joblib', 'entity_selection.npz']:
        src = real_v1 / item
        dst = target_dir / item
        if not dst.exists() and not dst.is_symlink():
            try:os.link(src, dst)
            except OSError:shutil.copy2(src, dst)
            print(f"Linked root file: {item}")
            
    cfg = Config.load('config/windows.json' if os.name=='nt' else 'config/real.json')
    cfg = replace(cfg, working_dir=str(target_dir), retrieval_posting_limit=500, max_candidates_per_anchor=64)
    
    print("\n--- Building Rebuilt Supplement Index ---")
    t0 = time.time()
    manifest = build_supplement(cfg, 'train')
    print(f"Supplement index built in {time.time()-t0:.1f}s")
    print(f"Manifest: {manifest}")
    
    # Verify with DiskBlocker
    print("\nInitializing DiskBlocker with new supplement index...")
    blocker = DiskBlocker(cfg, 'train')
    print(f"Blocker initialized. Native: {blocker.native.lib}, Extra: {blocker.extra.lib}")
    
    # Measure recall on 784 diagnostic missed pairs
    print("\n--- Measuring Recall on the 784 Missed Pairs ---")
    val_data = joblib.load('work/real_v1/features/validation.joblib')
    v1_pairs = set(zip(val_data['pairs'].source1_entity_id, val_data['pairs'].candidate_entity_id))
    
    con = connect(target_train / 'records.sqlite', readonly=True)
    con.row_factory = sqlite3.Row
    val_anchors = list(val_data['truth_counts'].keys())
    
    missed_pairs = []
    for start in range(0, len(val_anchors), 900):
        chunk = val_anchors[start:start+900]
        q = f'SELECT entity_id, matches FROM labels WHERE entity_id IN ({",".join(["?"]*len(chunk))})'
        for row in con.execute(q, chunk):
            s1 = row['entity_id']
            matches = [m.strip() for m in row['matches'].split(',') if m.strip()]
            for m in matches:
                if (s1, m) not in v1_pairs:
                    missed_pairs.append((s1, m))
                    
    missed_s1_set = {p[0] for p in missed_pairs}
    print(f"Total diagnostic missed true pairs: {len(missed_pairs)} across {len(missed_s1_set)} anchors")
    
    # Fetch integer rids for missed anchors
    missed_list = list(missed_s1_set)
    rids = []
    for start in range(0, len(missed_list), 900):
        chunk = missed_list[start:start+900]
        q = f'SELECT rid FROM anchors WHERE entity_id IN ({",".join(["?"]*len(chunk))})'
        for row in con.execute(q, chunk):
            rids.append(row[0])
            
    anchors_to_test = fetch_records(con, 'anchors', rids)
    pairs_new, _, stats_new, truth_new = blocker.retrieve(anchors_to_test, with_truth=True)
    con.close()
    
    new_pair_set = set(zip(pairs_new.source1_entity_id, pairs_new.candidate_entity_id))
    recovered = [p for p in missed_pairs if p in new_pair_set]
    
    print(f"\n=======================================================")
    print(f"DIAGNOSTIC 784 MISSED PAIRS RECOVERY REPORT:")
    print(f"  Total Missed Pairs Evaluated:  {len(missed_pairs)}")
    print(f"  Pairs Recovered by New Index:  {len(recovered)} ({len(recovered)/len(missed_pairs)*100:.2f}%)")
    print(f"  Still Missed:                  {len(missed_pairs) - len(recovered)} ({(len(missed_pairs)-len(recovered))/len(missed_pairs)*100:.2f}%)")
    print(f"=======================================================\n")
    
    # Check overall validation candidate recall projection
    baseline_retained = 33892
    total_true = 34676
    new_val_recall = (baseline_retained + len(recovered)) / total_true
    print(f"Projected Validation Candidate Recall: {baseline_retained + len(recovered)} / {total_true} = {new_val_recall*100:.2f}%")
    
    report = {
        'status': 'rebuilt_index_success',
        'supplement_manifest': manifest,
        'missed_pairs_evaluated': len(missed_pairs),
        'missed_pairs_recovered': len(recovered),
        'recovery_pct': round(len(recovered)/len(missed_pairs)*100, 2),
        'projected_candidate_recall': new_val_recall
    }
    Path('reports/improvements/phase7_rebuilt_index_verification.json').write_text(json.dumps(report, indent=2))
    print("Saved report to reports/improvements/phase7_rebuilt_index_verification.json")

if __name__ == '__main__':
    main()
