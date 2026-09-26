"""Diagnose why 200 of the 614 simulated recoverable pairs were not retained in production."""
import sqlite3, joblib
import pandas as pd, numpy as np
from src.config import Config
from src.disk_blocking import DiskBlocker
from src.disk_store import connect, fetch_records
from src.test_rebuilt_channels import get_keys

def main():
    val_data = joblib.load('work/real_v1/features/validation.joblib')
    v1_pairs = set(zip(val_data['pairs'].source1_entity_id, val_data['pairs'].candidate_entity_id))
    
    con = connect('work/cap64_rebuilt/train/records.sqlite', readonly=True)
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
    rids = [r[0] for r in con.execute(f'SELECT rid FROM anchors WHERE entity_id IN ({",".join(["?"]*len(missed_s1_set))})', list(missed_s1_set))]
    anchors = fetch_records(con, 'anchors', rids)
    
    cfg = Config.load('config/real.json')
    from dataclasses import replace
    cfg = replace(cfg, working_dir='work/cap64_rebuilt', retrieval_posting_limit=500, max_candidates_per_anchor=64)
    blocker = DiskBlocker(cfg, 'train')
    
    # Check raw_retrieve vs final retrieve
    rids_raw, masks, offsets, text, indexer, repeated = blocker.raw_retrieve(anchors)
    raw_pairs = set(zip(anchors.entity_id.to_numpy()[repeated], text['entity_id'][indexer]))
    
    final_pairs, _, stats, truth = blocker.retrieve(anchors, with_truth=True)
    final_pair_set = set(zip(final_pairs.source1_entity_id, final_pairs.candidate_entity_id))
    
    # Categorize the 784 missed pairs
    recovered_in_final = [p for p in missed_pairs if p in final_pair_set] # 414
    in_raw_but_dropped_by_cap = [p for p in missed_pairs if p in raw_pairs and p not in final_pair_set]
    not_in_raw = [p for p in missed_pairs if p not in raw_pairs]
    
    print(f"Total diagnostic missed pairs: {len(missed_pairs)}")
    print(f"1. Retained in final Cap 64:              {len(recovered_in_final)} ({len(recovered_in_final)/len(missed_pairs)*100:.1f}%)")
    print(f"2. Retrieved in Raw Index but dropped by Cap 64: {len(in_raw_but_dropped_by_cap)} ({len(in_raw_but_dropped_by_cap)/len(missed_pairs)*100:.1f}%)")
    print(f"3. Not retrieved in Raw Index:            {len(not_in_raw)} ({len(not_in_raw)/len(missed_pairs)*100:.1f}%)")
    
    # Check why not in raw:
    # Compare with simulation keys:
    sim_has_keys = 0
    suppressed_by_posting_limit = 0
    key_slot_truncation = 0
    
    s1_map = {r['entity_id']: r for r in anchors.to_dict('records')}
    t_map = {}
    for start in range(0, len(not_in_raw), 900):
        chunk = [p[1] for p in not_in_raw[start:start+900]]
        for r in con.execute(f'SELECT * FROM targets WHERE entity_id IN ({",".join(["?"]*len(chunk))})', chunk):
            t_map[r['entity_id']] = r
    con.close()
    
    sample_dropped_by_cap = []
    for p in in_raw_but_dropped_by_cap[:5]:
        sample_dropped_by_cap.append({'s1': p[0], 'target': p[1]})
        
    sample_not_in_raw = []
    for p in not_in_raw[:5]:
        sample_not_in_raw.append({'s1': p[0], 'target': p[1]})
        
    diag_report = {
        'total_missed_pairs': len(missed_pairs),
        'recovered_in_final_cap64': len(recovered_in_final),
        'retrieved_in_raw_dropped_by_cap': len(in_raw_but_dropped_by_cap),
        'not_retrieved_in_raw': len(not_in_raw),
        'sample_dropped_by_cap': sample_dropped_by_cap,
        'sample_not_in_raw': sample_not_in_raw
    }
    import json
    from pathlib import Path
    Path('reports/improvements/phase7_gap_diagnosis.json').write_text(json.dumps(diag_report, indent=2))
    print("\nSaved diagnosis to reports/improvements/phase7_gap_diagnosis.json")

if __name__ == '__main__':
    main()
