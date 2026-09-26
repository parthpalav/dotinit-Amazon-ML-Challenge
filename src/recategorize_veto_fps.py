"""Re-diagnose the veto model's 873 actual residual False Positives."""
import json, re, sqlite3
from collections import Counter
from pathlib import Path
import joblib, numpy as np, pandas as pd

LEGAL_SUFFIXES = {'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'pllc', 'llp', 'co', 'gmbh', 'sa', 'sarl', 'foundation', 'trust'}

def main():
    val_data = joblib.load('work/real_v1/features/validation.joblib')
    p_val = np.load('work/improvements/veto_cache/p_val.npy')
    pairs = val_data['pairs']
    y_val = pairs.label.to_numpy()
    
    threshold = 0.6750000000000003
    accepted_mask = p_val >= threshold
    
    # All accepted positive predictions across the 10,000 validation anchors
    accepted_pairs = pairs[accepted_mask].copy()
    accepted_pairs['prob'] = p_val[accepted_mask]
    
    # Positive target counts across accepted predictions: target -> list of anchor IDs
    target_to_accepted_anchors = {}
    for row in accepted_pairs.itertuples(index=False):
        target_to_accepted_anchors.setdefault(row.candidate_entity_id, []).append(row.source1_entity_id)
        
    # Isolate False Positives (accepted but y == 0)
    fp_mask = accepted_mask & (y_val == 0)
    fp_pairs = pairs[fp_mask].copy()
    fp_pairs['prob'] = p_val[fp_mask]
    
    total_fps = len(fp_pairs)
    print(f'Total False Positives in veto model: {total_fps}')
    
    con = sqlite3.connect('work/real_v1/train/records.sqlite')
    con.row_factory = sqlite3.Row
    
    categories = Counter()
    samples = {
        'duplicate_target_conflict': [],
        'street_number_conflict': [],
        'shared_address_different_name': [],
        'franchise_or_multi_location': [],
        'legal_suffix_or_type_mismatch': [],
        'other': []
    }
    
    for row in fp_pairs.itertuples(index=False):
        a_id = row.source1_entity_id
        t_id = row.candidate_entity_id
        prob = float(row.prob)
        
        a = con.execute('SELECT * FROM anchors WHERE entity_id=?', (a_id,)).fetchone()
        t = con.execute('SELECT * FROM targets WHERE entity_id=?', (t_id,)).fetchone()
        
        n1, a1 = a['name_norm'], a['address_norm']
        n2, a2 = t['name_norm'], t['address_norm']
        
        tok1 = set(n1.split())
        tok2 = set(n2.split())
        nums1 = set(re.findall(r'\b\d+\b', a1))
        nums2 = set(re.findall(r'\b\d+\b', a2))
        
        # Check duplicate target conflict: does this target appear in accepted predictions for another anchor?
        claimed_anchors = target_to_accepted_anchors.get(t_id, [])
        is_dup_conflict = len(set(claimed_anchors) - {a_id}) > 0
        
        # We classify mutually exclusively or track multiple flags
        # Specifically isolate duplicate_target_conflict
        sample_item = {
            's1_id': a_id,
            'target_id': t_id,
            'prob': prob,
            's1_name': n1,
            'target_name': n2,
            's1_address': a1,
            'target_address': a2,
            'country': a['country_norm']
        }
        
        if is_dup_conflict:
            categories['duplicate_target_conflict'] += 1
            if len(samples['duplicate_target_conflict']) < 5:
                samples['duplicate_target_conflict'].append({
                    **sample_item,
                    'competing_anchors': list(set(claimed_anchors) - {a_id})
                })
                
        # Also check underlying text features
        if nums1 and nums2 and not (nums1 & nums2):
            categories['street_number_conflict'] += 1
            if len(samples['street_number_conflict']) < 5:
                samples['street_number_conflict'].append(sample_item)
                
        name_common = tok1 & tok2
        if len(name_common) == 0:
            categories['shared_address_different_name'] += 1
            if len(samples['shared_address_different_name']) < 5:
                samples['shared_address_different_name'].append(sample_item)
                
        leg1 = tok1 & LEGAL_SUFFIXES
        leg2 = tok2 & LEGAL_SUFFIXES
        if leg1 and leg2 and leg1 != leg2:
            categories['legal_suffix_or_type_mismatch'] += 1
            if len(samples['legal_suffix_or_type_mismatch']) < 5:
                samples['legal_suffix_or_type_mismatch'].append(sample_item)
                
        if (n1 == n2 or (len(name_common) >= 2 and len(name_common) == len(tok1))) and not (nums1 & nums2):
            categories['franchise_or_multi_location'] += 1
            if len(samples['franchise_or_multi_location']) < 5:
                samples['franchise_or_multi_location'].append(sample_item)
                
        # If none of the above
        if not is_dup_conflict and not (nums1 and nums2 and not (nums1 & nums2)) and len(name_common) > 0:
            categories['other'] += 1
            if len(samples['other']) < 5:
                samples['other'].append(sample_item)
                
    con.close()
    
    print('--- Veto Model Residual FP Categories (Non-exclusive) ---')
    breakdown = {}
    for cat, count in categories.most_common():
        pct = (count / total_fps) * 100
        print(f'  {cat:35s}: {count:4d} ({pct:5.1f}%)')
        breakdown[cat] = {'count': count, 'percentage': round(pct, 2)}
        
    out_record = {
        'total_fps': total_fps,
        'threshold': threshold,
        'breakdown': breakdown,
        'samples': samples
    }
    
    out_path = Path('reports/improvements/veto_fp_recategorization.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_record, indent=2))
    print(f'\nSaved fresh recategorization to {out_path}')

if __name__ == '__main__':
    main()
