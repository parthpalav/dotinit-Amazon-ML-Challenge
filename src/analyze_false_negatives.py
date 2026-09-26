"""Deep dive into model-rejected False Negatives (true matches in candidates scored below threshold)."""
import json, re, sqlite3
from collections import Counter
from pathlib import Path
import joblib, numpy as np, pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio

def main():
    val_data = joblib.load('work/real_v1/features/validation.joblib')
    p_val = np.load('work/improvements/veto_cache/p_val.npy')
    veto_val = joblib.load('work/improvements/veto_cache/validation_veto.joblib')
    pairs = val_data['pairs']
    y_val = pairs.label.to_numpy()
    
    threshold = 0.6750000000000003
    
    # Model-rejected FNs: candidate pairs that ARE true matches (y == 1) but p < threshold
    fn_mask = (y_val == 1) & (p_val < threshold)
    fn_pairs = pairs[fn_mask].copy()
    fn_pairs['prob'] = p_val[fn_mask]
    fn_veto = veto_val[fn_mask].copy()
    
    total_model_fns = len(fn_pairs)
    print(f'Total model-rejected False Negatives in candidate pool: {total_model_fns}')
    
    # Check probability distribution of these FNs
    prob_bins = {
        '0.60 <= P < 0.675 (Near Miss)': int(np.sum((fn_pairs.prob >= 0.60) & (fn_pairs.prob < threshold))),
        '0.40 <= P < 0.60': int(np.sum((fn_pairs.prob >= 0.40) & (fn_pairs.prob < 0.60))),
        '0.20 <= P < 0.40': int(np.sum((fn_pairs.prob >= 0.20) & (fn_pairs.prob < 0.40))),
        'P < 0.20 (Severe Miss)': int(np.sum(fn_pairs.prob < 0.20))
    }
    
    print('\nProbability Distribution of Model-Rejected FNs:')
    for b, c in prob_bins.items():
        print(f'  {b:30s}: {c:4d} ({c/total_model_fns*100:5.1f}%)')
        
    con = sqlite3.connect('work/real_v1/train/records.sqlite')
    con.row_factory = sqlite3.Row
    
    # Analyze root causes and check if veto features fired incorrectly
    veto_overtrigger = Counter()
    categories = Counter()
    
    samples = {
        'veto_overtrigger_number_conflict': [],
        'veto_overtrigger_name_conflict': [],
        'indic_script_name_divergence': [],
        'severe_address_divergence': [],
        'abbreviation_or_short_name': [],
        'subtle_low_confidence': []
    }
    
    detailed_cases = []
    
    for i, row in enumerate(fn_pairs.itertuples(index=False)):
        a_id = row.source1_entity_id
        t_id = row.candidate_entity_id
        prob = float(row.prob)
        veto_row = fn_veto.iloc[i]
        
        a = con.execute('SELECT * FROM anchors WHERE entity_id=?', (a_id,)).fetchone()
        t = con.execute('SELECT * FROM targets WHERE entity_id=?', (t_id,)).fetchone()
        
        n1, a1 = a['name_norm'], a['address_norm']
        n2, a2 = t['name_norm'], t['address_norm']
        
        n_sim = ratio(n1, n2) / 100.0
        a_sim = token_set_ratio(a1, a2) / 100.0
        
        # Check veto over-triggers on true matches:
        is_num_conflict = bool(veto_row['veto_address_number_conflict'] > 0.5)
        is_core_conflict = bool(veto_row['veto_core_name_conflict'] > 0.5)
        is_shared_bldg_veto = bool(veto_row['veto_shared_building_diff_name'] > 0.5)
        is_indic = bool(veto_row['veto_is_indic_target'] > 0.5)
        
        if is_num_conflict: veto_overtrigger['veto_address_number_conflict'] += 1
        if is_core_conflict: veto_overtrigger['veto_core_name_conflict'] += 1
        if is_shared_bldg_veto: veto_overtrigger['veto_shared_building_diff_name'] += 1
        
        case_info = {
            's1_id': a_id,
            'target_id': t_id,
            'prob': prob,
            's1_name': n1,
            's1_addr': a1,
            't_name': n2,
            't_addr': a2,
            'country': a['country_norm'],
            'name_sim': n_sim,
            'addr_sim': a_sim,
            'is_indic': is_indic
        }
        
        # Categorize primary failure reason
        assigned = False
        if is_indic and n_sim < 0.30:
            categories['indic_script_name_divergence'] += 1
            if len(samples['indic_script_name_divergence']) < 5:
                samples['indic_script_name_divergence'].append(case_info)
            assigned = True
        elif is_num_conflict and a_sim >= 0.70:
            categories['veto_overtrigger_number_conflict'] += 1
            if len(samples['veto_overtrigger_number_conflict']) < 5:
                samples['veto_overtrigger_number_conflict'].append(case_info)
            assigned = True
        elif is_core_conflict and a_sim >= 0.70:
            categories['veto_overtrigger_name_conflict'] += 1
            if len(samples['veto_overtrigger_name_conflict']) < 5:
                samples['veto_overtrigger_name_conflict'].append(case_info)
            assigned = True
        elif a_sim < 0.40 and n_sim >= 0.70:
            categories['severe_address_divergence'] += 1
            if len(samples['severe_address_divergence']) < 5:
                samples['severe_address_divergence'].append(case_info)
            assigned = True
        elif n_sim < 0.60 and a_sim < 0.60:
            categories['abbreviation_or_short_name'] += 1
            if len(samples['abbreviation_or_short_name']) < 5:
                samples['abbreviation_or_short_name'].append(case_info)
            assigned = True
            
        if not assigned:
            categories['subtle_low_confidence'] += 1
            if len(samples['subtle_low_confidence']) < 5:
                samples['subtle_low_confidence'].append(case_info)
                
        detailed_cases.append(case_info)
        
    con.close()
    
    print('\n--- Veto Features Over-Triggering on True Matches (False Veto Analysis) ---')
    for feat, c in veto_overtrigger.most_common():
        pct = (c / total_model_fns) * 100
        print(f'  {feat:35s}: {c:4d} ({pct:5.1f}% of rejected FNs)')
        
    print('\n--- Root Cause Breakdown of Model-Rejected False Negatives ---')
    for cat, count in categories.most_common():
        pct = (count / total_model_fns) * 100
        print(f'  {cat:35s}: {count:4d} ({pct:5.1f}%)')
        
    print('\n--- Concrete Examples per FN Failure Category ---')
    for cat, ex_list in samples.items():
        if not ex_list: continue
        print(f'\n=== Category: {cat} ===')
        for ex in ex_list[:2]:
            print(f'  Prob: {ex["prob"]:.4f} | S1: {ex["s1_id"]} ({ex["country"]}) vs T: {ex["target_id"]}')
            print(f'    S1: {ex["s1_name"]} | {ex["s1_addr"]}')
            print(f'    T : {ex["t_name"]} | {ex["t_addr"]}')
            print(f'    Sim: Name={ex["name_sim"]:.2f}, Addr={ex["addr_sim"]:.2f}')
            
    report = {
        'total_model_rejected_fns': total_model_fns,
        'probability_distribution': prob_bins,
        'veto_overtrigger_counts': dict(veto_overtrigger),
        'category_breakdown': {k: {'count': v, 'pct': round(v/total_model_fns*100, 2)} for k, v in categories.items()},
        'sample_cases': samples
    }
    
    out_path = Path('reports/improvements/fn_deepdive.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f'\nSaved False Negative deep-dive report to: {out_path}')

if __name__ == '__main__':
    main()
