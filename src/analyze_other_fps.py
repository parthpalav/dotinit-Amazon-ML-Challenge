"""Deep dive into the 620 'Other' False Positives to test target-side duplication."""
import json, re, sqlite3
from collections import Counter
from pathlib import Path
import joblib, numpy as np, pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio

LEGAL_SUFFIXES = {'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'pllc', 'llp', 'co', 'gmbh', 'sa', 'sarl', 'foundation', 'trust'}

def main():
    val_data = joblib.load('work/real_v1/features/validation.joblib')
    p_val = np.load('work/improvements/veto_cache/p_val.npy')
    pairs = val_data['pairs']
    y_val = pairs.label.to_numpy()
    
    threshold = 0.6750000000000003
    accepted_mask = p_val >= threshold
    fp_mask = accepted_mask & (y_val == 0)
    
    fp_pairs = pairs[fp_mask].copy()
    fp_pairs['prob'] = p_val[fp_mask]
    
    con = sqlite3.connect('work/real_v1/train/records.sqlite')
    con.row_factory = sqlite3.Row
    
    # Isolate the 620 'other' FPs using the exact same rule from recategorize_veto_fps
    other_fps = []
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
        
        name_common = tok1 & tok2
        num_conflict = nums1 and nums2 and not (nums1 & nums2)
        
        # Check if it was caught by other categories:
        # 1. num_conflict
        # 2. shared_address_different_name: len(name_common) == 0
        # 3. legal_suffix_or_type_mismatch
        # 4. franchise_or_multi_location: (n1==n2 or len(name_common)>=2 and len(name_common)==len(tok1)) and not (nums1 & nums2)
        leg1 = tok1 & LEGAL_SUFFIXES
        leg2 = tok2 & LEGAL_SUFFIXES
        leg_conflict = leg1 and leg2 and leg1 != leg2
        franchise = (n1 == n2 or (len(name_common) >= 2 and len(name_common) == len(tok1))) and not (nums1 & nums2)
        
        if not num_conflict and len(name_common) > 0 and not leg_conflict and not franchise:
            other_fps.append({
                'anchor_id': a_id,
                'target_id': t_id,
                'prob': prob,
                'a_name': n1,
                'a_addr': a1,
                't_name': n2,
                't_addr': a2,
                'country': a['country_norm'],
                't_source': t_id[:2]
            })
            
    print(f'Total isolated "Other" False Positives: {len(other_fps)}')
    
    # Map all anchor candidate probabilities for margin calculation: (anchor, target) -> prob
    pair_prob_map = dict(zip(zip(pairs.source1_entity_id, pairs.candidate_entity_id), p_val))
    
    # Now analyze against ground truth
    singleton_anchor_count = 0
    has_truth_anchor_count = 0
    
    similarity_bins = {
        '>= 0.90': 0,
        '0.80 - 0.89': 0,
        '0.70 - 0.79': 0,
        '0.50 - 0.69': 0,
        '< 0.50': 0
    }
    
    same_source_count = 0
    diff_source_count = 0
    
    detailed_comparisons = []
    margin_distribution = []
    
    for item in other_fps:
        a_id = item['anchor_id']
        t_fp_id = item['target_id']
        
        lbl_row = con.execute('SELECT matches FROM labels WHERE entity_id=?', (a_id,)).fetchone()
        matches_str = lbl_row['matches'] if lbl_row else ''
        true_targets = [m.strip() for m in matches_str.split(',') if m.strip()]
        
        if not true_targets:
            singleton_anchor_count += 1
            item['truth_type'] = 'true_singleton'
            item['max_sim_to_true'] = 0.0
            similarity_bins['< 0.50'] += 1
            continue
            
        has_truth_anchor_count += 1
        item['truth_type'] = 'has_truth_matches'
        
        # Compare falsely accepted target against each ground-truth true target
        best_sim = -1.0
        best_true_info = None
        
        for true_id in true_targets:
            t_true = con.execute('SELECT * FROM targets WHERE entity_id=?', (true_id,)).fetchone()
            if not t_true: continue
            
            true_n, true_a = t_true['name_norm'], t_true['address_norm']
            n_sim = ratio(item['t_name'], true_n) / 100.0
            a_sim = token_set_ratio(item['t_addr'], true_a) / 100.0
            combined_sim = 0.5 * n_sim + 0.5 * a_sim
            
            # Probability of true target
            p_true = float(pair_prob_map.get((a_id, true_id), 0.0))
            
            if combined_sim > best_sim:
                best_sim = combined_sim
                best_true_info = {
                    'true_id': true_id,
                    'true_source': true_id[:2],
                    'true_name': true_n,
                    'true_addr': true_a,
                    'name_sim': n_sim,
                    'addr_sim': a_sim,
                    'combined_sim': combined_sim,
                    'p_true': p_true,
                    'p_fp': item['prob'],
                    'prob_margin': item['prob'] - p_true
                }
                
        item['best_comparison'] = best_true_info
        item['max_sim_to_true'] = best_sim
        margin_distribution.append(best_true_info['prob_margin'])
        
        if best_true_info['true_source'] == item['t_source']:
            same_source_count += 1
        else:
            diff_source_count += 1
            
        if best_sim >= 0.90:
            similarity_bins['>= 0.90'] += 1
        elif best_sim >= 0.80:
            similarity_bins['0.80 - 0.89'] += 1
        elif best_sim >= 0.70:
            similarity_bins['0.70 - 0.79'] += 1
        elif best_sim >= 0.50:
            similarity_bins['0.50 - 0.69'] += 1
        else:
            similarity_bins['< 0.50'] += 1
            
        detailed_comparisons.append(item)
        
    con.close()
    
    total = len(other_fps)
    print('\n================ FP "OTHER" DEEP DIVE RESULTS ================')
    print(f'Total "Other" FPs Analyzed: {total}')
    print(f'  Anchors that are True Singletons (no match in GT): {singleton_anchor_count} ({singleton_anchor_count/total*100:.1f}%)')
    print(f'  Anchors with Valid True Matches in GT:           {has_truth_anchor_count} ({has_truth_anchor_count/total*100:.1f}%)')
    
    print('\nTarget-Side Similarity to Ground-Truth True Target:')
    for b, c in similarity_bins.items():
        print(f'  Similarity {b:15s}: {c:4d} ({c/total*100:5.1f}%)')
        
    near_dup_count = similarity_bins['>= 0.90']
    print(f'\nHypothesis Test: Near-duplicate Target Duplication (Similarity >= 0.90):')
    print(f'  CONFIRMED: {near_dup_count} / {total} ({near_dup_count/total*100:.1f}%) of "Other" FPs have a near-identical True Target!')
    
    print(f'\nSource Alignment (when true target exists):')
    print(f'  Same Source (e.g. S2 vs S2 or S3 vs S3): {same_source_count} ({same_source_count/has_truth_anchor_count*100:.1f}%)')
    print(f'  Cross Source (e.g. S2 vs S3):            {diff_source_count} ({diff_source_count/has_truth_anchor_count*100:.1f}%)')
    
    # Margin analysis:
    margins = np.array(margin_distribution)
    print('\nConfidence Margin (P_FP - P_TrueTarget) Distribution:')
    print(f'  Mean Margin:   {margins.mean():.4f}')
    print(f'  Median Margin: {np.median(margins):.4f}')
    print(f'  Margin <= 0.05 (near-tie): {np.sum(margins <= 0.05)} ({np.sum(margins <= 0.05)/len(margins)*100:.1f}%)')
    print(f'  Margin <= 0.10:           {np.sum(margins <= 0.10)} ({np.sum(margins <= 0.10)/len(margins)*100:.1f}%)')
    print(f'  Margin <= 0.20:           {np.sum(margins <= 0.20)} ({np.sum(margins <= 0.20)/len(margins)*100:.1f}%)')
    
    # Concrete examples
    print('\n--- Concrete Examples of Target-Side Duplication ---')
    examples = [item for item in detailed_comparisons if item['max_sim_to_true'] >= 0.90][:5]
    for i, ex in enumerate(examples, 1):
        comp = ex['best_comparison']
        print(f'\n[{i}] Anchor: {ex["anchor_id"]} ({ex["country"]})')
        print(f'    Anchor Name:  {ex["a_name"]} | {ex["a_addr"]}')
        print(f'    FP Target ({ex["t_source"]}):    {comp["true_id"] if False else ex["target_id"]} (P={ex["prob"]:.4f})')
        print(f'      {ex["t_name"]} | {ex["t_addr"]}')
        print(f'    TRUE Target ({comp["true_source"]}):  {comp["true_id"]} (P={comp["p_true"]:.4f})')
        print(f'      {comp["true_name"]} | {comp["true_addr"]}')
        print(f'    Similarity: Name={comp["name_sim"]:.3f}, Addr={comp["addr_sim"]:.3f} -> Combined={comp["combined_sim"]:.3f}')
        print(f'    Prob Margin (FP - True): {comp["prob_margin"]:+.4f}')
        
    report = {
        'total_other_fps': total,
        'singleton_anchors': singleton_anchor_count,
        'anchors_with_ground_truth': has_truth_anchor_count,
        'similarity_distribution': similarity_bins,
        'near_duplicate_count_gte_09': near_dup_count,
        'near_duplicate_pct_gte_09': round(near_dup_count / total * 100, 2),
        'same_source_conflicts': same_source_count,
        'diff_source_conflicts': diff_source_count,
        'margin_analysis': {
            'mean_margin': float(margins.mean()),
            'median_margin': float(np.median(margins)),
            'margin_le_0_05_count': int(np.sum(margins <= 0.05)),
            'margin_le_0_05_pct': float(np.sum(margins <= 0.05) / len(margins)),
            'margin_le_0_10_count': int(np.sum(margins <= 0.10)),
            'margin_le_0_10_pct': float(np.sum(margins <= 0.10) / len(margins))
        },
        'sample_cases': examples
    }
    
    out_path = Path('reports/improvements/other_fp_deepdive.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f'\nSaved detailed deep dive to: {out_path}')

if __name__ == '__main__':
    main()
