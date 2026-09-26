"""Enhanced features for missing address recovery, tempered vetoes, and candidate margins."""
import re, unicodedata
import numpy as np, pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio

GENERIC_NAME = set('and the of company corporation group services international inc ltd limited private pvt llc llp corp co sarl sas sasu sci eurl sa association pllc'.split())
GENERIC_ADDR = set('rd st ave ln blvd dr near main building apt ste and the floor block flat opposite near unit pmb'.split())

def compute_enhanced_features(pairs, con, p_preliminary=None, record_maps=None):
    """Compute missing-address recovery features, tempered vetoes, and margin signals."""
    from .disk_store import fetch_records
    
    a_rids = pairs.anchor_rid.unique()
    t_rids = pairs.target_rid.unique()
    
    if record_maps is None:
        anchors = fetch_records(con, 'anchors', a_rids).set_index('entity_id')
        targets = fetch_records(con, 'targets', t_rids).set_index('entity_id')
    else:
        from types import SimpleNamespace
        anchors, targets = (SimpleNamespace(loc=records) for records in record_maps)
    
    # Pre-calculate margin signals if preliminary probabilities are supplied
    if p_preliminary is not None:
        p_df = pairs[['source1_entity_id']].copy()
        p_df['prob'] = p_preliminary
        p_df['source'] = pairs.candidate_entity_id.str[:2]
        max_p = p_df.groupby(['source1_entity_id', 'source'])['prob'].transform('max')
        margins = (max_p - p_df['prob']).to_numpy(dtype=np.float32)
        is_top = (margins <= 1e-6).astype(np.float32)
    else:
        margins = np.zeros(len(pairs), dtype=np.float32)
        is_top = np.ones(len(pairs), dtype=np.float32)
        
    rows = []
    for i, p in enumerate(pairs.itertuples(index=False)):
        a = anchors.loc[p.source1_entity_id]
        t = targets.loc[p.candidate_entity_id]
        
        n1, a1 = a['name_norm'], a['address_norm']
        n2, a2 = t['name_norm'], t['address_norm']
        
        # 1. Missing address indicators
        is_target_addr_missing = float(len(a2.strip()) == 0)
        is_anchor_addr_missing = float(len(a1.strip()) == 0)
        addr_missing_either = float(is_target_addr_missing or is_anchor_addr_missing)
        
        n_ratio = ratio(n1, n2) / 100.0
        n_set = token_set_ratio(n1, n2) / 100.0
        a_set = token_set_ratio(a1, a2) / 100.0 if not addr_missing_either else 0.50
        
        # Explicit recovery feature: target address is missing but name similarity is high
        name_dominant_blank_addr = float(is_target_addr_missing and n_ratio >= 0.75)
        name_exact_blank_addr = float(is_target_addr_missing and n1 == n2)
        
        # 2. Tempered veto features
        nums1 = set(re.findall(r'\b\d+\b', a1))
        nums2 = set(re.findall(r'\b\d+\b', a2))
        raw_num_conflict = float(bool(nums1 and nums2 and not (nums1 & nums2)))
        
        tok1 = set(n1.split()) - GENERIC_NAME
        tok2 = set(n2.split()) - GENERIC_NAME
        raw_core_conflict = float(bool(tok1 and tok2 and not (tok1 & tok2)))
        
        # Tempering conditions:
        # A number conflict is tempered if the street/locality and name are near-identical (campus/relocation)
        num_tempered = raw_num_conflict
        if raw_num_conflict > 0.5 and n_ratio >= 0.85 and a_set >= 0.80:
            num_tempered = 0.0 # Suppress veto for legitimate relocation/campus
            
        # A core name conflict is tempered if the full address is >= 95% identical (DBA/rebrand at same location)
        name_tempered = raw_core_conflict
        if raw_core_conflict > 0.5 and a_set >= 0.95:
            name_tempered = 0.0 # Suppress veto for legitimate DBA/acquisition at exact same address
            
        rows.append({
            'feat_is_target_addr_missing': is_target_addr_missing,
            'feat_is_anchor_addr_missing': is_anchor_addr_missing,
            'feat_addr_missing_either': addr_missing_either,
            'feat_name_dominant_blank_addr': name_dominant_blank_addr,
            'feat_name_exact_blank_addr': name_exact_blank_addr,
            'feat_veto_num_conflict_tempered': num_tempered,
            'feat_veto_name_conflict_tempered': name_tempered,
            'feat_margin_to_best_candidate': margins[i],
            'feat_is_top_candidate': is_top[i]
        })
        
    return pd.DataFrame(rows, dtype=np.float32, index=pairs.index)
