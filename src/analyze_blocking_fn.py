"""Extract and categorize false negatives (missed true pairs) in blocking."""
import sqlite3, joblib, re, unicodedata
from collections import Counter
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio

GENERIC = set('and the of company corporation group services international inc ltd limited private pvt llc llp corp co sarl sas sasu sci eurl sa association'.split())

def fold(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def get_tokens(text):
    return set(fold(text).split())

def is_acronym(n1, n2):
    # Check if one is acronym of the other
    words1 = [w for w in fold(n1).split() if w not in GENERIC and w]
    words2 = [w for w in fold(n2).split() if w not in GENERIC and w]
    if not words1 or not words2:
        return False
    # If one is a single short token
    if len(words1) == 1 and len(words1[0]) >= 2:
        acr = ''.join(w[0] for w in words2)
        if words1[0] == acr or words1[0] in acr:
            return True
    if len(words2) == 1 and len(words2[0]) >= 2:
        acr = ''.join(w[0] for w in words1)
        if words2[0] == acr or words2[0] in acr:
            return True
    return False

def categorize(a_row, t_row):
    n1, a1, c1 = a_row['name_norm'], a_row['address_norm'], a_row['country_norm']
    n2, t_a2, c2 = t_row['name_norm'], t_row['address_norm'], t_row['country_norm']
    
    t1 = get_tokens(n1) - GENERIC
    t2 = get_tokens(n2) - GENERIC
    
    num1 = set(re.findall(r'\d+', a1))
    num2 = set(re.findall(r'\d+', t_a2))
    
    name_r = ratio(n1, n2)
    name_sort = token_sort_ratio(n1, n2)
    name_set = token_set_ratio(n1, n2)
    addr_set = token_set_ratio(a1, t_a2)
    
    reasons = []
    
    if c1 and c2 and c1 != c2:
        reasons.append('country_mismatch')
    elif not c1 or not c2:
        reasons.append('missing_country')
        
    if is_acronym(n1, n2):
        reasons.append('acronym_or_initials')
        
    if not (t1 & t2):
        reasons.append('zero_informative_name_tokens_overlap')
    elif len(t1 & t2) == 1 and name_set < 70:
        reasons.append('low_name_overlap')
        
    if addr_set < 50:
        reasons.append('severe_address_discrepancy')
    elif addr_set >= 70 and not (t1 & t2):
        reasons.append('strong_address_shared_but_name_differs')
        
    if num1 and num2 and not (num1 & num2):
        reasons.append('number_conflict')
        
    if name_r < 60 and name_sort < 65 and not is_acronym(n1, n2):
        reasons.append('heavy_name_divergence_dba')
        
    if not reasons:
        reasons.append('subtle_spelling_or_token_pruned')
        
    return {
        'name_ratio': name_r,
        'name_set_ratio': name_set,
        'addr_set_ratio': addr_set,
        'shared_tokens': list(t1 & t2),
        'country_pair': f"{c1}->{c2}",
        'reasons': reasons
    }

def main():
    val = joblib.load('work/real_v1/features/validation.joblib')
    val_cand_set = set(zip(val['pairs']['source1_entity_id'], val['pairs']['candidate_entity_id']))
    
    con = sqlite3.connect('work/real_v1/train/records.sqlite')
    con.row_factory = sqlite3.Row
    
    anchors_in_val = list(val['truth_counts'].keys())
    
    missed_pairs = []
    for start in range(0, len(anchors_in_val), 900):
        chunk = anchors_in_val[start:start+900]
        q = f'SELECT entity_id, matches FROM labels WHERE entity_id IN ({",".join(["?"]*len(chunk))})'
        for row in con.execute(q, chunk):
            s1 = row['entity_id']
            matches = [m.strip() for m in row['matches'].split(',') if m.strip()]
            for m in matches:
                if (s1, m) not in val_cand_set:
                    missed_pairs.append((s1, m))
                    
    print(f"Total missed true pairs: {len(missed_pairs)}")
    
    # Fetch details for missed pairs
    s1_ids = list({p[0] for p in missed_pairs})
    t_ids = list({p[1] for p in missed_pairs})
    
    s1_map = {}
    for start in range(0, len(s1_ids), 900):
        chunk = s1_ids[start:start+900]
        q = f'SELECT * FROM anchors WHERE entity_id IN ({",".join(["?"]*len(chunk))})'
        for r in con.execute(q, chunk):
            s1_map[r['entity_id']] = r
            
    t_map = {}
    for start in range(0, len(t_ids), 900):
        chunk = t_ids[start:start+900]
        q = f'SELECT * FROM targets WHERE entity_id IN ({",".join(["?"]*len(chunk))})'
        for r in con.execute(q, chunk):
            t_map[r['entity_id']] = r
            
    reason_counts = Counter()
    analyzed = []
    
    for s1, t in missed_pairs:
        a_row = s1_map[s1]
        t_row = t_map[t]
        info = categorize(a_row, t_row)
        for r in info['reasons']:
            reason_counts[r] += 1
        analyzed.append((s1, t, a_row, t_row, info))
        
    print("\n--- Missed Pairs Category Distribution ---")
    for r, count in reason_counts.most_common():
        pct = (count / len(missed_pairs)) * 100
        print(f"  {r:45s}: {count:4d} ({pct:5.1f}%)")
        
    print("\n--- Country Breakdown ---")
    country_counts = Counter(info['country_pair'] for _, _, _, _, info in analyzed)
    for cp, count in country_counts.most_common():
        pct = (count / len(missed_pairs)) * 100
        print(f"  {cp:25s}: {count:4d} ({pct:5.1f}%)")

    print("\n--- Top 25 Sample Missed Pairs ---")
    for i, (s1, t, a_row, t_row, info) in enumerate(analyzed[:25], 1):
        print(f"\n[{i}] S1: {s1} ({a_row['country_norm']}) | T: {t} ({t_row['country_norm']})")
        print(f"    S1 Name:    {a_row['name_norm']}")
        print(f"    T  Name:    {t_row['name_norm']}")
        print(f"    S1 Address: {a_row['address_norm']}")
        print(f"    T  Address: {t_row['address_norm']}")
        print(f"    Metrics:    Name Ratio={info['name_ratio']:.1f}, TokenSet={info['name_set_ratio']:.1f}, AddrSet={info['addr_set_ratio']:.1f}")
        print(f"    Reasons:    {', '.join(info['reasons'])}")

if __name__ == '__main__':
    main()
