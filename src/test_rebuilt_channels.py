"""Test the exact C++ key generation logic against the 784 missed pairs."""
import re, sqlite3, unicodedata
from collections import Counter
import joblib, numpy as np, pandas as pd

GENERIC_NAME = set('and the of company corporation group services international inc ltd limited private pvt llc llp corp co sarl sas sasu sci eurl sa association pllc'.split())
GENERIC_ADDR = set('rd st ave ln blvd dr near main building apt ste and the floor block flat opposite unit pmb road street lane avenue drive'.split())
GENERIC_SX = {'P613', 'L533', 'S612', 'C515'}

CONSONANTS = {
    0x15: 'k', 0x16: 'k', 0x17: 'g', 0x18: 'g', 0x19: 'n',
    0x1A: 'c', 0x1B: 'c', 0x1C: 'j', 0x1D: 'j', 0x1E: 'n',
    0x1F: 't', 0x20: 't', 0x21: 'd', 0x22: 'd', 0x23: 'n',
    0x24: 't', 0x25: 't', 0x26: 'd', 0x27: 'd', 0x28: 'n',
    0x2A: 'p', 0x2B: 'p', 0x2C: 'b', 0x2D: 'b', 0x2E: 'm',
    0x2F: 'y', 0x30: 'r', 0x32: 'l', 0x35: 'v',
    0x36: 's', 0x37: 's', 0x38: 's', 0x39: 'h'
}
VOWELS = {
    0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0A: 'u',
    0x0F: 'e', 0x10: 'a', 0x13: 'o', 0x14: 'a',
    0x3E: 'a', 0x3F: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u',
    0x47: 'e', 0x48: 'a', 0x4B: 'o', 0x4C: 'a'
}

def transliterate_indic(text):
    out = []
    for c in text:
        cp = ord(c)
        if 0x0900 <= cp <= 0x0D7F:
            off = (cp - 0x0900) % 0x80
            if off in CONSONANTS: out.append(CONSONANTS[off])
            elif off in VOWELS: out.append(VOWELS[off])
        else:
            out.append(c)
    return ''.join(out)

def soundex(name):
    name = name.upper()
    if not name: return ''
    codes = {'B': '1', 'F': '1', 'P': '1', 'V': '1',
             'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
             'D': '3', 'T': '3',
             'L': '4',
             'M': '5', 'N': '5',
             'R': '6'}
    first = name[0]
    tail = [codes.get(c, '') for c in name[1:]]
    res = [first]
    prev = codes.get(first, '')
    for c in tail:
        if c and c != prev: res.append(c)
        prev = c
    return (''.join(res) + '0000')[:4]

def get_keys(name, address, country):
    keys = set()
    if not country: return keys
    
    # 1. Transliterate name and get tokens
    t_name = transliterate_indic(name)
    n_toks = [w for w in t_name.split() if w not in GENERIC_NAME and len(w) > 2]
    
    # Soundex
    sxs = {soundex(w) for w in n_toks} - GENERIC_SX
    
    # Address tokens
    nums = set(re.findall(r'\b\d+\b', address))
    words = [w for w in address.split() if w not in GENERIC_ADDR and len(w) >= 3 and not w.isdigit()]
    
    # Channel 1a: SXN (Soundex + Address Number)
    for sx in sxs:
        for num in nums:
            keys.add(f"SXN|{country}|{sx}|{num}")
            
    # Channel 1b: SXW (Soundex + Address Word)
    for sx in sxs:
        for w in words[:6]:
            keys.add(f"SXW|{country}|{sx}|{w}")
            
    # Channel 2a: AN (Address Number + Word)
    for num in list(nums)[:4]:
        for w in words[:6]:
            keys.add(f"AN|{country}|{num}|{w}")
            
    # Channel 2b: AW2 (Address Word Pairs)
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            keys.add(f"AW2|{country}|{words[i]}|{words[j]}")
            
    # Channel 3: NP (Name word pairs)
    for i in range(len(n_toks)):
        for j in range(i+1, len(n_toks)):
            keys.add(f"NP|{country}|{n_toks[i]}|{n_toks[j]}")
            
    return keys

def main():
    val_data = joblib.load('work/real_v1/features/validation.joblib')
    val_pairs = val_data['pairs']
    existing_cand_set = set(zip(val_pairs.source1_entity_id, val_pairs.candidate_entity_id))
    
    con = sqlite3.connect('work/real_v1/train/records.sqlite')
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
                if (s1, m) not in existing_cand_set:
                    missed_pairs.append((s1, m))
                    
    print(f"Total missed true pairs: {len(missed_pairs)}")
    
    # Fetch details
    s1_ids = list({p[0] for p in missed_pairs})
    t_ids = list({p[1] for p in missed_pairs})
    
    s1_map = {}
    for start in range(0, len(s1_ids), 900):
        chunk = s1_ids[start:start+900]
        for r in con.execute(f'SELECT * FROM anchors WHERE entity_id IN ({",".join(["?"]*len(chunk))})', chunk):
            s1_map[r['entity_id']] = r
            
    t_map = {}
    for start in range(0, len(t_ids), 900):
        chunk = t_ids[start:start+900]
        for r in con.execute(f'SELECT * FROM targets WHERE entity_id IN ({",".join(["?"]*len(chunk))})', chunk):
            t_map[r['entity_id']] = r
    con.close()
    
    recovered_sxn = 0
    recovered_sxw = 0
    recovered_an = 0
    recovered_aw2 = 0
    recovered_np = 0
    recovered_any = 0
    
    for s1, t in missed_pairs:
        a = s1_map[s1]
        tgt = t_map[t]
        
        k_a = get_keys(a['name_norm'], a['address_norm'], a['country_norm'])
        k_t = get_keys(tgt['name_norm'], tgt['address_norm'], tgt['country_norm'])
        
        shared = k_a & k_t
        has_sxn = any(k.startswith('SXN|') for k in shared)
        has_sxw = any(k.startswith('SXW|') for k in shared)
        has_an = any(k.startswith('AN|') for k in shared)
        has_aw2 = any(k.startswith('AW2|') for k in shared)
        has_np = any(k.startswith('NP|') for k in shared)
        
        if has_sxn: recovered_sxn += 1
        if has_sxw: recovered_sxw += 1
        if has_an: recovered_an += 1
        if has_aw2: recovered_aw2 += 1
        if has_np: recovered_np += 1
        if shared: recovered_any += 1
        
    print(f"\n--- Diagnostic Missed Pairs Recovery Breakdown ({len(missed_pairs)} total missed) ---")
    print(f"Recovered by SXN (Soundex + Number): {recovered_sxn} ({recovered_sxn/len(missed_pairs)*100:.1f}%)")
    print(f"Recovered by SXW (Soundex + Word):   {recovered_sxw} ({recovered_sxw/len(missed_pairs)*100:.1f}%)")
    print(f"Recovered by AN (Address Num+Word):  {recovered_an} ({recovered_an/len(missed_pairs)*100:.1f}%)")
    print(f"Recovered by AW2 (Address Word-Word):{recovered_aw2} ({recovered_aw2/len(missed_pairs)*100:.1f}%)")
    print(f"Recovered by NP (Name Word Pairs):   {recovered_np} ({recovered_np/len(missed_pairs)*100:.1f}%)")
    print(f"Total Recovered by Combined Keys:    {recovered_any} / {len(missed_pairs)} ({recovered_any/len(missed_pairs)*100:.1f}%)")
    
    total_true = 34676
    baseline_retained = 33892
    new_candidate_recall = (baseline_retained + recovered_any) / total_true
    print(f"\nProjected Candidate Recall: {baseline_retained + recovered_any}/{total_true} = {new_candidate_recall*100:.2f}%")

if __name__ == '__main__':
    main()
