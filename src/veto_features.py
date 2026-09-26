"""Precision-focused veto features and acronym resolution."""
import re, unicodedata, math
import numpy as np, pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio

LEGAL_FORMS = {
    'llc': 1, 'pllc': 1,
    'inc': 2, 'corp': 2, 'corporation': 2,
    'ltd': 3, 'limited': 3, 'pvt': 3,
    'llp': 4,
    'co': 5, 'company': 5,
    'sa': 6, 'sarl': 6, 'sas': 6, 'sasu': 6, 'sci': 6, 'eurl': 6,
    'foundation': 7, 'trust': 7, 'association': 7
}

CATEGORIES = {
    'medical': {'clinic', 'hospital', 'dental', 'dds', 'md', 'pharma', 'pharmacy', 'health', 'healthcare', 'medical', 'surgery', 'care', 'medicine'},
    'education': {'school', 'college', 'academy', 'institute', 'university', 'education', 'classes', 'vidyalaya', 'gurukula'},
    'food': {'cafe', 'restaurant', 'bakery', 'pizza', 'grill', 'bar', 'kitchen', 'hotel', 'dining', 'foods', 'sweets', 'dhaba'},
    'finance': {'bank', 'capital', 'finance', 'financial', 'wealth', 'investment', 'securities', 'insurance', 'finserv'},
    'real_estate': {'realty', 'realestate', 'construction', 'builders', 'developers', 'properties', 'infra', 'infrastructure', 'developers'},
    'logistics': {'logistics', 'transport', 'cargo', 'movers', 'freight', 'travels', 'tourist', 'motors', 'auto'},
    'tech': {'technologies', 'software', 'it', 'solutions', 'tech', 'systems', 'digital', 'infotech'},
    'retail': {'mart', 'store', 'shop', 'enterprise', 'retail', 'supermart', 'trading', 'jewellers', 'boutique', 'jewellery'}
}

GENERIC_NAME = set('and the of company corporation group services international inc ltd limited private pvt llc llp corp co sarl sas sasu sci eurl sa association pllc'.split())
STOPWORDS = {'and', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', '&', 'de', 'du', 'la', 'le'}

def transliterate_indic(text):
    out = []
    for c in text:
        if '\u0900' <= c <= '\u0d7f':
            try:
                name = unicodedata.name(c)
                m = re.search(r'LETTER ([A-Z]+)', name)
                if m:
                    out.append(m.group(1).lower())
                else:
                    m2 = re.search(r'VOWEL SIGN ([A-Z]+)', name)
                    if m2:
                        out.append(m2.group(1).lower())
            except Exception:
                out.append(c)
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
        if c and c != prev:
            res.append(c)
        prev = c
    return (''.join(res) + '0000')[:4]

def extract_acronym_score(n1, n2):
    n1_clean = n1.lower().replace('&', ' and ')
    n2_clean = n2.lower().replace('&', ' and ')
    w1 = [w for w in re.findall(r'[a-z0-9]+', n1_clean) if w not in LEGAL_FORMS]
    w2 = [w for w in re.findall(r'[a-z0-9]+', n2_clean) if w not in LEGAL_FORMS]
    
    def test_pair(short_toks, long_toks):
        if len(short_toks) == 1 and len(long_toks) >= 2:
            acr_all = ''.join(w[0] for w in long_toks)
            acr_nostop = ''.join(w[0] for w in long_toks if w not in STOPWORDS)
            target = short_toks[0]
            if target == acr_all or target == acr_nostop:
                return 1.0
            if target in (acr_all, acr_nostop) or acr_nostop.startswith(target):
                return 0.7
        return 0.0

    return max(test_pair(w1, w2), test_pair(w2, w1))

def compute_veto_features(pairs, con):
    """Compute precision-focused veto features for a DataFrame of pairs."""
    from .disk_store import fetch_records
    
    a_rids = pairs.anchor_rid.unique()
    t_rids = pairs.target_rid.unique()
    
    anchors = fetch_records(con, 'anchors', a_rids).set_index('entity_id')
    targets = fetch_records(con, 'targets', t_rids).set_index('entity_id')
    
    rows = []
    for p in pairs.itertuples(index=False):
        a = anchors.loc[p.source1_entity_id]
        t = targets.loc[p.candidate_entity_id]
        
        n1, a1, c1 = a['name_norm'], a['address_norm'], a['country_norm']
        n2, a2, c2 = t['name_norm'], t['address_norm'], t['country_norm']
        
        # 1. Number extraction
        nums1 = set(re.findall(r'\b\d+\b', a1))
        nums2 = set(re.findall(r'\b\d+\b', a2))
        
        num_both = bool(nums1 and nums2)
        num_overlap = len(nums1 & nums2)
        num_conflict = int(num_both and num_overlap == 0)
        num_jaccard = num_overlap / max(1, len(nums1 | nums2)) if (nums1 or nums2) else 1.0
        
        # Leading / primary street number difference
        m1 = re.search(r'\b(\d+)\b', a1)
        m2 = re.search(r'\b(\d+)\b', a2)
        if m1 and m2:
            val1, val2 = int(m1.group(1)), int(m2.group(1))
            primary_num_diff = min(100.0, float(abs(val1 - val2)))
            primary_num_match = float(val1 == val2)
        else:
            primary_num_diff = 0.0
            primary_num_match = 0.5 # missing
            
        # 2. Name tokens & Core overlap
        tok1 = set(n1.split())
        tok2 = set(n2.split())
        core1 = tok1 - GENERIC_NAME
        core2 = tok2 - GENERIC_NAME
        
        core_overlap = len(core1 & core2)
        core_conflict = int(bool(core1 and core2) and core_overlap == 0)
        core_jaccard = core_overlap / max(1, len(core1 | core2)) if (core1 or core2) else 0.0
        
        # 3. Address overlap & Building sharing
        addr_tok1 = set(a1.split())
        addr_tok2 = set(a2.split())
        addr_jaccard = len(addr_tok1 & addr_tok2) / max(1, len(addr_tok1 | addr_tok2)) if (addr_tok1 or addr_tok2) else 0.0
        
        # Shared building but different business
        shared_building_diff_name = float(addr_jaccard >= 0.60 and core_overlap == 0)
        
        # 4. Acronym resolution
        acronym_score = extract_acronym_score(n1, n2)
        
        # 5. Legal form constraint
        leg1 = {LEGAL_FORMS[w] for w in tok1 if w in LEGAL_FORMS}
        leg2 = {LEGAL_FORMS[w] for w in tok2 if w in LEGAL_FORMS}
        legal_type_conflict = float(bool(leg1 and leg2 and not (leg1 & leg2)))
        
        # 6. Category cross-check
        cat1 = {c for c, words in CATEGORIES.items() if words & tok1}
        cat2 = {c for c, words in CATEGORIES.items() if words & tok2}
        cat_conflict = float(bool(cat1 and cat2 and not (cat1 & cat2)))
        cat_match = float(bool(cat1 & cat2))
        
        # 7. Indic transliteration & Phonetic soundex match
        is_indic = float(any(ord(c) > 591 for c in n2))
        if is_indic:
            n2_trans = transliterate_indic(n2)
            trans_ratio = ratio(n1, n2_trans) / 100.0
            t_toks = [w for w in n2_trans.split() if w not in GENERIC_NAME and len(w) > 2]
        else:
            trans_ratio = ratio(n1, n2) / 100.0
            t_toks = [w for w in tok2 if w not in GENERIC_NAME and len(w) > 2]
            
        a_toks = [w for w in tok1 if w not in GENERIC_NAME and len(w) > 2]
        a_sx = {soundex(w) for w in a_toks}
        t_sx = {soundex(w) for w in t_toks}
        phonetic_match = float(bool(a_sx & t_sx))
        
        rows.append({
            'veto_address_number_conflict': float(num_conflict),
            'veto_address_numeric_jaccard': float(num_jaccard),
            'veto_primary_num_diff': float(primary_num_diff),
            'veto_primary_num_match': float(primary_num_match),
            'veto_core_name_conflict': float(core_conflict),
            'veto_core_name_jaccard': float(core_jaccard),
            'veto_shared_building_diff_name': float(shared_building_diff_name),
            'veto_acronym_score': float(acronym_score),
            'veto_legal_type_conflict': float(legal_type_conflict),
            'veto_category_conflict': float(cat_conflict),
            'veto_category_match': float(cat_match),
            'veto_is_indic_target': float(is_indic),
            'veto_indic_trans_ratio': float(trans_ratio),
            'veto_phonetic_soundex_match': float(phonetic_match)
        })
        
    return pd.DataFrame(rows, dtype=np.float32, index=pairs.index)
