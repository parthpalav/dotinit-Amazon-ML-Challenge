"""Preserve raw distinctions as learned evidence, never as hard match vetoes."""
import re
from collections import Counter
import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz.fuzz import ratio, token_sort_ratio
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment
from .disk_store import fetch_records

VERSION = 1
LEGAL = {'pvt': 'private', 'ltd': 'limited', 'inc': 'incorporated',
         'corp': 'corporation', 'co': 'company'}
FORMS = ['private', 'limited', 'incorporated', 'corporation', 'company',
         'llp', 'llc', 'plc', 'sa', 'sarl', 'sas', 'gmbh']


def text(s):
    return anyascii(str(s or '')).lower()


def record(r):
    n = text(r.business_name); a = text(r.business_address)
    nt = re.findall(r'[a-z0-9]+', n)
    nt = [LEGAL.get(t, t) for t in nt]
    core = [t for t in nt if t not in FORMS and t not in ('and', 'the')]
    return n, a, nt, core


def pair(left, right):
    n, a, nt, core = left; m, b, mt, other = right
    d = {'name_ratio': ratio(n, m)/100, 'name_sort': token_sort_ratio(n, m)/100,
         'address_ratio': ratio(a, b)/100, 'name_edit_count': Levenshtein.distance(n, m),
         'name_length_delta': len(n)-len(m)}
    for f in FORMS:
        d['left_'+f] = int(f in nt); d['right_'+f] = int(f in mt)
    lf = set(nt)&set(FORMS); rf = set(mt)&set(FORMS)
    d['legal_equal'] = int(lf == rf); d['legal_difference'] = len(lf ^ rf)
    for label, x, y in [('name', core, other), ('address', re.findall(r'[a-z0-9]+', a), re.findall(r'[a-z0-9]+', b))]:
        # Cancel exact tokens, then align each remaining token at most once.
        cx, cy = Counter(x), Counter(y); shared = cx & cy
        xx = list((cx-shared).elements())[:32]; yy = list((cy-shared).elements())[:32]
        d[label+'_exact_tokens'] = sum(shared.values())
        d[label+'_left_remaining'] = len(xx); d[label+'_right_remaining'] = len(yy)
        size = max(len(xx), len(yy)); scores = np.zeros((size, size))
        for i, u in enumerate(xx):
            for j, v in enumerate(yy): scores[i, j] = ratio(u, v)/100
        if size:
            ii, jj = linear_sum_assignment(-scores); vals = scores[ii, jj]
            d[label+'_aligned_mean'] = float(vals.mean()); d[label+'_aligned_min'] = float(vals.min())
            d[label+'_aligned_low'] = int((vals < .7).sum())
            edits = [(xx[i], yy[j]) for i,j in zip(ii,jj) if i < len(xx) and j < len(yy)]
        else:
            d[label+'_aligned_mean'] = 1.; d[label+'_aligned_min'] = 1.; d[label+'_aligned_low'] = 0; edits=[]
        d[label+'_short_edits'] = sum(min(len(u),len(v)) <= 3 for u,v in edits)
        d[label+'_one_char_edits'] = sum(Levenshtein.distance(u,v)==1 for u,v in edits)
        d[label+'_numeric_edits'] = sum(any(c.isdigit() for c in u+v) for u,v in edits)
        d[label+'_unmatched_short'] = sum(len(t)<=3 for t in xx+yy)
    for tag, pat in [('groups',r'\d+(?:\s*[-/]\s*\d+)+'), ('numbers',r'\d+'),
                     ('units',r'(?:suite|ste|unit|apt|floor|fl|shop|plot|no)\.?\s*([a-z0-9/-]+)')]:
        x = re.findall(pat,a); y = re.findall(pat,b)
        x = [re.sub(r'\s+','',t) for t in x]; y = [re.sub(r'\s+','',t) for t in y]
        d[tag+'_left_count']=len(x); d[tag+'_right_count']=len(y)
        d[tag+'_equal']=int(x==y); d[tag+'_both_present']=int(bool(x and y))
        d[tag+'_difference']=len(set(x)^set(y)); d[tag+'_ratio']=ratio(' '.join(x),' '.join(y))/100
    return d


def features(pairs, con):
    anc = fetch_records(con,'anchors',pairs.anchor_rid.unique())
    tar = fetch_records(con,'targets',pairs.target_rid.unique())
    aa={r.rid:record(r) for r in anc.itertuples(index=False)}
    tt={r.rid:record(r) for r in tar.itertuples(index=False)}
    return pd.DataFrame([pair(aa[r.anchor_rid],tt[r.target_rid]) for r in pairs.itertuples(index=False)],index=pairs.index,dtype=np.float32).add_prefix('raw_')
