from types import SimpleNamespace
import numpy as np
from src.raw_evidence import record,pair


def rec(n,a=''):
    return record(SimpleNamespace(business_name=n,business_address=a))


def test_legal_abbreviation_preserved_but_equivalent():
    d=pair(rec('Acme Pvt Ltd'),rec('Acme Private Limited'))
    assert d['legal_equal']==1 and d['name_left_remaining']==0
    changed=pair(rec('Acme Pvt Ltd'),rec('Acme LLP'))
    assert changed['legal_equal']==0


def test_number_group_distinction_and_missing_text_finite():
    d=pair(rec('Acme','36/1/1 Main Rd'),rec('Acme','36/1/3 Main Rd'))
    assert d['groups_equal']==0 and d['groups_both_present']==1
    assert all(np.isfinite(v) for v in pair(rec(''),rec('')).values())


def test_duplicate_tokens_cannot_match_one_token_twice():
    d=pair(rec('Vijay Vijay Clinic'),rec('Vijay Clinic'))
    assert d['name_left_remaining']==1 and d['name_aligned_low']==1
