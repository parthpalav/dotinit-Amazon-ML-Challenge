from io import StringIO
import pytest
from src.campaign_country_hybrid import merge_rows


def test_country_hybrid_only_replaces_france():
    head='source1_entity_id\tmatched_entity_ids\n'
    old=StringIO(head+'A\told-us\nB\told-fr\nC\told-in\n')
    new=StringIO(head+'A\tnew-us\nB\tnew-fr\nC\tnew-in\n')
    pool=StringIO('source1_entity_id\tcandidate_entity_ids\nA\told-us,new-us\nB\told-fr,new-fr\nC\told-in,new-in\n');out=StringIO()
    r=merge_rows([('A','US'),('B','France'),('C','India')],old,new,pool,out)
    assert out.getvalue()==head+'A\tnew-us\nB\told-fr\nC\tnew-in\n'
    assert r['US']['changed_anchors_vs_main']==r['India']['changed_anchors_vs_main']==0
    assert r['France']['changed_anchors_vs_main']==1
    assert r['ALL']['matches']==3


def test_country_hybrid_rejects_cross_input_ownership_collision():
    head='source1_entity_id\tmatched_entity_ids\n'
    old=StringIO(head+'A\tx\nB\tz\n');new=StringIO(head+'A\tz\nB\ty\n')
    pool=StringIO('source1_entity_id\tcandidate_entity_ids\nA\tx,z\nB\ty,z\n')
    with pytest.raises(ValueError,match='duplicate target'):
        merge_rows([('A','US'),('B','France')],old,new,pool,StringIO())
