"""Integration tests on actual supplied records, with optional full-run checks."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import os
from itertools import zip_longest
import numpy as np
import pandas as pd
import pytest
from src.config import Config
from src.contacts import extract_contacts
from src.disk_blocking import DiskBlocker
from src.disk_store import fetch_records,connect
from src.features import FeatureEngineer
from src.normalization import normalize_business_name
from src.preprocessing import preprocess
from src.real_metrics import entity_scores
from src.evaluation import evaluate


@pytest.fixture(scope='module')
def real():
    config_path=Path(os.environ.get('BER_TEST_CONFIG', 'config/windows.json' if sys.platform=='win32' and Path('config/windows.json').exists() else 'config/real.json'))
    if not config_path.exists():pytest.skip('Real resource config not configured')
    config=Config.load(str(config_path))
    if not Path(config.dataset_dir).exists():pytest.skip('Real competition files unavailable')
    return config


@pytest.mark.parametrize('split,source',[(split,source) for split in ('train','test') for source in (1,2,3)])
def test_real_source_schema(real,split,source):
    frame=pd.read_csv(Path(real.dataset_dir)/split/f'{split}_source{source}.tsv',sep='\t',dtype=str,keep_default_na=False,nrows=256)
    assert frame.columns.tolist()==['entity_id','business_name','business_address','country']
    assert frame.entity_id.str.startswith(f'S{source}-').all()
    assert frame.entity_id.is_unique
    assert not frame.entity_id.eq('').any()
    normalized=preprocess(frame)
    assert len(normalized)==len(frame)
    assert normalized.entity_id.equals(frame.entity_id)
    assert all(col in normalized for col in frame)


def test_actual_ground_truth_format(real):
    frame=pd.read_csv(Path(real.dataset_dir)/'train/train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False,nrows=256)
    assert frame.columns.tolist()==['source1_entity_id','matched_entity_ids']
    assert frame.source1_entity_id.str.startswith('S1-').all()
    for cell in frame.matched_entity_ids:
        ids=cell.split(',') if cell else []
        assert len(ids)==len(set(ids))
        assert all(x.startswith(('S2-','S3-')) for x in ids)
    assert frame.matched_entity_ids.eq('').any()


def test_real_full_audit_id_integrity(real):
    report=json.loads((Path(real.reports_dir)/'real_data_quality.json').read_text(encoding='utf-8'))
    for name,info in report.items():
        if name.endswith('.tsv'):
            assert info['counts']['rows']==info['counts']['unique_ids']
            assert info['counts']['duplicate_ids']==0
    for source in (1,2,3):assert report[f'relationship_source{source}']['train_test_id_overlap']==0
    manifest=json.loads((Path(real.working_dir)/'train/store_manifest.json').read_text(encoding='utf-8'))
    assert manifest['counts']['positive_pairs']>0
    assert manifest['counts']['duplicate_pairs']==manifest['counts']['conflicting_target_labels']==0


def test_preserves_indic_vowel_marks_and_contacts():
    assert normalize_business_name('राम मार्केटिंग')=='राम मार्केटिंग'
    result=extract_contacts('Info@Example.COM | https://www.example.com/path/ | Tel: +91 (080) 0123-4567')
    assert result['email']=='Info@example.com'
    assert result['website_domain']=='example.com'
    assert result['phone']=='+9108001234567'
    assert extract_contacts('25 Main Rd, Postal 123456')['phone']==''


def test_preprocessing_accepts_null_identifying_fields():
    frame=pd.DataFrame({'entity_id':['S1-null'], 'business_name':[None],
                        'business_address':[np.nan], 'country':['France']})
    result=preprocess(frame)
    assert result.name_norm.iloc[0]==result.address_norm.iloc[0]==''
    assert result.phone.iloc[0]==result.email.iloc[0]==''
    assert result.country_norm.iloc[0]=='france'
    assert pd.isna(result.business_name.iloc[0])


@pytest.fixture(scope='module')
def real_candidates(real):
    blocker=DiskBlocker(real,'train')
    selected=np.load(Path(real.working_dir)/'entity_selection.npz')['calibration'][:32]
    anchors=fetch_records(blocker.con,'anchors',selected)
    pairs,targets,stats,truth=blocker.retrieve(anchors,True)
    yield anchors,pairs,targets,stats,truth
    blocker.close()


def test_actual_candidate_generation_and_recall(real,real_candidates):
    anchors,pairs,targets,stats,truth=real_candidates
    assert not pairs.duplicated(['source1_entity_id','candidate_entity_id']).any()
    assert set(pairs.source1_entity_id)<=set(anchors.entity_id)
    assert pairs.candidate_entity_id.str.startswith(('S2-','S3-')).all()
    expected=sum(target in truth[source] for source,target in zip(pairs.source1_entity_id,pairs.candidate_entity_id))
    assert expected==stats['counts']['retained_true']==pairs.label.sum()
    assert stats['counts']['true_pairs']==sum(map(len,truth.values()))
    assert stats['counts']['candidate_pairs']==len(pairs)
    assert stats['counts']['max_candidates']<=real.max_candidates_per_anchor
    assert expected>0
    # A fixed real calibration slice is a regression sentinel, not the full
    # recall estimate (the full and held-out audits are reported separately).
    assert expected/stats['counts']['true_pairs']>=0.90


def test_actual_missing_address_and_feature_extraction(real,real_candidates):
    anchors,pairs,targets,stats,truth=real_candidates
    con=connect(Path(real.working_dir)/'train/records.sqlite',readonly=True)
    rid=con.execute("SELECT rid FROM targets WHERE business_address='' LIMIT 1").fetchone()[0]
    missing=fetch_records(con,'targets',[rid]);con.close()
    assert preprocess(missing).address_norm.iloc[0]==''
    frame=preprocess(pd.concat([anchors,targets],ignore_index=True))
    engineer=FeatureEngineer(2000).fit(frame)
    features=engineer.transform(pairs,engineer.prepare(frame))
    assert len(features)==len(pairs)
    assert np.isfinite(features.to_numpy()).all()
    assert not any('entity_id' in field or field=='label' for field in features)


def test_grouped_retrieval_features_equal_original_on_real_candidates(real_candidates,monkeypatch):
    from rapidfuzz import process
    import src.coarse_ranker as ranker
    anchors,pairs,targets,stats,truth=real_candidates
    positions={entity:i for i,entity in enumerate(anchors.entity_id)}
    target_positions={entity:i for i,entity in enumerate(targets.entity_id)}
    repeated=np.asarray([positions[x] for x in pairs.source1_entity_id])
    indexer=np.asarray([target_positions[x] for x in pairs.candidate_entity_id])
    text={field:targets[field].to_numpy() for field in ('name_norm','address_norm','country_norm')}
    masks=np.zeros(len(pairs),dtype=np.uint16)
    grouped=ranker.coarse_features(anchors,text,indexer,repeated,masks,similarity_threads=2)
    monkeypatch.setattr(ranker,'grouped_similarities',lambda left,right,boundaries,scorer:
                        process.cpdist(left,right,scorer=scorer,dtype=np.float32,workers=1))
    original=ranker.coarse_features(anchors,text,indexer,repeated,masks)
    np.testing.assert_array_equal(grouped,original)


def test_vectorized_metric_matches_original_singleton_convention():
    truth={'S1-a':{'S2-a','S3-a'},'S1-b':set(),'S1-c':{'S2-c'}}
    pred={'S1-a':{'S2-a'},'S1-b':set(),'S1-c':set()}
    original=evaluate(truth,pred,100)
    vector=entity_scores(np.array([0,0,1]),np.array([1,0,0]),np.array([.9,.1,.2]),np.array([2,0,1]),.5,100)
    for name in ('precision','recall','f0.5','singleton_accuracy','true_positive','false_positive','false_negative'):
        assert vector[name]==pytest.approx(original[name])


def test_supplied_official_validator_accepts_real_subset_singletons(real,tmp_path):
    testdir=tmp_path/'test';testdir.mkdir()
    frames=[]
    for source in (1,2,3):
        frame=pd.read_csv(Path(real.dataset_dir)/'test'/f'test_source{source}.tsv',sep='\t',dtype=str,keep_default_na=False,nrows=10)
        frame.to_csv(testdir/f'test_source{source}.tsv',sep='\t',index=False);frames.append(frame)
    matching=tmp_path/'matching_results.tsv';candidate=tmp_path/'candidate_pairs.tsv'
    for path,column in ((matching,'matched_entity_ids'),(candidate,'candidate_entity_ids')):
        pd.DataFrame({'source1_entity_id':frames[0].entity_id,column:''}).to_csv(path,sep='\t',index=False)
    command=[sys.executable,'-X','utf8',str(Path(real.resource_dir)/'utils/validate_submission.py'),'--matching',str(matching),'--candidate',str(candidate),'--test-dir',str(testdir),'--check-ids']
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0 and 'PASS' in result.stdout
    bad=pd.read_csv(matching,sep='\t',dtype=str,keep_default_na=False)
    bad.loc[0,'matched_entity_ids']=bad.loc[0,'source1_entity_id'];bad.to_csv(matching,sep='\t',index=False)
    assert subprocess.run(command,capture_output=True,text=True,encoding='utf-8').returncode==1


def test_final_real_submission_and_official_result(real):
    report=Path(real.reports_dir)/'official_validator_command.json'
    if not report.exists():pytest.skip('Full real inference and official validation have not completed yet')
    validation=json.loads(report.read_text(encoding='utf-8'));stats=json.loads((Path(real.reports_dir)/'test_inference.json').read_text(encoding='utf-8'))
    assert validation['exit_code']==0
    assert '--check-ids' in validation['command']
    assert stats['counts']['anchors']==json.loads((Path(real.working_dir)/'test/store_manifest.json').read_text(encoding='utf-8'))['counts']['source1']
    assert stats['counts']['scored_pairs']==stats['counts']['candidate_pairs']
    for filename,second in (('matching_results.tsv','matched_entity_ids'),('candidate_pairs.tsv','candidate_entity_ids')):
        path=Path(real.output_dir)/filename
        assert path.stat().st_size>0
        with path.open(encoding="utf-8") as stream:assert stream.readline().strip().split('\t')==['source1_entity_id',second]
    # Streaming equality to unique raw anchors proves coverage, ordering, row count
    # and absence of duplicate output IDs without retaining millions of strings.
    output=Path(real.output_dir)
    with (Path(real.dataset_dir)/'test/test_source1.tsv').open(encoding='utf-8') as raw, \
         (output/'matching_results.tsv').open(encoding='utf-8') as matching, \
         (output/'candidate_pairs.tsv').open(encoding='utf-8') as candidate:
        for stream in (raw,matching,candidate):next(stream)
        rows=matches=candidates=empty=0
        for expected,m,c in zip_longest(raw,matching,candidate):
            assert expected is not None and m is not None and c is not None
            entity=expected.split('\t',1)[0]
            ma=m.rstrip('\n').split('\t');ca=c.rstrip('\n').split('\t')
            assert len(ma)==len(ca)==2 and ma[0]==ca[0]==entity
            mids=ma[1].split(',') if ma[1] else []
            cids=ca[1].split(',') if ca[1] else []
            assert len(mids)==len(set(mids)) and len(cids)==len(set(cids))
            assert set(mids)<=set(cids)
            rows+=1;matches+=len(mids);candidates+=len(cids);empty+=not mids
    assert rows==stats['counts']['anchors']
    assert matches==stats['counts']['predicted_matches']
    assert candidates==stats['counts']['scored_pairs']
    assert empty==stats['counts']['predicted_singletons']
