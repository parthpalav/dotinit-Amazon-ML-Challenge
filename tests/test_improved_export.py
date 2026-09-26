"""Submission export must preserve empty anchors and resolve exact-score ties safely."""
import argparse,json,sqlite3
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src import finalize_improved
from src.rescoring import DTYPE,filehash

@pytest.mark.parametrize('unique,expected',[(False,'S1-1\tS2-1,S2-2\nS1-2\tS2-1\nS1-3\t\n'),(True,'S1-1\tS2-2\nS1-2\t\nS1-3\t\n')])
def test_export_ties_empty_and_candidate_alignment(tmp_path,monkeypatch,unique,expected):
 monkeypatch.chdir(tmp_path)
 store=tmp_path/'store/test';store.mkdir(parents=True)
 con=sqlite3.connect(store/'records.sqlite');con.executescript('create table anchors(rid integer,entity_id text);create table targets(rid integer,entity_id text);')
 con.executemany('insert into anchors values(?,?)',[(1,'S1-1'),(2,'S1-2'),(3,'S1-3')]);con.executemany('insert into targets values(?,?)',[(1,'S2-1'),(2,'S2-2')]);con.commit();con.close()
 (store/'entity_id.bin').write_bytes(b'S2-1\0S2-2\0');np.save(store/'entity_id_offsets.npy',np.array([0,5,10]))
 monkeypatch.setattr(finalize_improved.Config,'load',lambda _:SimpleNamespace(working_dir=str(tmp_path/'store')))
 report=tmp_path/'reports/improvements';report.mkdir(parents=True)
 (report/'frozen_selection.json').write_text(json.dumps({'sha256':'frozen','threshold':.675}))
 candidate=tmp_path/'candidates.tsv';candidate.write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S2-2\nS1-2\tS2-1\nS1-3\t\n')
 root=tmp_path/'scores';root.mkdir();(root/'signature.json').write_text(json.dumps({'model_sha256':'frozen','candidate_sha256':filehash(candidate),'batch':100}))
 (root/'SCORING_COMPLETE.json').write_text(json.dumps({'anchors':3}));np.save(root/'00000000.npy',np.array([(1,1,.9),(1,2,.8),(2,1,.9)],dtype=DTYPE))
 out=tmp_path/'submission'
 finalize_improved.finalize(argparse.Namespace(work=str(root),candidates=str(candidate),output=str(out),unique_owner=unique))
 assert (out/'matching_results.tsv').read_text()=='source1_entity_id\tmatched_entity_ids\n'+expected
 report=json.loads((out/'validation.json').read_text());assert report['anchors']==3
 assert report['duplicate_target_ids']==(0 if unique else 1)
