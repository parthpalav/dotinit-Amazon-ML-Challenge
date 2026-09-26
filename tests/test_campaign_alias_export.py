import argparse,json,sqlite3
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src import campaign_alias_export as export
from src.rescoring import DTYPE,filehash

def test_alias_export_preserves_owners_abstains_ties_and_expands_pool(tmp_path,monkeypatch):
 store=tmp_path/'store/test';store.mkdir(parents=True);con=sqlite3.connect(store/'records.sqlite');con.executescript('create table anchors(rid integer,entity_id text);create table targets(rid integer,entity_id text);');con.executemany('insert into anchors values(?,?)',[(1,'S1-1'),(2,'S1-2'),(3,'S1-3')]);con.executemany('insert into targets values(?,?)',[(1,'S2-1'),(2,'S2-2'),(3,'S3-3'),(4,'S3-4')]);con.commit();con.close();(store/'entity_id.bin').write_bytes(b'S2-1\0S2-2\0S3-3\0S3-4\0');np.save(store/'entity_id_offsets.npy',np.array([0,5,10,15,20]))
 monkeypatch.setattr(export.Config,'load',lambda _:SimpleNamespace(working_dir=str(tmp_path/'store')))
 main=tmp_path/'main';alias=tmp_path/'alias';base=tmp_path/'base';out=tmp_path/'out'
 for p in [main,alias,base]:p.mkdir()
 matching='source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\tS2-2\nS1-3\t\n';(base/'matching_results.tsv').write_text(matching);(base/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\nS1-2\tS2-2\nS1-3\t\n')
 cs=filehash(base/'candidate_pairs.tsv');ms=filehash(base/'matching_results.tsv');(base/'validation.json').write_text(json.dumps({'anchors':3,'pairs':2,'matches':2,'duplicate_target_ids':0,'model_sha256':'frozen','matching_sha256':ms,'candidate_sha256':cs}));(main/'signature.json').write_text(json.dumps({'batch':1000,'model_sha256':'frozen','candidate_sha256':cs}));(alias/'signature.json').write_text(json.dumps({'batch':100,'main_signature_sha256':filehash(main/'signature.json'),'recipe':{'model_sha256':'frozen','existing_pair_threshold':.6,'new_pair_threshold':.975}}))
 for p in [main,alias]:(p/'SCORING_COMPLETE.json').write_text(json.dumps({'anchors':3}))
 np.save(main/'00000000.npy',np.array([(1,1,.9),(2,2,.8)],dtype=DTYPE));np.save(alias/'00000000.npy',np.array([(1,2,.999),(1,3,.99),(2,3,.99),(3,4,.99)],dtype=DTYPE))
 export.finalize(argparse.Namespace(main_work=str(main),work=str(alias),base_output=str(base),output=str(out)))
 assert (base/'matching_results.tsv').read_text()==matching
 assert (out/'matching_results.tsv').read_text()=='source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\tS2-2\nS1-3\tS3-4\n'
 assert (out/'candidate_pairs.tsv').read_text()=='source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S2-2,S3-3\nS1-2\tS2-2,S3-3\nS1-3\tS3-4\n'
 report=json.loads((out/'validation.json').read_text());assert report['pairs']==6;assert report['matches']==3;assert report['new_matches']==1;assert report['skipped_alias_assignments_to_owned_targets']==1;assert report['duplicate_target_ids']==0
