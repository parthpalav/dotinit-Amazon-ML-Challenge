"""Export aliases as additions only, preserving every accepted main owner."""
import argparse,json,mmap
from pathlib import Path
import numpy as np
from .config import Config
from .disk_store import connect
from .rescoring import DTYPE,filehash

def finalize(args):
 cfg=Config.load('config/windows.json');main=Path(args.main_work);alias=Path(args.work);base_out=Path(args.base_output);out=Path(args.output);con=connect(Path(cfg.working_dir)/'test/records.sqlite',True);n=con.execute('select count(*) from anchors').fetchone()[0];nt=con.execute('select count(*) from targets').fetchone()[0]
 main_sig=json.loads((main/'signature.json').read_text());sig=json.loads((alias/'signature.json').read_text());recipe=sig['recipe'];base_report=json.loads((base_out/'validation.json').read_text())
 if sig['main_signature_sha256']!=filehash(main/'signature.json') or main_sig['model_sha256']!=recipe['model_sha256'] or base_report['model_sha256']!=recipe['model_sha256']:raise ValueError('Main model/signature mismatch')
 for root in [main,alias]:
  if json.loads((root/'SCORING_COMPLETE.json').read_text())['anchors']!=n:raise ValueError('Incomplete scoring coverage')
 if base_report['anchors']!=n or base_report['duplicate_target_ids']!=0:raise ValueError('Main export not validated as unique-owner')
 if filehash(base_out/'matching_results.tsv')!=base_report['matching_sha256'] or filehash(base_out/'candidate_pairs.tsv')!=main_sig['candidate_sha256']:raise ValueError('Main export changed')
 if out.exists() and any(out.iterdir()):raise ValueError('Output must be empty')
 main_paths=[main/f'{start:08d}.npy' for start in range(0,n,main_sig['batch'])];alias_paths=[alias/f'{start:08d}.npy' for start in range(0,n,sig['batch'])]
 if any(not p.exists() for p in main_paths+alias_paths):raise ValueError('Missing score shard')
 best=np.zeros(nt+1);ties=np.zeros(nt+1,np.uint32)
 for path in main_paths:
  a=np.load(path);np.maximum.at(best,a['target'],a['p'])
 for path in main_paths:
  a=np.load(path);win=a['p']==best[a['target']];np.add.at(ties,a['target'][win],1)
 owned=(best>=recipe['existing_pair_threshold'])&(ties==1)
 if int(owned.sum())!=base_report['matches']:raise ValueError('Main ownership disagrees with validated export')
 del best,ties
 best=np.zeros(nt+1);ties=np.zeros(nt+1,np.uint32);new_pairs=0;skipped_owned=0
 for path in alias_paths:
  a=np.load(path);start=int(path.stem)
  if a.dtype!=DTYPE or not np.isfinite(a['p']).all() or np.any((a['p']<0)|(a['p']>1)) or np.any((a['target']<1)|(a['target']>nt)) or np.any((a['anchor']<=start)|(a['anchor']>min(start+sig['batch'],n))):raise ValueError('Invalid alias shard')
  new_pairs+=len(a);eligible=~owned[a['target']];skipped_owned+=int((owned[a['target']]&(a['p']>=recipe['new_pair_threshold'])).sum());np.maximum.at(best,a['target'][eligible],a['p'][eligible])
 for path in alias_paths:
  a=np.load(path);win=~owned[a['target']]&(a['p']==best[a['target']]);np.add.at(ties,a['target'][win],1)
 out.mkdir(parents=True,exist_ok=True);packed=Path(cfg.working_dir)/'test';offsets=np.load(packed/'entity_id_offsets.npy',mmap_mode='r');idfile=(packed/'entity_id.bin').open('rb');ids=mmap.mmap(idfile.fileno(),0,access=mmap.ACCESS_READ)
 def entity(r):return ids[int(offsets[r-1]):int(offsets[r])-1].decode('utf-8')
 stats={'anchors':0,'pairs':0,'matches':0,'empty':0,'new_candidates':new_pairs,'new_matches':0,'baseline_matches':0,'skipped_alias_assignments_to_owned_targets':skipped_owned,'unique_owner':True,'existing_threshold':recipe['existing_pair_threshold'],'new_pair_threshold':recipe['new_pair_threshold'],'model_sha256':recipe['model_sha256'],'base_matching_sha256':base_report['matching_sha256'],'preserves_every_existing_match':True};seen=np.zeros(nt+1,np.uint8)
 with (base_out/'candidate_pairs.tsv').open(encoding='utf-8') as oldcand,(base_out/'matching_results.tsv').open(encoding='utf-8') as oldmatch,(out/'candidate_pairs.tsv').open('w',encoding='utf-8',newline='') as candidates,(out/'matching_results.tsv').open('w',encoding='utf-8',newline='') as matching:
  if next(oldcand).rstrip('\r\n')!='source1_entity_id\tcandidate_entity_ids' or next(oldmatch).rstrip('\r\n')!='source1_entity_id\tmatched_entity_ids':raise ValueError('Base schema')
  candidates.write('source1_entity_id\tcandidate_entity_ids\n');matching.write('source1_entity_id\tmatched_entity_ids\n')
  anchor_iter=iter(con.execute('select rid,entity_id from anchors order by rid'))
  for path in alias_paths:
   a=np.load(path);start=int(path.stem);selected=(a['p']>=recipe['new_pair_threshold'])&~owned[a['target']]&(a['p']==best[a['target']])&(ties[a['target']]==1);chosen=a[selected];np.add.at(seen,chosen['target'],1)
   for rid in range(start+1,min(start+sig['batch'],n)+1):
    got,source=next(anchor_iter);c=next(oldcand).rstrip('\r\n').split('\t');m=next(oldmatch).rstrip('\r\n').split('\t')
    if got!=rid or c[0]!=source or m[0]!=source:raise ValueError('Anchor alignment')
    before_c=c[1].split(',') if c[1] else [];before_m=m[1].split(',') if m[1] else [];new_c=[entity(int(r)) for r in a['target'][a['anchor']==rid]];new_m=[entity(int(r)) for r in chosen['target'][chosen['anchor']==rid]];pool=before_c+new_c;matches=before_m+new_m
    if len(pool)!=len(set(pool)) or len(matches)!=len(set(matches)) or not set(matches)<=set(pool):raise ValueError('Expanded candidate/subset integrity')
    candidates.write(source+'\t'+','.join(pool)+'\n');matching.write(source+'\t'+','.join(matches)+'\n');stats['anchors']+=1;stats['pairs']+=len(pool);stats['baseline_matches']+=len(before_m);stats['new_matches']+=len(new_m);stats['matches']+=len(matches);stats['empty']+=not matches
   if start%100000==0:print('ALIAS_EXPORTED',start,flush=True)
  if next(oldcand,None) is not None or next(oldmatch,None) is not None or next(anchor_iter,None) is not None:raise ValueError('Extra base rows')
 if stats['anchors']!=n or stats['baseline_matches']!=base_report['matches'] or stats['pairs']!=base_report['pairs']+new_pairs or np.any(seen>1) or np.any(owned&(seen>0)):raise ValueError('Expanded ownership/coverage invariant')
 stats['duplicate_target_ids']=0;stats['excess_target_assignments']=0;stats['matching_sha256']=filehash(out/'matching_results.tsv');stats['candidate_sha256']=filehash(out/'candidate_pairs.tsv');stats['streaming_validation']='PASS: all anchors; all scored added candidates exported; candidate uniqueness; valid added IDs; matching subset; original matches preserved; global unique ownership';(out/'validation.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');print(json.dumps(stats,indent=2));ids.close();idfile.close();con.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--main-work',default='work/campaign_0931/test_final_oof');p.add_argument('--work',default='work/campaign_0931/test_alias_v2');p.add_argument('--base-output',default='outputs/campaign_0931_oof_unique');p.add_argument('--output',default='outputs/campaign_0931_oof_alias_unique');finalize(p.parse_args())
