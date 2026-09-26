"""Export separately named pair-threshold and unique-owner submissions from score shards."""
import argparse,json,mmap,os
from pathlib import Path
import numpy as np
from .config import Config
from .disk_store import connect
from .rescoring import filehash

def finalize(args):
 root=Path(args.work);cfg=Config.load('config/windows.json');lock=json.loads(Path('reports/improvements/frozen_selection.json').read_text(encoding='utf-8'))
 if not (root/'SCORING_COMPLETE.json').exists():raise ValueError('Full scoring is not complete')
 sig=json.loads((root/'signature.json').read_text(encoding='utf-8'))
 if sig['model_sha256']!=lock['sha256']:raise ValueError('Scores do not match frozen model')
 if filehash(args.candidates)!=sig['candidate_sha256']:raise ValueError('Candidate file changed')
 con=connect(Path(cfg.working_dir)/'test/records.sqlite',True)
 n=con.execute('select count(*) from anchors').fetchone()[0];nt=con.execute('select count(*) from targets').fetchone()[0]
 if json.loads((root/'SCORING_COMPLETE.json').read_text(encoding='utf-8'))['anchors']!=n:raise ValueError('Scoring anchor coverage mismatch')
 paths=[root/f'{start:08d}.npy' for start in range(0,n,sig['batch'])]
 if any(not p.exists() for p in paths):raise ValueError('Missing score shard')
 best=np.full(nt+1,-np.inf);ties=np.zeros(nt+1,np.uint32)
 for path in paths:
  a=np.load(path);np.maximum.at(best,a['target'],a['p'])
 for path in paths:
  a=np.load(path);win=a['p']==best[a['target']];np.add.at(ties,a['target'][win],1)
 out=Path(args.output)
 if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty to avoid overwriting a submission')
 out.mkdir(parents=True,exist_ok=True)
 source=Path(args.candidates).resolve();dest=out/'candidate_pairs.tsv';os.link(source,dest)
 packed=Path(cfg.working_dir)/'test';offsets=np.load(packed/'entity_id_offsets.npy',mmap_mode='r')
 idfile=(packed/'entity_id.bin').open('rb');ids=mmap.mmap(idfile.fileno(),0,access=mmap.ACCESS_READ)
 def entity(r):return ids[int(offsets[r-1]):int(offsets[r])-1].decode('utf-8')
 stats={'anchors':0,'pairs':0,'matches':0,'empty':0,'unique_owner':args.unique_owner,'threshold':lock['threshold'],'model_sha256':lock['sha256'],'candidate_sha256':sig['candidate_sha256']}
 seen=np.zeros(nt+1,np.uint32)
 with (out/'matching_results.tsv').open('w',encoding='utf-8',newline='') as f, source.open(encoding='utf-8') as candidates:
  f.write('source1_entity_id\tmatched_entity_ids\n');next(candidates)
  for path in paths:
   start=int(path.stem);a=np.load(path);stats['pairs']+=len(a)
   if not np.isfinite(a['p']).all() or np.any((a['p']<0)|(a['p']>1)):raise ValueError('Invalid probabilities')
   if np.any(np.diff(a['anchor'].astype(np.int64))<0):raise ValueError('Unsorted shard')
   selected=a['p']>=lock['threshold']
   if args.unique_owner:selected&=(a['p']==best[a['target']])&(ties[a['target']]==1)
   chosen=a[selected];np.add.at(seen,chosen['target'],1)
   for rid,anchor in con.execute('select rid,entity_id from anchors where rid>? and rid<=? order by rid',(start,min(start+sig['batch'],n))):
    line=next(candidates).rstrip('\r\n').split('\t');candidate_ids=line[1].split(',') if line[1] else []
    if line[0]!=anchor or len(candidate_ids)!=len(set(candidate_ids)):raise ValueError('Candidate integrity')
    row=a[a['anchor']==rid];row_ids=[entity(int(r)) for r in row['target']]
    if row_ids!=candidate_ids:raise ValueError('Scored candidates differ from submitted candidates')
    matched=[entity(int(r)) for r in chosen['target'][chosen['anchor']==rid]]
    if len(matched)!=len(set(matched)) or not set(matched)<=set(candidate_ids):raise ValueError('Matching integrity')
    f.write(anchor+'\t'+','.join(matched)+'\n');stats['anchors']+=1;stats['matches']+=len(matched);stats['empty']+=not matched
  if next(candidates,None) is not None:raise ValueError('Extra candidate rows')
 if stats['anchors']!=n:raise ValueError('Missing anchors')
 stats['duplicate_target_ids']=int((seen>1).sum());stats['excess_target_assignments']=int(np.maximum(seen.astype(np.int64)-1,0).sum())
 if args.unique_owner and stats['duplicate_target_ids']:raise ValueError('Ownership invariant failed')
 stats['matching_sha256']=filehash(out/'matching_results.tsv');stats['streaming_validation']='PASS: full row order, IDs through trusted index, candidate equality, subset, uniqueness and finite scores'
 (out/'validation.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');print(json.dumps(stats,indent=2))
 ids.close();idfile.close();con.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--output',required=True);p.add_argument('--candidates',default='outputs/real_submission/candidate_pairs.tsv');p.add_argument('--unique-owner',action='store_true');finalize(p.parse_args())
