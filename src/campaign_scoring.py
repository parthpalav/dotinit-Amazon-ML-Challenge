"""Resume compact contextual reranking directly from the verified champion scores.
The original matcher/retrieval is not rerun. Complete anchors form each shard.
"""
import argparse,collections,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .config import Config
from .disk_store import connect,fetch_records
from .context_experiments import context
from .peer_evidence import features as peers
from .detail_evidence import features as details
from .rescoring import DTYPE,filehash,sourcehash
W={}

def init_worker(config,model,scores):
 cfg=Config(**config);W['con']=connect(Path(cfg.working_dir)/'test/records.sqlite',True)
 W['model']=joblib.load(model);W['scores']=np.load(scores,mmap_mode='r');W['stats']=joblib.load('work/improvements/evidence_stats_test.joblib')

def predict(model,X):
 if list(X.columns)!=model['feature_names']:raise ValueError('Context feature schema mismatch')
  matcher=model['matcher']
 if hasattr(matcher,'members'):
  return sum(w*predict({'matcher':m,'feature_names':model['feature_names']},X) for w,m in zip(matcher.weights,matcher.members))
 p=matcher.estimator.predict_proba(X,thread_count=1)[:,1]
 if matcher.calibrator is not None:p=matcher.calibrator.predict_proba(matcher.logits(p))[:,1]
 if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid contextual probability')
 return p

def score_batch(task):
 start,end,lo,hi=task;b=W['scores'][lo:hi];con=W['con']
 if not len(b):return start,end,np.empty(0,DTYPE)
 anc=fetch_records(con,'anchors',range(start+1,end+1));target=fetch_records(con,'targets',b['target'])
 ai=np.searchsorted(anc.rid.to_numpy(),b['anchor']);ti=np.searchsorted(target.rid.to_numpy(),b['target'])
 if not np.array_equal(anc.rid.to_numpy()[ai],b['anchor']) or not np.array_equal(target.rid.to_numpy()[ti],b['target']):raise ValueError('Missing record')
 pairs=pd.DataFrame({'anchor_rid':b['anchor'],'target_rid':b['target'],'source1_entity_id':anc.entity_id.to_numpy()[ai],'candidate_entity_id':target.entity_id.to_numpy()[ti]})
 X=pd.concat([context(pairs,b['p']),peers(pairs,b['p'],con),details(pairs,con,W['stats'])],axis=1)
 result=np.array(b,copy=True);result['p']=predict(W['model'],X);return start,end,result

def tasks(scores,n,batch,limit=None):
 # Count once: repeated searchsorted on a strided structured field can copy
 # the entire 55-million-row column for every shard on NumPy implementations.
 counts=np.zeros(n+1,np.int64);previous=0
 for lo in range(0,len(scores),1000000):
  a=scores['anchor'][lo:lo+1000000]
  if len(a) and (int(a[0])<previous or int(a[0])<1 or int(a[-1])>n or np.any(a[1:]<a[:-1])):raise ValueError('Unsorted or invalid anchor IDs')
  if len(a):previous=int(a[-1]);counts+=np.bincount(a,minlength=n+1)
 offsets=np.r_[0,np.cumsum(counts[1:])];n=min(n,limit) if limit else n
 for start in range(0,n,batch):
  end=min(start+batch,n);yield start,end,int(offsets[start]),int(offsets[end])

def run(args):
 cfg=Config.load('config/windows.json');root=Path(args.work);root.mkdir(parents=True,exist_ok=True)
 frozen=json.loads(Path(args.frozen).read_text());recipe=next(r for r in frozen['models'] if r['name']==args.model_name)
 if filehash(recipe['path'])!=recipe['sha256']:raise ValueError('Frozen model changed')
 if args.limit is None:
  results=json.loads(Path(args.confirmation).read_text())
  if results[args.model_name]['paired_95ci'][0]<=0:raise ValueError('Fresh confirmation did not establish improvement')
 reference=json.loads(Path('work/improvements/test_scores_d10_direct/signature.json').read_text())
 if filehash(args.candidates)!=reference['candidate_sha256']:raise ValueError('Candidate pool changed')
 if filehash('artifacts/improvements/catboost_d10_evidence.joblib')!=reference['model_sha256']:raise ValueError('Champion changed')
 scores=np.load(args.scores,mmap_mode='r');info=json.loads(Path('work/campaign_0931/score_import.json').read_text())
 if scores.dtype!=DTYPE or len(scores)!=info['pairs'] or info['next_anchor']!=1732544:raise ValueError('Incomplete base scores')
 sig={'model_sha256':recipe['sha256'],'candidate_sha256':reference['candidate_sha256'],'base_scores_sha256':filehash(args.scores),'batch':args.batch,
      'store_manifest_sha256':filehash(Path(cfg.working_dir)/'test/store_manifest.json'),
      'code':{s:sourcehash(Path('src')/s) for s in ['campaign_scoring.py','campaign_ensemble.py','context_experiments.py','peer_evidence.py','detail_evidence.py','evidence.py']}}
 manifest=root/'signature.json'
 if manifest.exists() and json.loads(manifest.read_text())!=sig:raise ValueError('Scoring signature changed; use another work directory')
 manifest.write_text(json.dumps(sig,indent=2),encoding='utf-8')
 (root/'selection.json').write_text(json.dumps({'sha256':recipe['sha256'],'threshold':recipe['threshold'],'base_model_sha256':reference['model_sha256']},indent=2),encoding='utf-8')
 started=time.time();done=0;pending=collections.deque();total=0
 def consume():
  nonlocal done
  start,end,a=pending.popleft().result();path=root/f'{start:08d}.npy';tmp=path.with_suffix('.partial')
  with tmp.open('wb') as f:np.save(f,a)
  tmp.replace(path);done+=end-start;elapsed=time.time()-started
  progress={'last_completed_anchor':end,'anchors_processed_this_run':done,'seconds':elapsed,'anchors_per_second':done/elapsed,'estimated_remaining_seconds':max(0,1732544-end)/(done/elapsed),'complete':False}
  (root/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8');print('CONTEXT_SCORED',json.dumps(progress),flush=True)
 with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(cfg.to_dict(),recipe['path'],args.scores)) as pool:
  for task in tasks(scores,1732544,args.batch,args.limit):
   start,end,lo,hi=task;total=end;path=root/f'{start:08d}.npy'
   if path.exists():
    a=np.load(path,mmap_mode='r');b=scores[lo:hi]
    if a.dtype!=DTYPE or len(a)!=hi-lo or not np.array_equal(a['anchor'],b['anchor']) or not np.array_equal(a['target'],b['target']) or not np.isfinite(a['p']).all() or np.any((a['p']<0)|(a['p']>1)):raise ValueError('Invalid cached shard')
    continue
   pending.append(pool.submit(score_batch,task))
   if len(pending)>=args.workers*2:consume()
  while pending:consume()
 if args.limit is None:(root/'SCORING_COMPLETE.json').write_text(json.dumps({'signature':sig,'anchors':total,'seconds_this_run':time.time()-started},indent=2),encoding='utf-8')
 print('CONTEXT_SCORING_FINISHED',total,time.time()-started,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--frozen',default='reports/campaign_0931/frozen_confirmation_v3.json');p.add_argument('--confirmation',default='reports/campaign_0931/confirmation_v3_results.json');p.add_argument('--model-name',default='compact');p.add_argument('--workers',type=int,default=2);p.add_argument('--batch',type=int,default=1000);p.add_argument('--limit',type=int);p.add_argument('--work',default='work/campaign_0931/test_compact');p.add_argument('--scores',default='work/campaign_0931/test_scores.npy');p.add_argument('--candidates',default='outputs/real_submission/candidate_pairs.tsv');run(p.parse_args())
