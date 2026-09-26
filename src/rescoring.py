"""Rescore existing candidate TSVs without rerunning the expensive ranker.

Checkpointed, bounded-memory scoring. Original submissions are never overwritten.
Run only after selecting a model using training-derived validation.
"""
import argparse,collections,hashlib,itertools,json,os,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import joblib,numpy as np,pandas as pd
from .config import Config
from .disk_store import connect,fetch_records,NativeIndex,COLUMNS
from .disk_blocking import rules_text
from .preprocessing import preprocess
from .evidence import make_stats,enrich

DTYPE=np.dtype([('anchor','<u4'),('target','<u4'),('p','<f8')]);W={}

def filehash(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()

def sourcehash(path):
 return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()

def numeric_id(text):
 source,suffix=text.split('-',1);n=int(suffix)
 if source not in ('S2','S3') or not 0<=n<2**32:raise ValueError('Unsupported target ID syntax')
 return n+(2**32 if source=='S3' else 0)

def lookup_cache(cfg):
 root=Path('work/improvements');keys=root/'test_target_keys.npy';rids=root/'test_target_rids.npy'
 if keys.exists() and rids.exists():return
 con=connect(Path(cfg.working_dir)/'test/records.sqlite',True)
 n=con.execute('select count(*) from targets').fetchone()[0]
 k=np.empty(n,np.uint64);r=np.empty(n,np.uint32)
 for i,(rid,entity) in enumerate(con.execute('select rid,entity_id from targets order by rid')):k[i]=numeric_id(entity);r[i]=rid
 order=np.argsort(k)
 for path,value in ((keys,k[order]),(rids,r[order])):
  tmp=path.with_suffix('.partial')
  with tmp.open('wb') as f:np.save(f,value)
  tmp.replace(path)
 con.close()

def init_worker(config,model,direct=False):
 cfg=Config(**config);W['cfg']=cfg;W['con']=connect(Path(cfg.working_dir)/'test/records.sqlite',True)
 W['model']=joblib.load(model);W['engineer']=joblib.load(Path(cfg.working_dir)/'feature_engineer.joblib')
 W['keys']=np.load('work/improvements/test_target_keys.npy',mmap_mode='r');W['rids']=np.load('work/improvements/test_target_rids.npy',mmap_mode='r')
 if direct:
  from .provenance import PairProvenance
  W['provenance']=PairProvenance(Path(cfg.working_dir)/'native',joblib.load('work/improvements/test_posting_counts.joblib'),cfg.retrieval_posting_limit)
 else:
  W['base']=NativeIndex(Path(cfg.working_dir)/'native',Path(cfg.working_dir)/'test/blocking_index.bin')
  W['extra']=NativeIndex(Path(cfg.working_dir)/'native',Path(cfg.working_dir)/'test/supplement_index.bin',True)
 W['stats']=joblib.load('work/improvements/evidence_stats_test.joblib') if any(x.endswith('_v2') for x in W['model']['feature_names']) else None

def score_batch(batch):
 start,rows=batch;cfg=W['cfg'];anc=fetch_records(W['con'],'anchors',range(start+1,start+len(rows)+1))
 if anc.entity_id.tolist()!=[x[0] for x in rows]:raise ValueError('Candidate anchor order differs from test data')
 ids=[t for _,ts in rows for t in ts];counts=np.array([len(ts) for _,ts in rows]);repeat=np.repeat(np.arange(len(rows)),counts)
 keys=np.fromiter(map(numeric_id,ids),dtype=np.uint64,count=len(ids));positions=np.searchsorted(W['keys'],keys)
 if np.any(positions>=len(W['keys'])) or not np.array_equal(W['keys'][positions],keys):raise ValueError('Unknown candidate ID')
 rids=W['rids'][positions];masks=np.zeros(len(ids),np.uint16);offsets=np.r_[0,np.cumsum(counts)]
 targets=fetch_records(W['con'],'targets',rids)
 if 'provenance' in W:
  masks=W['provenance'].masks(anc,targets,repeat,np.searchsorted(targets.rid.to_numpy(),rids))
 else:
  for native in (W['base'],W['extra']):
   rr,mm,oo=native.lookup(native.keys(anc),cfg.retrieval_posting_limit)
   for i in range(len(rows)):
    lo,hi=offsets[i:i+2];raw=rr[oo[i]:oo[i+1]];p=np.searchsorted(raw,rids[lo:hi]);valid=p<len(raw)
    valid[valid]&=raw[p[valid]]==rids[lo:hi][valid]
    ix=np.flatnonzero(valid);masks[lo+ix]|=mm[oo[i]+p[ix]]
 if len(ids) and not np.all(masks):raise ValueError('Existing candidate no longer retrievable; stale index/config')
 pairs=pd.DataFrame({'anchor_rid':anc.rid.to_numpy()[repeat],'target_rid':rids,'source1_entity_id':anc.entity_id.to_numpy()[repeat],
   'candidate_entity_id':ids,'blocking_rules':[rules_text(x) for x in masks]})
 result=np.empty(len(ids),DTYPE);result['anchor']=start+repeat+1;result['target']=rids
 if len(ids):
  records=preprocess(pd.concat([anc[COLUMNS[:4]],targets[COLUMNS[:4]]],ignore_index=True))
  f=W['engineer'].transform(pairs,W['engineer'].prepare(records))
  if W['stats'] is not None:f=pd.concat([f,enrich(pairs,W['con'],W['stats'])],axis=1)
  if f.columns.tolist()!=W['model']['feature_names']:raise ValueError('Model feature schema mismatch')
  result['p']=W['model']['matcher'].predict(f)
 return start,len(rows),result

def batches(path,size,limit):
 with open(path,encoding='utf-8') as f:
  if next(f).rstrip('\n')!='source1_entity_id\tcandidate_entity_ids':raise ValueError('Candidate schema')
  start=0
  while True:
   rows=[]
   for _ in range(size):
    if limit is not None and start+len(rows)>=limit:break
    line=next(f,None)
    if line is None:break
    source,targets=line.rstrip('\r\n').split('\t')
    rows.append((source,targets.split(',') if targets else []))
   if not rows:return
   yield start,rows;start+=len(rows)

def score(args):
 cfg=Config.load('config/windows.json');root=Path(args.work);root.mkdir(parents=True,exist_ok=True)
 if args.limit is None:
  confirmation=json.loads(Path('reports/improvements/confirmation_evaluation.json').read_text(encoding='utf-8'))
  if confirmation['frozen_selection']['sha256']!=filehash(args.model) or confirmation['paired_bootstrap_95pct_ci'][0]<=0:
   raise ValueError('Full inference requires confirmed improvement from this exact model')
 lookup_cache(cfg)
 if args.direct_provenance:
  from .provenance import prepare_counts
  prepare_counts(cfg)
 model=joblib.load(args.model)
 if any(x.endswith('_v2') for x in model['feature_names']):make_stats(cfg,'test')
 signature={'model_sha256':filehash(args.model),'candidate_sha256':filehash(args.candidates),'batch':args.batch,'direct_provenance':args.direct_provenance,
  'code':{name:sourcehash(Path('src')/name) for name in ['rescoring.py','evidence.py','features.py','normalization.py','preprocessing.py','contacts.py','provenance.py','native/provenance.cpp']}}
 manifest=root/'signature.json'
 if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=signature:raise ValueError('Score cache signature changed; use fresh --work')
 manifest.write_text(json.dumps(signature,indent=2),encoding='utf-8')
 started=time.time();pending=collections.deque();done=0;total=0
 def consume():
  nonlocal done
  start,n,array=pending.popleft().result();path=root/f'{start:08d}.npy';tmp=path.with_suffix('.partial')
  with tmp.open('wb') as f:np.save(f,array)
  tmp.replace(path);done+=n
  status={'last_completed_anchor':start+n,'anchors_processed_this_run':done,'seconds':time.time()-started,'model':args.model,'scope':'benchmark' if args.limit else 'full scoring','complete':False}
  (root/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
  if done%1000==0:print('SCORED',json.dumps(status),flush=True)
 with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(cfg.to_dict(),args.model,args.direct_provenance)) as pool:
  for batch in batches(args.candidates,args.batch,args.limit):
   total+=len(batch[1])
   shard=root/f'{batch[0]:08d}.npy'
   if shard.exists():
    cached=np.load(shard,mmap_mode='r')
    if cached.dtype!=DTYPE or len(cached)!=sum(len(row[1]) for row in batch[1]) or not np.isfinite(cached['p']).all():raise ValueError('Invalid cached score shard')
    continue
   pending.append(pool.submit(score_batch,batch))
   if len(pending)>=args.workers*2:consume()
  while pending:consume()
 print('SCORING FINISHED',done,time.time()-started,flush=True)
 if args.limit is None:(root/'SCORING_COMPLETE.json').write_text(json.dumps({'signature':signature,'seconds_this_run':time.time()-started,'anchors':total}),encoding='utf-8')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--work',required=True)
 p.add_argument('--direct-provenance',action='store_true')
 p.add_argument('--candidates',default='outputs/real_submission/candidate_pairs.tsv');p.add_argument('--batch',type=int,default=100);p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int)
 score(p.parse_args())
