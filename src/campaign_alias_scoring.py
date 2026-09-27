"""Resumable one-hop alias scoring, following completed main OOF shards."""
import argparse,collections,gc,json,mmap,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .config import Config
from .disk_store import connect,fetch_records,COLUMNS
from .preprocessing import preprocess
from .evidence import enrich
from .context_experiments import context
from .peer_evidence import features as peers
from .detail_evidence import features as details
from .campaign_alias import proposals,name_index
from .campaign_scoring import tasks,predict
from .rescoring import DTYPE,filehash,sourcehash
W={}

def init_worker(config,main_work):
 cfg=Config(**config);W['con']=connect(Path(cfg.working_dir)/'test/records.sqlite',True);W['base']=np.load('work/campaign_0931/test_scores.npy',mmap_mode='r');W['main_work']=Path(main_work)
 z=np.load('work/campaign_0931/test_target_name_index.npz');W['keys']=z['keys'];W['rids']=z['rids'];W['stats']=joblib.load('work/improvements/evidence_stats_test.joblib');W['engineer']=joblib.load(Path(cfg.working_dir)/'feature_engineer.joblib');W['champ']=joblib.load('artifacts/improvements/catboost_d10_evidence.joblib');W['model']=joblib.load('artifacts/campaign_0931/oof_compact_d9.joblib');W['main_start']=None;W['offsets']=np.load(Path(cfg.working_dir)/'test/entity_id_offsets.npy',mmap_mode='r');W['id_file']=(Path(cfg.working_dir)/'test/entity_id.bin').open('rb');W['ids']=mmap.mmap(W['id_file'].fileno(),0,access=mmap.ACCESS_READ)

def score_arrays(base,main):
 if not np.array_equal(base['anchor'],main['anchor']) or not np.array_equal(base['target'],main['target']):raise ValueError('Base/main alignment')
 if not len(base):return np.empty(0,DTYPE)
 con=W['con'];anc=fetch_records(con,'anchors',np.unique(base['anchor']));ai=np.searchsorted(anc.rid.to_numpy(),base['anchor']);offsets=W['offsets'];ids=W['ids'];target_ids=[ids[int(offsets[r-1]):int(offsets[r])-1].decode('utf-8') for r in base['target']]
 old=pd.DataFrame({'anchor_rid':base['anchor'],'target_rid':base['target'],'source1_entity_id':anc.entity_id.to_numpy()[ai],'candidate_entity_id':target_ids});added=proposals(old,main['p'],con,W['keys'],W['rids'])
 if not len(added):return np.empty(0,DTYPE)
 affected=old.anchor_rid.isin(added.anchor_rid.unique());old=old[affected].copy();old['_p']=base['p'][affected];anc=anc[anc.rid.isin(added.anchor_rid.unique())];target=fetch_records(con,'targets',added.target_rid);records=preprocess(pd.concat([anc[COLUMNS[:4]],target[COLUMNS[:4]]],ignore_index=True));engineer=W['engineer'];X=engineer.transform(added,engineer.prepare(records));X=pd.concat([X.set_axis(added.index),enrich(added,con,W['stats'])],axis=1)
 added['_p']=predict(W['champ'],X);added['_new']=True;old['_new']=False;b=pd.concat([old,added]).sort_values('anchor_rid',kind='stable').reset_index(drop=True);q=b['_p'].to_numpy();X=pd.concat([context(b,q),peers(b,q,con),details(b,con,W['stats'])],axis=1);fresh=b['_new'].to_numpy();p=predict(W['model'],X.loc[fresh,W['model']['feature_names']]);rows=b[fresh];result=np.empty(len(rows),DTYPE);result['anchor']=rows.anchor_rid;result['target']=rows.target_rid;result['p']=p;return result

def score_batch(task):
 start,end,lo,hi=task;main_start=start//1000*1000
 if W['main_start']!=main_start:W['main_array']=np.load(W['main_work']/f'{main_start:08d}.npy');W['main_start']=main_start
 a=W['main_array'];a=a[(a['anchor']>start)&(a['anchor']<=end)];return start,end,score_arrays(W['base'][lo:hi],a)

def run(args):
 cfg=Config.load('config/windows.json');root=Path(args.work);root.mkdir(parents=True,exist_ok=True);main=Path(args.main_work);recipe=json.loads(Path('reports/campaign_0931/alias_frozen.json').read_text());confirmation=json.loads(Path('reports/campaign_0931/alias_confirmation_decision.json').read_text());base_sig=json.loads((main/'signature.json').read_text())
 if not confirmation['promotion_supported'] or confirmation['recipe']!=recipe:raise ValueError('Alias rule not confirmed')
 if base_sig['model_sha256']!=recipe['model_sha256'] or base_sig['batch']!=1000:raise ValueError('Unexpected main model or shard size')
 if sourcehash('src/campaign_alias.py')!=recipe['code_sha256'] or filehash('artifacts/campaign_0931/oof_compact_d9.joblib')!=recipe['model_sha256']:raise ValueError('Confirmed alias/model code changed')
 con=connect(Path(cfg.working_dir)/'test/records.sqlite',True);keys,rids=name_index(con,'test');del keys,rids;con.close();gc.collect()
 sig={'main_signature_sha256':filehash(main/'signature.json'),'recipe':recipe,'batch':500,'name_index_sha256':filehash('work/campaign_0931/test_target_name_index.npz'),'base_model_sha256':filehash('artifacts/improvements/catboost_d10_evidence.joblib'),'code':{s:sourcehash(Path('src')/s) for s in ['campaign_alias_scoring.py','campaign_alias.py','campaign_scoring.py','peer_evidence.py','detail_evidence.py','context_experiments.py','evidence.py','features.py','preprocessing.py','normalization.py','contacts.py']}}
 path=root/'signature.json'
 if path.exists() and json.loads(path.read_text())!=sig:raise ValueError('Alias scoring inputs/code changed; preserve old cache and use another work directory')
 path.write_text(json.dumps(sig,indent=2),encoding='utf-8');base=np.load('work/campaign_0931/test_scores.npy',mmap_mode='r');pending=collections.deque();started=time.time();done=0;new_count=0;total=0
 def consume():
  nonlocal done,new_count
  start,end,a=pending.popleft().result();dest=root/f'{start:08d}.npy';tmp=dest.with_suffix('.partial')
  with tmp.open('wb') as f:np.save(f,a)
  tmp.replace(dest);done+=end-start;new_count+=len(a);elapsed=time.time()-started;status={'last_completed_anchor':end,'anchors_processed_this_run':done,'new_pairs_this_run':new_count,'seconds':elapsed,'anchors_per_second':done/elapsed,'complete':False};(root/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
  if done%1000==0:print('ALIAS_TEST_SCORED',json.dumps(status),flush=True)
 with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(cfg.to_dict(),str(main))) as pool:
  for task in tasks(base,1732544,500,args.limit):
   start,end,lo,hi=task;total=end;dest=root/f'{start:08d}.npy'
   if dest.exists():
    a=np.load(dest,mmap_mode='r')
    if a.dtype!=DTYPE or not np.isfinite(a['p']).all() or np.any((a['p']<0)|(a['p']>1)) or np.any((a['anchor']<=start)|(a['anchor']>end)):raise ValueError('Invalid alias shard')
    continue
   expected=main/f'{start//1000*1000:08d}.npy';waiting=time.time()
   while not expected.exists():
    if time.time()-waiting>1800:raise RuntimeError('Main scoring has not produced the next shard for30minutes; resume main scoring first')
    time.sleep(2)
   pending.append(pool.submit(score_batch,task))
   if len(pending)>=args.workers*2:consume()
  while pending:consume()
 if args.limit is None:
  (root/'SCORING_COMPLETE.json').write_text(json.dumps({'signature':sig,'anchors':total,'seconds_this_run':time.time()-started},indent=2),encoding='utf-8');(root/'progress.json').write_text(json.dumps({'complete':True,'last_completed_anchor':total,'anchors_processed_this_run':done,'seconds':time.time()-started},indent=2),encoding='utf-8')
 print('ALIAS_TEST_FINISHED',total,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--work',default='work/campaign_0931/test_alias_v2');p.add_argument('--main-work',default='work/campaign_0931/test_final_oof');run(p.parse_args())
