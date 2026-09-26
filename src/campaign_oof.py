"""Cross-fitted base probabilities let the context model learn from 30k more anchors.
Never train stacking features on in-sample base-model predictions. Each anchor's
OOF probability comes from a base model fitted without that anchor's labels.
"""
from pathlib import Path
import argparse,gc,hashlib,json,time
import joblib,numpy as np,pandas as pd
from catboost import CatBoostClassifier
from .config import Config
from .disk_store import connect
from .evidence import make_stats
from .context_experiments import context,load
from .peer_evidence import features as peers
from .detail_evidence import features as details
from .model import CalibratedMatcher
from .experiments import evaluate,choose
ROOT=Path('work/campaign_0931');OUT=Path('reports/campaign_0931');ART=Path('artifacts/campaign_0931')

def base(split):
 part=joblib.load(Path('work/real_v1/features')/(split+'.joblib'));extra=joblib.load(Path('work/improvements')/(split+'_extra.joblib'))
 assert extra['pair_ids'].equals(part['pairs'][['source1_entity_id','candidate_entity_id']])
 part['features']=pd.concat([part['features'],extra['features']],axis=1);return part

def oof():
 part=base('fit');cal=base('calibration');pairs=part['pairs'];anchors=pairs.anchor_rid.unique().copy();rng=np.random.default_rng(20260928);rng.shuffle(anchors);folds=np.array_split(anchors,3);p=np.zeros(len(pairs))
 for k,ids in enumerate(folds):
  mask=pairs.anchor_rid.isin(ids).to_numpy();path=ROOT/f'oof_fold{k}.npz'
  if path.exists():
   saved=np.load(path);assert np.array_equal(saved['rows'],np.flatnonzero(mask));p[mask]=saved['p'];continue
  started=time.time();model=CatBoostClassifier(iterations=2306,depth=10,learning_rate=.05,l2_leaf_reg=5,loss_function='Logloss',task_type='GPU',random_seed=42,thread_count=3,allow_writing_files=False,verbose=300)
  model.fit(part['features'][~mask],pairs.label[~mask]);matcher=CalibratedMatcher(model);matcher.calibrate(cal['features'],cal['pairs'].label);p[mask]=matcher.predict(part['features'][mask]);np.savez(path,rows=np.flatnonzero(mask),p=p[mask]);joblib.dump({'matcher':matcher,'excluded_anchors':ids},ART/f'oof_base{k}.joblib');print('OOF_FOLD',k,time.time()-started,flush=True);del matcher,model;gc.collect()
 np.save(ROOT/'fit_oof_p.npy',p);print('OOF_COMPLETE',len(p),flush=True)

def prepare():
 part=joblib.load('work/real_v1/features/fit.joblib');pairs=part['pairs'];del part;gc.collect();p=np.load(ROOT/'fit_oof_p.npy');cfg=Config.load('config/windows.json');con=connect(Path(cfg.working_dir)/'train/records.sqlite',True);stats=make_stats(cfg,'train')
 anchors=pairs.anchor_rid.to_numpy();starts=np.r_[0,np.flatnonzero(anchors[1:]!=anchors[:-1])+1,len(anchors)];root=ROOT/'oof_meta_parts';root.mkdir(exist_ok=True);paths=[];started=time.time()
 for g in range(0,len(starts)-1,500):
  path=root/f'{g:05d}.joblib';paths.append(path)
  if path.exists():continue
  lo=starts[g];hi=starts[min(g+500,len(starts)-1)];batch=pairs.iloc[lo:hi];q=p[lo:hi];X=pd.concat([context(batch,q),peers(batch,q,con),details(batch,con,stats)],axis=1);tmp=path.with_suffix('.partial');joblib.dump(X,tmp);tmp.replace(path);print('OOF_META',g,round(time.time()-started,1),flush=True)
 X=pd.concat([joblib.load(p) for p in paths]);assert X.index.equals(pairs.index);joblib.dump(X,ROOT/'oof_meta_fit.joblib');con.close()

def train():
 X=joblib.load(ROOT/'oof_meta_fit.joblib');part=joblib.load('work/real_v1/features/fit.joblib');y=part['pairs'].label;del part;gc.collect()
 cal=joblib.load('work/real_v1/features/calibration.joblib');cp=np.load(ROOT/'calibration_champion_p.npy');cx=pd.concat([context(cal['pairs'],cp),joblib.load(ROOT/'peer_calibration.joblib')['features'],joblib.load(ROOT/'detail_calibration.joblib')['features']],axis=1)
 val=joblib.load('work/real_v1/features/validation.joblib');vp=np.load(ROOT/'validation_champion_p.npy');vx=pd.concat([context(val['pairs'],vp),joblib.load(ROOT/'peer_validation.joblib')['features'],joblib.load(ROOT/'detail_validation.joblib')['features']],axis=1)
 del cal['features'],val['features'];anchors=np.asarray(cal['pairs'].source1_entity_id.unique(),dtype=object).copy();rng=np.random.default_rng(20260926);rng.shuffle(anchors);held=set(anchors[:1000]);fit=~cal['pairs'].source1_entity_id.isin(held)
 assert X.columns.equals(cx.columns) and X.columns.equals(vx.columns)
 X=pd.concat([X,cx[fit]],ignore_index=True);y=pd.concat([y,cal['pairs'].label[fit]],ignore_index=True);report=[]
 for depth in [5,7,9]:
  started=time.time();model=CatBoostClassifier(iterations=2400,depth=depth,learning_rate=.04,l2_leaf_reg=8,loss_function='Logloss',task_type='GPU',random_seed=43,thread_count=3,allow_writing_files=False,verbose=300)
  model.fit(X,y,eval_set=(cx[~fit],cal['pairs'].label[~fit]),early_stopping_rounds=150);matcher=CalibratedMatcher(model);matcher.calibrate(cx[~fit],cal['pairs'].label[~fit]);p=matcher.predict(vx)
  stem=f'oof_compact_d{depth}';np.save(ROOT/(stem+'_validation_p.npy'),p);joblib.dump({'matcher':matcher,'feature_names':X.columns.tolist(),'training_scope':'30000 cross-fitted original fitting anchors + 4000 calibration anchors; 1000 held for sigmoid'},ART/(stem+'.joblib'))
  threshold,_=choose(val,p);row={'model':stem,'threshold':threshold,'trees':model.tree_count_,'seconds':time.time()-started,**evaluate(val,p,threshold)};report.append(row);print('OOF_RESULT',json.dumps(row),flush=True);(OUT/'oof_selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');del matcher,model;gc.collect()
 print('OOF_BEST',json.dumps(max(report,key=lambda r:r['f0.5'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['oof','prepare','train']);args=p.parse_args()
 from .rescoring import sourcehash
 signature={'code':{s:sourcehash(Path('src')/s) for s in ['campaign_oof.py','context_experiments.py','peer_evidence.py','detail_evidence.py']},'inputs':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in [Path('work/real_v1/features/fit.joblib'),Path('work/improvements/fit_extra.joblib'),Path('work/real_v1/features/calibration.joblib'),Path('work/improvements/calibration_extra.joblib')]}}
 path=ROOT/'oof_signature.json'
 if path.exists() and json.loads(path.read_text())!=signature:raise ValueError('OOF inputs/code changed; preserve old run and use a new campaign directory')
 path.write_text(json.dumps(signature,indent=2),encoding='utf-8');globals()[args.stage]()
