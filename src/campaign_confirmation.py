"""Frozen-model confirmation on previously unused anchors. No tuning here."""
from pathlib import Path
from dataclasses import replace
import gc,hashlib,json,logging,time
import joblib,numpy as np,pandas as pd
from .config import Config
from .real_pipeline import feature_subset
from .disk_store import connect
from .evidence import make_stats,enrich
from .context_experiments import context
from .peer_evidence import features as peer_features
from .detail_evidence import features as detail_features
from .experiments import evaluate
ROOT=Path('work/campaign_0931');OUT=Path('reports/campaign_0931')
CHOICES=[('full','context_peer_detail_d9',.675),('compact','context_peer_detail_compact_d5',.675)]

def main():
 logging.basicConfig(level=logging.INFO);ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 cfg=replace(Config.load('config/windows.json'),workers=1,reports_dir=str(OUT));sel=np.load(Path(cfg.working_dir)/'entity_selection.npz')
 used=np.concatenate([*[sel[k] for k in sel.files],np.load('work/improvements/confirmation_ids.npy')]);eligible=np.setdiff1d(np.arange(1,2206822,dtype=np.uint32),used)
 ids=np.sort(np.random.default_rng(20260927).choice(eligible,5000,replace=False));assert not np.intersect1d(ids,used).size
 np.save(ROOT/'confirmation_v3_ids.npy',ids)
 frozen={'seed':20260927,'anchors':5000,'excluded_prior_anchors':len(np.unique(used)),'models':[],'preferred':'compact if confirmation supports improvement; full diagnostic comparison; thresholds frozen before preparation'}
 for name,stem,t in CHOICES:
  path=Path('artifacts/campaign_0931')/(stem+'.joblib');frozen['models'].append({'name':name,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'threshold':t})
 freeze=OUT/'frozen_confirmation_v3.json'
 if freeze.exists():assert json.loads(freeze.read_text())==frozen
 else:freeze.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
 part=feature_subset(cfg,'confirmation_v3',ids,Path(cfg.working_dir)/'feature_engineer.joblib',10320219)
 con=connect(Path(cfg.working_dir)/'train/records.sqlite',True);stats=make_stats(cfg,'train');pairs=part['pairs'];base=part['features'];del part['features'];gc.collect()
 cache=ROOT/'confirmation_v3_all_features.joblib'
 if cache.exists():X=joblib.load(cache)
 else:
  extra=pd.concat([enrich(pairs.iloc[i:i+10000],con,stats) for i in range(0,len(pairs),10000)])
  full=pd.concat([base,extra],axis=1);champ=joblib.load('artifacts/improvements/catboost_d10_evidence.joblib');p=champ['matcher'].predict(full);np.save(ROOT/'confirmation_v3_champion_p.npy',p);del champ
  anchor=pairs.anchor_rid.to_numpy();starts=np.r_[0,np.flatnonzero(anchor[1:]!=anchor[:-1])+1,len(anchor)];peer=[];detail=[]
  for g in range(0,len(starts)-1,250):
   lo=starts[g];hi=starts[min(g+250,len(starts)-1)];batch=pairs.iloc[lo:hi];peer.append(peer_features(batch,p[lo:hi],con));detail.append(detail_features(batch,con,stats));print('CONFIRM_FEATURES',g,flush=True)
  X=pd.concat([full,context(pairs,p),pd.concat(peer),pd.concat(detail)],axis=1);joblib.dump(X,cache)
 p=np.load(ROOT/'confirmation_v3_champion_p.npy');results={'champion':evaluate(part,p,.675)};predictions={'champion':p}
 for recipe in frozen['models']:
  m=joblib.load(recipe['path']);q=m['matcher'].predict(X[m['feature_names']]);predictions[recipe['name']]=q;np.save(ROOT/('confirmation_v3_'+recipe['name']+'_p.npy'),q);results[recipe['name']]=evaluate(part,q,recipe['threshold']);del m
 truth=part['truth_counts'];pos={v:i for i,v in enumerate(truth)};ix=np.array([pos[x] for x in pairs.source1_entity_id]);counts=np.array(list(truth.values()));y=pairs.label.to_numpy();scores={}
 for name,q in predictions.items():
  keep=q>=.675;tp=np.bincount(ix,weights=keep*y,minlength=len(counts));pred=np.bincount(ix,weights=keep,minlength=len(counts));den=.25*counts+pred;scores[name]=np.divide(1.25*tp,den,out=np.ones(len(counts)),where=den>0)
 for name in ['full','compact']:
  delta=scores[name]-scores['champion'];rng=np.random.default_rng(984);boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)]);results[name]['paired_delta']=float(delta.mean());results[name]['paired_95ci']=np.quantile(boot,[.025,.975]).tolist()
 (OUT/'confirmation_v3_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print('CONFIRMATION',json.dumps(results),flush=True);con.close()
if __name__=='__main__':main()
