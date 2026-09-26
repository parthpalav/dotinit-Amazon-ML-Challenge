"""Second fresh confirmation for the expanded cross-fitted reranker, no tuning."""
from pathlib import Path
from dataclasses import replace
import gc,json,logging
import joblib,numpy as np,pandas as pd
from .config import Config
from .real_pipeline import feature_subset
from .disk_store import connect,fetch_records
from .evidence import make_stats,enrich
from .context_experiments import context
from .peer_evidence import features as peers
from .detail_evidence import features as details
from .experiments import evaluate
from .rescoring import filehash
ROOT=Path('work/campaign_0931');OUT=Path('reports/campaign_0931')

def main():
 logging.basicConfig(level=logging.INFO);cfg=replace(Config.load('config/windows.json'),workers=1,reports_dir=str(OUT));sel=np.load(Path(cfg.working_dir)/'entity_selection.npz')
 used=np.concatenate([*[sel[k] for k in sel.files],np.load('work/improvements/confirmation_ids.npy'),np.load(ROOT/'confirmation_v3_ids.npy')]);eligible=np.setdiff1d(np.arange(1,2206822,dtype=np.uint32),used);ids=np.sort(np.random.default_rng(20260928).choice(eligible,5000,replace=False));assert not np.intersect1d(ids,used).size;np.save(ROOT/'confirmation_v4_ids.npy',ids)
 best=max(json.loads((OUT/'oof_final_selection.json').read_text()),key=lambda r:r['f0.5']);recipes=[]
 for name,stem,t in [('compact','context_peer_detail_compact_d5',.675),('oof',best['model'],best['threshold'])]:
  path=Path('artifacts/campaign_0931')/(stem+'.joblib');recipes.append({'name':name,'path':str(path),'sha256':filehash(path),'threshold':t})
 frozen={'seed':20260928,'anchors':5000,'excluded_prior_anchors':len(np.unique(used)),'models':recipes,'selection':'OOF model and threshold selected on original validation only; compact comparator fixed'};path=OUT/'frozen_confirmation_v4.json'
 if path.exists():assert json.loads(path.read_text())==frozen
 else:path.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
 part=feature_subset(cfg,'confirmation_v4',ids,Path(cfg.working_dir)/'feature_engineer.joblib',10320219);pairs=part['pairs'];con=connect(Path(cfg.working_dir)/'train/records.sqlite',True);stats=make_stats(cfg,'train');cache=ROOT/'confirmation_v4_compact_features.joblib'
 if cache.exists():X=joblib.load(cache)
 else:
  extra=pd.concat([enrich(pairs.iloc[i:i+10000],con,stats) for i in range(0,len(pairs),10000)]);full=pd.concat([part['features'],extra],axis=1);champ=joblib.load('artifacts/improvements/catboost_d10_evidence.joblib');p=champ['matcher'].predict(full);np.save(ROOT/'confirmation_v4_champion_p.npy',p);del champ,extra,full;gc.collect()
  a=pairs.anchor_rid.to_numpy();starts=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,len(a)];blocks=[]
  for g in range(0,len(starts)-1,250):
   lo=starts[g];hi=starts[min(g+250,len(starts)-1)];b=pairs.iloc[lo:hi];q=p[lo:hi];blocks.append(pd.concat([context(b,q),peers(b,q,con),details(b,con,stats)],axis=1));print('CONFIRM4_FEATURES',g,flush=True)
  X=pd.concat(blocks);joblib.dump(X,cache)
 p=np.load(ROOT/'confirmation_v4_champion_p.npy');predictions={'champion':(p,.675)};results={'champion':evaluate(part,p,.675)}
 for r in recipes:
  m=joblib.load(r['path']);q=m['matcher'].predict(X[m['feature_names']]);predictions[r['name']]=(q,r['threshold']);np.save(ROOT/('confirmation_v4_'+r['name']+'_p.npy'),q);results[r['name']]=evaluate(part,q,r['threshold'])
 truth=part['truth_counts'];pos={v:i for i,v in enumerate(truth)};ix=np.array([pos[x] for x in pairs.source1_entity_id]);counts=np.array(list(truth.values()));y=pairs.label.to_numpy();scores={}
 for name,(q,t) in predictions.items():
  keep=q>=t;tp=np.bincount(ix,weights=keep*y,minlength=len(counts));pred=np.bincount(ix,weights=keep,minlength=len(counts));den=.25*counts+pred;scores[name]=np.divide(1.25*tp,den,out=np.ones(len(counts)),where=den>0)
 anc=fetch_records(con,'anchors',ids).set_index('entity_id');countries=anc.loc[list(truth),'country'].to_numpy()
 for name in scores:results[name]['country_f05']={c:float(scores[name][countries==c].mean()) for c in np.unique(countries)}
 for name in ['compact','oof']:
  delta=scores[name]-scores['champion'];rng=np.random.default_rng(984);boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)]);results[name]['paired_delta']=float(delta.mean());results[name]['paired_95ci']=np.quantile(boot,[.025,.975]).tolist()
 delta=scores['oof']-scores['compact'];rng=np.random.default_rng(985);boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)]);results['oof']['delta_vs_compact']=float(delta.mean());results['oof']['ci_vs_compact']=np.quantile(boot,[.025,.975]).tolist()
 (OUT/'confirmation_v4_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print('CONFIRMATION4',json.dumps(results),flush=True);con.close()
if __name__=='__main__':main()
