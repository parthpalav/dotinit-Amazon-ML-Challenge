"""Evaluate the frozen winner once on reserved anchors; never tune on this set."""
from pathlib import Path
import hashlib,json
import joblib,numpy as np,pandas as pd
from .experiments import evaluate
from .artifact_compat import load_legacy
from .disk_store import connect
from .config import Config

def per_anchor(part,p,t):
 truth=part['truth_counts'];pos={x:i for i,x in enumerate(truth)}
 ix=np.array([pos[x] for x in part['pairs'].source1_entity_id]);y=part['pairs'].label.to_numpy();chosen=p>=t
 pred=np.bincount(ix[chosen],minlength=len(truth));tp=np.bincount(ix[chosen&(y==1)],minlength=len(truth));actual=np.array(list(truth.values()))
 den=.25*actual+pred
 return np.divide(1.25*tp,den,out=np.ones(len(truth)),where=den>0)

def main():
 out=Path('reports/improvements');lock=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
 if hashlib.sha256(Path(lock['model']).read_bytes()).hexdigest()!=lock['sha256']:raise ValueError('Frozen model changed')
 cfg=Config.load('config/windows.json');part=joblib.load(Path(cfg.working_dir)/'features/confirmation.joblib')
 extra=joblib.load('work/improvements/confirmation_extra.joblib')
 if not extra['pair_ids'].equals(part['pairs'][['source1_entity_id','candidate_entity_id']]):raise ValueError('Pair alignment')
 model=joblib.load(lock['model']);f=pd.concat([part['features'],extra['features']],axis=1)
 if f.columns.tolist()!=model['feature_names']:raise ValueError('Feature schema')
 p=model['matcher'].predict(f);base=load_legacy('artifacts/real_model.joblib',recover_ranker=False);base['matcher'].estimator.set_params(n_jobs=2)
 bp=base['matcher'].predict(part['features'])
 newscore=per_anchor(part,p,lock['threshold']);oldscore=per_anchor(part,bp,base['threshold']);delta=newscore-oldscore
 rng=np.random.default_rng(20260926);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
 con=connect(Path(cfg.working_dir)/'train/records.sqlite',True);countries={}
 ids=list(part['truth_counts'])
 for start in range(0,len(ids),500):
  block=ids[start:start+500];countries.update(con.execute('select entity_id,country from anchors where entity_id in ('+','.join('?'*len(block))+')',block).fetchall())
 labels=np.array([countries[x] for x in ids]);con.close()
 report={'scope':'Reserved 5000 S1 anchors excluded from fitting, calibration, and model/threshold selection. US/India only; no estimate of France generalization.', 'frozen_selection':lock,'baseline':evaluate(part,bp,base['threshold']),'improved':evaluate(part,p,lock['threshold']),'paired_f05_delta':float(delta.mean()),'paired_bootstrap_95pct_ci':np.quantile(boot,[.025,.975]).tolist(),'countries':{str(c):{'anchors':int((labels==c).sum()),'baseline_f05':float(oldscore[labels==c].mean()),'improved_f05':float(newscore[labels==c].mean())} for c in np.unique(labels)}}
 np.save(out/'confirmation_probabilities.npy',p);np.save(out/'confirmation_baseline_probabilities.npy',bp)
 (out/'confirmation_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
 if report['paired_bootstrap_95pct_ci'][0]<=0:raise RuntimeError('Improvement not confirmed; do not promote')

if __name__=='__main__':main()
