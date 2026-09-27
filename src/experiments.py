"""Measured model experiments. Baseline artifacts/outputs are never overwritten."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .real_metrics import tune,entity_scores,probability_diagnostics
from .model import CalibratedMatcher

def evaluate(part,p,threshold):
    truth=part['truth_counts'];pos={v:i for i,v in enumerate(truth)}
    indices=np.array([pos[x] for x in part['pairs'].source1_entity_id],dtype=np.int32)
    y=part['pairs'].label.to_numpy();counts=np.array(list(truth.values()))
    return entity_scores(indices,y,p,counts,threshold,10320219)

def choose(part,p):
    truth=part['truth_counts'];pos={v:i for i,v in enumerate(truth)}
    indices=np.array([pos[x] for x in part['pairs'].source1_entity_id],dtype=np.int32)
    return tune(indices,part['pairs'].label.to_numpy(),p,np.array(list(truth.values())),10320219)

def run(args):
    from catboost import CatBoostClassifier
    out=Path('reports/improvements');out.mkdir(parents=True,exist_ok=True)
    artifactdir=Path('artifacts/improvements');artifactdir.mkdir(parents=True,exist_ok=True)
    data={s:joblib.load(Path('work/real_v1/features')/(s+'.joblib')) for s in ['fit','calibration','validation']}
    from .artifact_compat import load_legacy
    baseline=load_legacy('artifacts/real_model.joblib', recover_ranker=False); baseline['matcher'].estimator.set_params(n_jobs=4)
    p=baseline['matcher'].predict(data['validation']['features'])
    base=evaluate(data['validation'],p,baseline['threshold'])
    np.save(out/'baseline_validation_probabilities.npy',p)
    (out/'baseline_reproduction.json').write_text(json.dumps(base,indent=2), encoding='utf-8')
    print('BASELINE',json.dumps(base),flush=True)
    if abs(base['f0.5']-baseline['validation_metrics']['f0.5'])>1e-10:raise ValueError('Baseline parity failure')
    del baseline
    if args.enriched:
        for s in data:
            extra=joblib.load(Path('work/improvements')/(s+'_extra.joblib'))
            if not extra['pair_ids'].equals(data[s]['pairs'][['source1_entity_id','candidate_entity_id']]):raise ValueError('Extra feature rows differ')
            data[s]['features']=pd.concat([data[s]['features'],extra['features']],axis=1)
    name=f'catboost_d{args.depth}'+('_evidence' if args.enriched else '_baseline_features')
    started=time.time()
    model=CatBoostClassifier(iterations=args.iterations,depth=args.depth,learning_rate=.05,l2_leaf_reg=5,
        loss_function='Logloss',eval_metric='Logloss',random_seed=42,task_type=args.device,
        thread_count=4,allow_writing_files=False,verbose=100)
    model.fit(data['fit']['features'],data['fit']['pairs'].label,
        eval_set=(data['calibration']['features'],data['calibration']['pairs'].label),early_stopping_rounds=150)
    matcher=CalibratedMatcher(model);matcher.calibrate(data['calibration']['features'],data['calibration']['pairs'].label)
    p=matcher.predict(data['validation']['features']);threshold,table=choose(data['validation'],p)
    metrics=evaluate(data['validation'],p,threshold)
    report={'name':name,'metrics':metrics,'threshold':threshold,'trees':model.tree_count_,
        'seconds':time.time()-started,'device':args.device,'features':data['fit']['features'].columns.tolist(),
        'scope':'Existing validation used for model/threshold selection; fresh confirmation holdout required before promotion'}
    table.to_csv(out/(name+'_thresholds.tsv'),sep='\t',index=False)
    np.save(out/(name+'_validation_probabilities.npy'),p)
    joblib.dump({'matcher':matcher,'threshold':threshold,'feature_names':report['features'],'report':report},artifactdir/(name+'.joblib'))
    (out/(name+'.json')).write_text(json.dumps(report,indent=2), encoding='utf-8')
    print('RESULT',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--depth',type=int,default=8);p.add_argument('--iterations',type=int,default=1800)
    p.add_argument('--device',choices=['GPU','CPU'],default='GPU');p.add_argument('--enriched',action='store_true')
    run(p.parse_args())
