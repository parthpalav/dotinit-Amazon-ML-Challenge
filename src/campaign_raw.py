"""Resumable raw-text residual experiment; old champions are immutable inputs."""
from pathlib import Path
import argparse, gc, json, time, hashlib
import joblib, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from .raw_evidence import features, VERSION
from .disk_store import connect
from .context_experiments import context
from .campaign_scoring import predict
from .model import CalibratedMatcher
from .experiments import evaluate, choose
ROOT=Path('work/campaign_0942'); OUT=Path('reports/campaign_0942'); ART=Path('artifacts/campaign_0942')
OLD=Path('work/campaign_0931')
GATE=(.005,.9999)


def load(split):
    if split=='selection':
        part=joblib.load('work/real_v1/features/validation.joblib')
        p=np.load(OLD/'oof_compact_d9_validation_p.npy')
        X=pd.concat([context(part['pairs'],np.load(OLD/'validation_champion_p.npy')),
                     joblib.load(OLD/'peer_validation.joblib')['features'],joblib.load(OLD/'detail_validation.joblib')['features']],axis=1)
    else:
        part=joblib.load(f'work/windows_v1/features/confirmation_{split}.joblib')
        X=joblib.load(OLD/('confirmation_v3_all_features.joblib' if split=='v3' else 'confirmation_v4_compact_features.joblib'))
        model=joblib.load('artifacts/campaign_0931/oof_compact_d9.joblib'); X=X[model['feature_names']]
        p=predict(model,X)
    del part['features']; return part,X,p


def prepare():
    con=connect('work/windows_v1/train/records.sqlite',True)
    for split in ['selection','v3','v4']:
        part,X,p=load(split); pairs=part['pairs']; gate=(p>=GATE[0])&(p<GATE[1]); rows=np.flatnonzero(gate)
        folder=ROOT/('raw_'+split);folder.mkdir(exist_ok=True)
        signature={'version':VERSION,'source':hashlib.sha256(Path('src/raw_evidence.py').read_bytes()).hexdigest(),
                   'pairs':hashlib.sha256(pairs[['anchor_rid','target_rid']].to_numpy(dtype='<u4').tobytes()).hexdigest(),'gate':list(GATE)}
        manifest=folder/'signature.json'
        if manifest.exists():assert json.loads(manifest.read_text(encoding='utf-8'))==signature
        else:manifest.write_text(json.dumps(signature,indent=2),encoding='utf-8')
        blocks=[]
        for lo in range(0,len(rows),5000):
            path=folder/f'{lo:07d}.joblib'
            if not path.exists():
                x=features(pairs.iloc[rows[lo:lo+5000]],con); tmp=path.with_suffix('.partial');joblib.dump(x,tmp);tmp.replace(path)
            blocks.append(joblib.load(path));print('RAW_FEATURES',split,lo,len(rows),flush=True)
        raw=pd.concat(blocks);xx=X.iloc[rows].copy();xx['current_probability']=p[rows];xx=pd.concat([xx,raw],axis=1)
        assert xx.index.equals(pairs.iloc[rows].index)
        joblib.dump({'part':part,'X':xx,'p':p,'rows':rows},ROOT/(split+'_raw.joblib'))
        print('RAW_PREPARED',split,len(rows),flush=True);del part,X,xx,raw;gc.collect()
    con.close()


def subset(part,mask):
    pairs=part['pairs'].loc[mask].reset_index(drop=True);ids=set(pairs.source1_entity_id)
    return {'pairs':pairs,'truth_counts':{k:v for k,v in part['truth_counts'].items() if k in ids}}


def train(minimal=False,reverse=False):
    parts=[]; xs=[]; ps=[]; rs=[];offset=0
    for split in ['selection','v3','v4']:
        d=joblib.load(ROOT/(split+'_raw.joblib'))
        if reverse:
            from .campaign_reverse import evidence
            path=ROOT/(split+'_reverse_features.joblib')
            if path.exists():rx=joblib.load(path)
            else:
                con=connect('work/windows_v1/train/records.sqlite',True);rx=evidence(d['part']['pairs'].iloc[d['rows']],joblib.load(ROOT/(split+'_competitors.joblib')),con);joblib.dump(rx,path);con.close()
            assert rx.index.equals(d['X'].index);d['X']=pd.concat([d['X'],rx],axis=1)
        parts.append(d['part']);xs.append(d['X']);ps.append(d['p']);rs.append(d['rows']+offset);offset+=len(d['p'])
    X=pd.concat(xs,ignore_index=True);p=np.concatenate(ps);rows=np.concatenate(rs)
    if minimal or reverse:X=X[[c for c in X.columns if c.startswith(('raw_','reverse_')) or c=='current_probability']]
    part={'pairs':pd.concat([v['pairs'] for v in parts],ignore_index=True),'truth_counts':{k:v for d in parts for k,v in d['truth_counts'].items()}}
    ids=np.array(list(part['truth_counts']),dtype=object);np.random.default_rng(20260929).shuffle(ids)
    # 14k fit, 3k early stopping/calibration, 3k development selection.
    fitids=set(ids[6000:]);calids=set(ids[3000:6000]);selids=set(ids[:3000])
    fit=part['pairs'].iloc[rows].source1_entity_id.isin(fitids).to_numpy();cal=part['pairs'].iloc[rows].source1_entity_id.isin(calids).to_numpy()
    select=part['pairs'].source1_entity_id.isin(selids).to_numpy();sp=subset(part,select);base=p[select]
    y=part['pairs'].iloc[rows].label.to_numpy();report={'baseline':evaluate(sp,base,.6000000000000002),'experiments':[],'gate':GATE,'split_seed':20260929}
    for depth in [5,7]:
        started=time.time();model=CatBoostClassifier(iterations=1800,depth=depth,learning_rate=.035,l2_leaf_reg=10,loss_function='Logloss',task_type='GPU',devices='0',random_seed=44,thread_count=3,allow_writing_files=False,verbose=200)
        model.fit(X.loc[fit],y[fit],eval_set=(X.loc[cal],y[cal]),early_stopping_rounds=150)
        matcher=CalibratedMatcher(model);matcher.calibrate(X.loc[cal],y[cal]);q=p.copy();q[rows]=predict({'matcher':matcher,'feature_names':X.columns.tolist()},X)
        stem=f'raw_{"reverse" if reverse else "minimal" if minimal else "residual"}_d{depth}';joblib.dump({'matcher':matcher,'feature_names':X.columns.tolist(),'gate':GATE,'training_ids':ids[6000:],'calibration_ids':ids[3000:6000]},ART/(stem+'.joblib'))
        np.save(ROOT/(stem+'_development_p.npy'),q)
        for weight in [0.5,1.0]:
            qq=(1-weight)*p+weight*q;t,_=choose(sp,qq[select]);r={'model':stem,'weight':weight,'threshold':t,'trees':model.tree_count_,'seconds':time.time()-started,**evaluate(sp,qq[select],t)};report['experiments'].append(r);print('RAW_RESULT',json.dumps(r),flush=True)
        (OUT/('raw_reverse_selection.json' if reverse else 'raw_minimal_selection.json' if minimal else 'raw_selection.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        del matcher,model;gc.collect()
    print('RAW_DONE',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train']);parser.add_argument('--minimal',action='store_true');parser.add_argument('--reverse',action='store_true');args=parser.parse_args()
    for folder in [ROOT,OUT,ART]:folder.mkdir(parents=True,exist_ok=True)
    if args.stage=='train':train(args.minimal,args.reverse)
    else:prepare()
