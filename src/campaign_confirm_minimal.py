"""Pre-register and evaluate the inexpensive raw-text-only correction."""
import argparse,json
import joblib,numpy as np,pandas as pd
from .campaign_raw import ROOT,OUT,ART,GATE
from .campaign_confirm_raw import scores
from .raw_evidence import features
from .disk_store import connect
from .campaign_scoring import predict
from .rescoring import filehash
from .experiments import evaluate


def freeze():
    r=max(json.loads((OUT/'raw_minimal_selection.json').read_text(encoding='utf-8'))['experiments'],key=lambda x:x['f0.5'])
    f={'recipe':r,'model_hash':filehash(ART/(r['model']+'.joblib')),'feature_hash':filehash('src/raw_evidence.py'),'gate':list(GATE),
       'validation':'Same unused v5 as the full correction; frozen before v5 labels/scores inspected. Two fixed model hypotheses; use 97.5% paired interval.'}
    path=OUT/'minimal_frozen_v5.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==f
    else:
        if (OUT/'raw_confirmation_v5.json').exists():raise RuntimeError('v5 already revealed; use a fresh confirmation')
        path.write_text(json.dumps(f,indent=2),encoding='utf-8')


def run():
    frozen=json.loads((OUT/'minimal_frozen_v5.json').read_text(encoding='utf-8'));r=frozen['recipe'];path=ART/(r['model']+'.joblib')
    assert filehash(path)==frozen['model_hash'] and filehash('src/raw_evidence.py')==frozen['feature_hash']
    model=joblib.load(path);part=joblib.load(ROOT/'v5_base.joblib');p=np.load(ROOT/'v5_main_p.npy');q=p.copy();rows=np.flatnonzero((p>=GATE[0])&(p<GATE[1]));con=connect('work/windows_v1/train/records.sqlite',True)
    for lo in range(0,len(rows),5000):
        ix=rows[lo:lo+5000];X=features(part['pairs'].iloc[ix],con);X['current_probability']=p[ix];q[ix]=predict(model,X[model['feature_names']]);print('MINIMAL_CONFIRM',lo,len(rows),flush=True)
    q=(1-r['weight'])*p+r['weight']*q;np.save(ROOT/'v5_minimal_p.npy',q)
    delta=scores(part,q,r['threshold'])-scores(part,p,.6000000000000002);rng=np.random.default_rng(1943)
    boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(4000)]);ci=np.quantile(boot,[.0125,.9875]).tolist()
    report={'baseline':evaluate(part,p,.6000000000000002),'minimal':evaluate(part,q,r['threshold']),'delta':float(delta.mean()),'paired_97_5ci':ci,'promotion_supported':bool(ci[0]>0),'frozen':frozen}
    (OUT/'minimal_confirmation_v5.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('MINIMAL_CONFIRMATION',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);args=p.parse_args();globals()[args.stage]()
