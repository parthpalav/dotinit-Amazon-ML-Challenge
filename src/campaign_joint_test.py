"""Resume global-owner fusion using saved neural predictions; CPU inference."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,collections,json,multiprocessing,time
import joblib,numpy as np,pandas as pd
from .campaign_neural_test import ROOT,OUT,ART,MAIN,RAW,verify_frozen,logit,atomic_npy
from .campaign_test_records import TestRecords
from .campaign_reverse import Reverse,evidence
from .raw_evidence import features
from .campaign_scoring import predict
from .disk_store import fetch_records
from .rescoring import filehash,sourcehash
W={}


def initialize():
    W['con']=TestRecords();W['rev']=Reverse('test');W['rev'].con.close();W['rev'].con=W['con'];W['reverse']=joblib.load(ART/'raw_reverse_d7.joblib');W['stack']=joblib.load(ART/'joint_stack_d3.joblib')


def batch(start):
    name=f'{start:08d}.npy';a=np.load(MAIN/name);r=np.load(RAW/name);npred=np.load(ROOT/'test_neural_components'/name);mask=(a['p']>=.005)&(a['p']<.9999);b=a.copy()
    if len(npred)!=mask.sum() or not np.array_equal(a[['anchor','target']],r[['anchor','target']]):raise ValueError('Joint pair alignment mismatch')
    if mask.any():
        pairs=pd.DataFrame({'anchor_rid':a['anchor'][mask],'target_rid':a['target'][mask]});rx=features(pairs,W['con']);rx['current_probability']=a['p'][mask]
        # Bounded retrieval batches match the trained independent per-target search.
        ids=np.unique(pairs.target_rid);chunks=[]
        for lo in range(0,len(ids),400):chunks.append(W['rev'].candidates(fetch_records(W['con'],'targets',ids[lo:lo+400])))
        cx=evidence(pairs,pd.concat(chunks,ignore_index=True),W['con']);xx=pd.concat([rx,cx],axis=1);vp=predict(W['reverse'],xx[W['reverse']['feature_names']])
        X=pd.DataFrame({'current':logit(a['p'][mask]),'raw':logit(r['p'][mask]),'neural':logit(npred.astype(np.float64)),'reverse':logit(vp)});b['p'][mask]=W['stack']['estimator'].predict_proba(X[W['stack']['features']],thread_count=1)[:,1]
    return start,b,int(mask.sum())


def run(args):
    frozen=verify_frozen();recipe=next(r for r in frozen['recipes'] if r['model']=='joint_stack_d3')
    if args.limit is None:
        result=json.loads((OUT/'confirmation_v6.json').read_text(encoding='utf-8'))
        if result['frozen']!=frozen or result['chosen']!='joint_stack_d3':raise ValueError('Joint fresh confirmation not passed')
    if not (ROOT/'reverse_test/COMPLETE.json').exists():raise ValueError('Build reverse_test first')
    root=ROOT/'test_joint';root.mkdir(exist_ok=True);neural=ROOT/'test_neural'
    sig={'model_sha256':filehash(ART/'joint_stack_d3.joblib'),'candidate_sha256':json.loads((MAIN/'signature.json').read_text(encoding='utf-8'))['candidate_sha256'],'batch':1000,'frozen':frozen,'inference_code':sourcehash('src/campaign_joint_test.py'),'records_parity':json.loads((OUT/'test_records_parity.json').read_text(encoding='utf-8')),'neural_signature':filehash(neural/'signature.json'),'reverse_index_signature':filehash(ROOT/'reverse_test/signature.json')}
    if sig['records_parity']['adapter_hash']!=filehash('src/campaign_test_records.py'):raise ValueError('Adapter changed')
    path=root/'signature.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==sig
    else:path.write_text(json.dumps(sig,indent=2),encoding='utf-8')
    (root/'selection.json').write_text(json.dumps({'sha256':sig['model_sha256'],'threshold':recipe['threshold']},indent=2),encoding='utf-8')
    n=1732544;limit=n if args.limit is None else min(n,args.limit);pending=collections.deque();started=time.time();done=0;count=0
    def receipt(name):
        r=json.loads((neural/name).with_suffix('.json').read_text(encoding='utf-8'))
        if r!={'main':filehash(MAIN/name),'raw':filehash(RAW/name),'output':filehash(neural/name),'neural':filehash(ROOT/'test_neural_components'/name)}:raise ValueError('Neural input changed')
        return r
    def consume():
        nonlocal done,count
        future,input_receipt=pending.popleft();start,b,k=future.result();p=root/f'{start:08d}.npy';atomic_npy(p,b);p.with_suffix('.json').write_text(json.dumps({'inputs':input_receipt,'output':filehash(p)}),encoding='utf-8');done+=min(1000,n-start);count+=k;elapsed=time.time()-started
        status={'last_completed_anchor':min(n,start+1000),'anchors_this_run':done,'pairs_this_run':count,'seconds':elapsed,'anchors_per_second':done/elapsed,'estimated_remaining_seconds':max(0,limit-start-1000)/(done/elapsed),'complete':False};(root/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print('JOINT_TEST',json.dumps(status),flush=True)
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for start in range(0,limit,1000):
            p=root/f'{start:08d}.npy';r=receipt(p.name)
            if p.exists() and p.with_suffix('.json').exists():
                old=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
                if old!={'inputs':r,'output':filehash(p)}:raise ValueError('Joint cache changed')
                continue
            pending.append((pool.submit(batch,start),r))
            if len(pending)>=args.workers*2:consume()
        while pending:consume()
    if args.limit is None:
        status={'anchors':n,'complete':True,'signature':sig,'seconds_this_run':time.time()-started};(root/'SCORING_COMPLETE.json').write_text(json.dumps(status,indent=2),encoding='utf-8');(root/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int);run(p.parse_args())
