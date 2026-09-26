"""Resume cheap raw-text correction from completed main-model score shards."""
import argparse,collections,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .disk_store import connect
from .raw_evidence import features
from .campaign_scoring import predict
from .rescoring import DTYPE,filehash
W={};MAIN=Path('work/campaign_0931/test_final_oof');OUT=Path('reports/campaign_0942')


def initialize(recipe):
    W['con']=connect('work/windows_v1/test/records.sqlite',True);W['model']=joblib.load('artifacts/campaign_0942/'+recipe['model']+'.joblib');W['recipe']=recipe


def batch(start):
    path=MAIN/f'{start:08d}.npy';a=np.load(path);b=a.copy();low,high=W['model']['gate'];mask=(a['p']>=low)&(a['p']<high)
    if mask.any():
        pairs=pd.DataFrame({'anchor_rid':a['anchor'][mask],'target_rid':a['target'][mask]});X=features(pairs,W['con']);X['current_probability']=a['p'][mask]
        q=predict(W['model'],X[W['model']['feature_names']]);weight=W['recipe']['weight'];b['p'][mask]=(1-weight)*a['p'][mask]+weight*q
    return start,b,int(mask.sum()),filehash(path)


def run(args):
    frozen=json.loads((OUT/'minimal_frozen_v5.json').read_text(encoding='utf-8'));recipe=frozen['recipe'];mp=Path('artifacts/campaign_0942')/(recipe['model']+'.joblib')
    if filehash(mp)!=frozen['model_hash'] or filehash('src/raw_evidence.py')!=frozen['feature_hash']:raise ValueError('Frozen model/features changed')
    if args.limit is None:
        decision=json.loads((OUT/'minimal_confirmation_v5.json').read_text(encoding='utf-8'))
        if not decision['promotion_supported'] or decision['frozen']!=frozen:raise ValueError('Fresh confirmation required')
    main_sig=json.loads((MAIN/'signature.json').read_text(encoding='utf-8'));n=1732544
    if json.loads((MAIN/'SCORING_COMPLETE.json').read_text(encoding='utf-8'))['anchors']!=n:raise ValueError('Main incomplete')
    root=Path(args.work);root.mkdir(parents=True,exist_ok=True)
    sig={'model_sha256':frozen['model_hash'],'candidate_sha256':main_sig['candidate_sha256'],'batch':1000,'main_signature_sha256':filehash(MAIN/'signature.json'),'raw_sha256':frozen['feature_hash'],'scoring_sha256':filehash(__file__),'recipe':recipe}
    path=root/'signature.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==sig
    else:path.write_text(json.dumps(sig,indent=2),encoding='utf-8')
    (root/'selection.json').write_text(json.dumps({'sha256':frozen['model_hash'],'threshold':recipe['threshold']},indent=2),encoding='utf-8')
    started=time.time();done=0;changed=0;pending=collections.deque();limit=n if args.limit is None else min(n,args.limit)
    def consume():
        nonlocal done,changed
        start,a,count,input_hash=pending.popleft().result();dest=root/f'{start:08d}.npy';tmp=dest.with_suffix('.partial')
        with tmp.open('wb') as f:np.save(f,a)
        tmp.replace(dest);dest.with_suffix('.json').write_text(json.dumps({'input_hash':input_hash,'output_hash':filehash(dest)}),encoding='utf-8')
        done+=min(1000,n-start);changed+=count;elapsed=time.time()-started;status={'last_completed_anchor':min(n,start+1000),'anchors_this_run':done,'gated_pairs_this_run':changed,'seconds':elapsed,'anchors_per_second':done/elapsed,'estimated_remaining_seconds':max(0,limit-start-1000)/(done/elapsed),'complete':False}
        (root/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print('RAW_TEST',json.dumps(status),flush=True)
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(recipe,)) as pool:
        for start in range(0,limit,1000):
            dest=root/f'{start:08d}.npy';receipt=dest.with_suffix('.json')
            if dest.exists() and receipt.exists():
                r=json.loads(receipt.read_text(encoding='utf-8'))
                if r['input_hash']!=filehash(MAIN/dest.name) or r['output_hash']!=filehash(dest):raise ValueError('Cached scores changed')
                continue
            pending.append(pool.submit(batch,start))
            if len(pending)>=args.workers*2:consume()
        while pending:consume()
    if args.limit is None:
        report={'anchors':n,'signature':sig,'seconds_this_run':time.time()-started,'complete':True}
        (root/'SCORING_COMPLETE.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(root/'progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RAW_SCORING_DONE',done,time.time()-started,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int);p.add_argument('--work',default='work/campaign_0942/test_minimal');run(p.parse_args())
