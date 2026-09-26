"""Bounded, resumable GPU inference using frozen multilingual fusion recipes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,time
import joblib,numpy as np,pandas as pd
from .campaign_test_records import TestRecords,ROOT as EXTRAS
from .disk_store import fetch_records
from .rescoring import filehash,sourcehash
ROOT=Path('work/campaign_0942');OUT=Path('reports/campaign_0942');ART=Path('artifacts/campaign_0942')
MAIN=Path('work/campaign_0931/test_final_oof');RAW=ROOT/'test_minimal_packed'


def verify_frozen():
    frozen=json.loads((OUT/'frozen_v6.json').read_text(encoding='utf-8'))
    for path,h in frozen['models'].items():
        if filehash(path)!=h:raise ValueError('Frozen artifact changed: '+path)
    for path,h in frozen['code'].items():
        if sourcehash('src/'+path)!=h:raise ValueError('Frozen code changed: '+path)
    return frozen


def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))


def atomic_npy(path,a):
    tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:np.save(f,a)
    tmp.replace(path)


def run(args):
    frozen=verify_frozen();recipe=next(r for r in frozen['recipes'] if r['model']=='neural_stack_d3')
    if args.limit is None:
        result=json.loads((OUT/'confirmation_v6.json').read_text(encoding='utf-8'))
        if result['frozen']!=frozen or result['comparisons']['neural_vs_raw']['ci'][0]<=0:raise ValueError('Neural fresh confirmation not passed')
    checked=json.loads((OUT/'test_records_parity.json').read_text(encoding='utf-8'))
    if checked['adapter_hash']!=filehash('src/campaign_test_records.py') or checked['manifest_hash']!=filehash(EXTRAS/'COMPLETE.json'):raise ValueError('Record adapter changed after parity')
    root=ROOT/'test_neural';components=ROOT/'test_neural_components'
    for path in [root,components]:path.mkdir(exist_ok=True)
    sig={'model_sha256':filehash(ART/'neural_stack_d3.joblib'),'candidate_sha256':json.loads((MAIN/'signature.json').read_text(encoding='utf-8'))['candidate_sha256'],'batch':1000,'frozen':frozen,'inference_code':sourcehash('src/campaign_neural_test.py'),'records_parity':checked,'raw_signature':filehash(RAW/'signature.json'),'batch_size':64,'precision':'bfloat16'}
    path=root/'signature.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==sig
    else:path.write_text(json.dumps(sig,indent=2),encoding='utf-8')
    (root/'selection.json').write_text(json.dumps({'sha256':sig['model_sha256'],'threshold':recipe['threshold']},indent=2),encoding='utf-8')
    import torch
    from transformers import AutoModelForSequenceClassification,XLMRobertaTokenizer
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(2);model=AutoModelForSequenceClassification.from_pretrained(str(ART/'neural_minilm')).cuda().eval();tokenizer=XLMRobertaTokenizer.from_pretrained(str(ART/'neural_minilm'));stack=joblib.load(ART/'neural_stack_d3.joblib');records=TestRecords()
    print('NEURAL_DEVICE',torch.cuda.get_device_name(0),flush=True)
    n=1732544;limit=n if args.limit is None else min(n,args.limit);todo=[]
    for start in range(0,limit,1000):
        p=root/f'{start:08d}.npy';receipt=p.with_suffix('.json');comp=components/p.name
        if p.exists() and receipt.exists() and comp.exists():
            r=json.loads(receipt.read_text(encoding='utf-8'))
            if r!={'main':filehash(MAIN/p.name),'raw':filehash(RAW/p.name),'output':filehash(p),'neural':filehash(comp)}:raise ValueError('Inference cache changed')
        else:todo.append(start)
    def prepare(start):
        a=np.load(MAIN/f'{start:08d}.npy');raw=np.load(RAW/f'{start:08d}.npy')
        if not np.array_equal(a[['anchor','target']],raw[['anchor','target']]):raise ValueError('Raw pair alignment mismatch')
        mask=(a['p']>=.005)&(a['p']<.9999);b=a[mask]
        if not len(b):return start,a,raw,mask,None
        anc=fetch_records(records,'anchors',np.unique(b['anchor']));tar=fetch_records(records,'targets',np.unique(b['target']))
        def serial(r):return 'name: '+str(r.business_name)+'; address: '+str(r.business_address)+'; country: '+str(r.country)
        aa={r.rid:serial(r) for r in anc.itertuples(index=False)};tt={r.rid:serial(r) for r in tar.itertuples(index=False)}
        tokens=tokenizer([aa[r] for r in b['anchor']],[tt[r] for r in b['target']],max_length=192,truncation=True,padding='max_length',return_tensors='np')['input_ids']
        return start,a,raw,mask,tokens
    started=time.time();done=0;count=0
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(prepare,todo[0]) if todo else None
        for j,start in enumerate(todo):
            _,a,raw,mask,tokens=future.result()
            future=pool.submit(prepare,todo[j+1]) if j+1<len(todo) else None
            npred=np.zeros(int(mask.sum()),dtype=np.float32)
            with torch.inference_mode():
                for i in range(0,len(npred),64):
                    x=torch.as_tensor(tokens[i:i+64],device='cuda',dtype=torch.long)
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=x,attention_mask=x.ne(tokenizer.pad_token_id)).logits
                    npred[i:i+len(x)]=logits.float().softmax(-1)[:,1].cpu().numpy()
            b=a.copy()
            if len(npred):
                X=pd.DataFrame({'current':logit(a['p'][mask]),'raw':logit(raw['p'][mask]),'neural':logit(npred.astype(np.float64))});b['p'][mask]=stack['estimator'].predict_proba(X[stack['features']],thread_count=1)[:,1]
            p=root/f'{start:08d}.npy';comp=components/p.name;atomic_npy(comp,npred);atomic_npy(p,b)
            p.with_suffix('.json').write_text(json.dumps({'main':filehash(MAIN/p.name),'raw':filehash(RAW/p.name),'output':filehash(p),'neural':filehash(comp)}),encoding='utf-8')
            done+=min(1000,n-start);count+=len(npred);elapsed=time.time()-started;status={'last_completed_anchor':min(start+1000,n),'anchors_this_run':done,'neural_pairs_this_run':count,'seconds':elapsed,'anchors_per_second':done/elapsed,'estimated_remaining_seconds':max(0,limit-start-1000)/(done/elapsed),'complete':False,'device':torch.cuda.get_device_name(0)}
            (root/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print('NEURAL_TEST',json.dumps(status),flush=True)
    if args.limit is None:
        status={'anchors':n,'complete':True,'signature':sig,'seconds_this_run':time.time()-started};(root/'SCORING_COMPLETE.json').write_text(json.dumps(status,indent=2),encoding='utf-8');(root/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('NEURAL_TEST_DONE',done,time.time()-started,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);run(p.parse_args())
