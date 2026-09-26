"""Length-sorted GPU batches; tokenization and learned model remain unchanged."""
import numpy as np


def infer(model,tokens,pad,batch=256):
    import torch
    lengths=(tokens!=pad).sum(axis=1);order=np.argsort(lengths,kind='stable');result=np.empty(len(tokens),dtype=np.float64)
    with torch.inference_mode():
        for lo in range(0,len(tokens),batch):
            ids=order[lo:lo+batch];width=min(192,int(np.ceil(lengths[ids].max()/16)*16));x=torch.as_tensor(tokens[ids,:width],device='cuda',dtype=torch.long)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=x,attention_mask=x.ne(pad)).logits
            result[ids]=logits.float().softmax(-1)[:,1].cpu().numpy()
    return result


def pilot():
    import json,time,joblib,torch,pandas as pd
    from transformers import AutoModelForSequenceClassification,XLMRobertaTokenizer
    from .campaign_neural_test import ROOT,OUT,ART,MAIN,RAW,logit
    from .campaign_test_records import TestRecords
    from .disk_store import fetch_records
    from .rescoring import sourcehash
    torch.set_num_threads(2);model=AutoModelForSequenceClassification.from_pretrained(str(ART/'neural_minilm')).cuda().eval();tokenizer=XLMRobertaTokenizer.from_pretrained(str(ART/'neural_minilm'));records=TestRecords();stack=joblib.load(ART/'neural_stack_d3.joblib');results=[]
    for start in [0,1000]:
        name=f'{start:08d}.npy';a=np.load(MAIN/name);r=np.load(RAW/name);old=np.load(ROOT/'test_neural_components'/name);mask=(a['p']>=.005)&(a['p']<.9999);b=a[mask]
        def serial(x):return 'name: '+str(x.business_name)+'; address: '+str(x.business_address)+'; country: '+str(x.country)
        aa={x.rid:serial(x) for x in fetch_records(records,'anchors',np.unique(b['anchor'])).itertuples(index=False)};tt={x.rid:serial(x) for x in fetch_records(records,'targets',np.unique(b['target'])).itertuples(index=False)}
        tokens=tokenizer([aa[i] for i in b['anchor']],[tt[i] for i in b['target']],max_length=192,truncation=True,padding='max_length',return_tensors='np')['input_ids']
        begin=time.time();p=infer(model,tokens,tokenizer.pad_token_id);elapsed=time.time()-begin
        preds=[]
        for q in [old.astype(np.float64),p]:
            X=pd.DataFrame({'current':logit(b['p']),'raw':logit(r['p'][mask]),'neural':logit(q)});preds.append(stack['estimator'].predict_proba(X,thread_count=1)[:,1])
        row={'start':start,'pairs':len(p),'gpu_seconds':elapsed,'pairs_per_second':len(p)/elapsed,'length_quantiles':np.quantile((tokens!=tokenizer.pad_token_id).sum(1),[0,.5,.9,1]).tolist(),'max_neural_probability_delta':float(abs(p-old).max()),'mean_neural_probability_delta':float(abs(p-old).mean()),'fusion_decision_disagreements':int(((preds[0]>=.6750000000000003)!=(preds[1]>=.6750000000000003)).sum())};results.append(row);print('BATCH_PILOT',json.dumps(row),flush=True)
    report={'implementation_hash':sourcehash('src/neural_batched.py'),'results':results,'scope':'Unlabelled test timing/numerical comparison only; promotion still requires fresh validation.'};(OUT/'neural_batch_pilot.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':pilot()
