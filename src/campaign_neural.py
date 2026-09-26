"""Checkpointed multilingual pair classifier, using supplied labelled pairs only."""
from pathlib import Path
import argparse,gc,json,time,math,os
import joblib,numpy as np,pandas as pd
from .campaign_raw import ROOT,OUT,ART,GATE
from .disk_store import connect,fetch_records

MODEL='microsoft/Multilingual-MiniLM-L12-H384'
PRETRAINED=ART/'pretrained_minilm'; BEST=ART/'neural_minilm'; LENGTH=192


def prepare():
    from huggingface_hub import snapshot_download,model_info
    from transformers import XLMRobertaTokenizer
    if not (PRETRAINED/'config.json').exists():
        info=model_info(MODEL);snapshot_download(MODEL,revision=info.sha,local_dir=str(PRETRAINED),allow_patterns=['*.json','pytorch_model.bin','*.model','README.md','LICENSE*'])
        (OUT/'neural_pretrained.json').write_text(json.dumps({'model':MODEL,'revision':info.sha,'license':'MIT','source':'https://huggingface.co/'+MODEL,'purpose':'Generic pretrained language representation; no external identity lookup'},indent=2),encoding='utf-8')
    tokenizer=XLMRobertaTokenizer.from_pretrained(str(PRETRAINED));con=connect('work/windows_v1/train/records.sqlite',True)
    parts=[];rows=[];probs=[];offset=0
    for split in ['selection','v3','v4']:
        d=joblib.load(ROOT/(split+'_raw.joblib'));parts.append(d['part']);rows.append(d['rows']+offset);probs.append(d['p']);offset+=len(d['p']);del d;gc.collect()
    part={'pairs':pd.concat([d['pairs'] for d in parts],ignore_index=True),'truth_counts':{k:v for d in parts for k,v in d['truth_counts'].items()}};rows=np.concatenate(rows);p=np.concatenate(probs)
    ids=np.array(list(part['truth_counts']),dtype=object);np.random.default_rng(20260929).shuffle(ids)
    source=part['pairs'].iloc[rows].source1_entity_id
    split=np.where(source.isin(set(ids[6000:])),0,np.where(source.isin(set(ids[3000:6000])),1,2)).astype(np.int8)
    folder=ROOT/'neural_tokens';folder.mkdir(exist_ok=True);chunks=[]
    for lo in range(0,len(rows),4000):
        path=folder/f'{lo:07d}.npy'
        if not path.exists():
            b=part['pairs'].iloc[rows[lo:lo+4000]];a=fetch_records(con,'anchors',b.anchor_rid.unique());t=fetch_records(con,'targets',b.target_rid.unique())
            def serial(r):return 'name: '+str(r.business_name)+'; address: '+str(r.business_address)+'; country: '+str(r.country)
            aa={r.rid:serial(r) for r in a.itertuples(index=False)};tt={r.rid:serial(r) for r in t.itertuples(index=False)}
            tokenized=tokenizer([aa[x] for x in b.anchor_rid],[tt[x] for x in b.target_rid],max_length=LENGTH,truncation=True,padding='max_length',return_tensors='np')
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:np.save(f,tokenized['input_ids'].astype(np.int32))
            tmp.replace(path)
        chunks.append(np.load(path));print('NEURAL_TOKENIZED',lo,len(rows),flush=True)
    np.save(ROOT/'neural_input_ids.npy',np.concatenate(chunks));joblib.dump({'part':part,'rows':rows,'p':p,'split':split,'pad_token_id':tokenizer.pad_token_id,'model':MODEL,'length':LENGTH},ROOT/'neural_data.joblib');con.close()


def train(args):
    import torch
    from transformers import AutoModelForSequenceClassification,XLMRobertaTokenizer,get_linear_schedule_with_warmup
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU training refused')
    torch.set_num_threads(2);torch.manual_seed(1942);torch.cuda.manual_seed_all(1942);torch.backends.cuda.matmul.allow_tf32=True
    print('NEURAL_DEVICE',torch.cuda.get_device_name(0),torch.__version__,flush=True)
    data=joblib.load(ROOT/'neural_data.joblib');tokens=np.load(ROOT/'neural_input_ids.npy',mmap_mode='r');labels=data['part']['pairs'].iloc[data['rows']].label.to_numpy(dtype=np.int64);trainids=np.flatnonzero(data['split']==0);calids=np.flatnonzero(data['split']==1)
    model=AutoModelForSequenceClassification.from_pretrained(str(PRETRAINED),num_labels=2).cuda()
    # Keep multilingual token embeddings intact and avoid 96M embedding optimizer states.
    for par in model.bert.embeddings.parameters():par.requires_grad=False
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=4e-5,weight_decay=.01)
    steps=math.ceil(len(trainids)/args.batch);total=steps*args.epochs;scheduler=get_linear_schedule_with_warmup(optimizer,max(1,total//20),total)
    checkpoint=ROOT/'neural_checkpoint.pt';epoch0=0;next_batch=0;best_loss=float('inf');global_step=0
    def save(epoch,next_batch):
        tmp=checkpoint.with_suffix('.partial');torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'next_batch':next_batch,'best_loss':best_loss,'global_step':global_step,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'batch':args.batch,'epochs':args.epochs},tmp);tmp.replace(checkpoint)
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['batch']==args.batch and ck['epochs']==args.epochs
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);epoch0=ck['epoch'];next_batch=ck['next_batch'];best_loss=ck['best_loss'];global_step=ck['global_step'];torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck;gc.collect()
    started=time.time();losses=[]
    for epoch in range(epoch0,args.epochs):
        order=np.random.default_rng(1942+epoch).permutation(trainids);model.train()
        for batch in range(next_batch,steps):
            ix=order[batch*args.batch:(batch+1)*args.batch];x=torch.as_tensor(np.array(tokens[ix]),device='cuda',dtype=torch.long);y=torch.as_tensor(labels[ix],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_ids=x,attention_mask=x.ne(data['pad_token_id']),labels=y).loss
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();global_step+=1;losses.append(float(loss.detach()))
            if global_step%100==0:
                print('NEURAL_TRAIN',json.dumps({'epoch':epoch,'step':global_step,'total_steps':total,'loss':float(np.mean(losses[-100:])),'seconds':time.time()-started,'gpu_allocated_gb':torch.cuda.memory_allocated()/1e9}),flush=True)
            if global_step%500==0:save(epoch,batch+1)
        next_batch=0;model.eval();cal_loss=0
        with torch.inference_mode():
            for lo in range(0,len(calids),args.batch*2):
                ix=calids[lo:lo+args.batch*2];x=torch.as_tensor(np.array(tokens[ix]),device='cuda',dtype=torch.long);y=torch.as_tensor(labels[ix],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_ids=x,attention_mask=x.ne(data['pad_token_id']),labels=y).loss
                cal_loss+=float(loss)*len(ix)
        cal_loss/=len(calids)
        if cal_loss<best_loss:
            best_loss=cal_loss;model.save_pretrained(str(BEST));XLMRobertaTokenizer.from_pretrained(str(PRETRAINED)).save_pretrained(str(BEST))
            (OUT/'neural_best.json').write_text(json.dumps({'epoch':epoch,'calibration_logloss':best_loss,'model':MODEL,'frozen_embeddings':True,'fit_anchors':14000,'calibration_anchors':3000,'selection_anchors':3000,'parameters':sum(p.numel() for p in model.parameters()),'device':torch.cuda.get_device_name(0)},indent=2),encoding='utf-8')
        print('NEURAL_EPOCH',epoch,'calibration_loss',cal_loss,'best',best_loss,flush=True);save(epoch+1,0)


def infer(args):
    import torch
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(2);data=joblib.load(ROOT/'neural_data.joblib');tokens=np.load(ROOT/'neural_input_ids.npy',mmap_mode='r');model=AutoModelForSequenceClassification.from_pretrained(str(BEST)).cuda().eval();p=np.zeros(len(tokens),np.float64)
    with torch.inference_mode():
        for lo in range(0,len(tokens),args.batch*2):
            x=torch.as_tensor(np.array(tokens[lo:lo+args.batch*2]),device='cuda',dtype=torch.long)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=x,attention_mask=x.ne(data['pad_token_id'])).logits
            p[lo:lo+len(x)]=logits.float().softmax(-1)[:,1].cpu().numpy()
            if lo%10000<args.batch*2:print('NEURAL_INFER',lo,len(tokens),flush=True)
    np.save(ROOT/'neural_development_p.npy',p)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','infer']);parser.add_argument('--batch',type=int,default=32);parser.add_argument('--epochs',type=int,default=3);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:globals()[args.stage](args)
