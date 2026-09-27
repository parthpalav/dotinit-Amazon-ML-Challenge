"""Freeze neural/joint recipes, then compare on a new untouched 5k anchors."""
from pathlib import Path
from dataclasses import replace
import gc,json
import joblib,numpy as np,pandas as pd
from .campaign_raw import ROOT,OUT,ART,GATE
from .campaign_confirm_raw import scores
from .config import Config
from .real_pipeline import initialize_worker,process_batch,_WORKER,aggregate,enrich_stats
from .sorted_lookup import SortedLookup
from .disk_store import connect,fetch_records
from .evidence import make_stats,enrich
from .context_experiments import context
from .peer_evidence import features as peers
from .detail_evidence import features as details
from .raw_evidence import features as raw
from .campaign_reverse import Reverse,evidence as reverse_evidence
from .campaign_scoring import predict
from .rescoring import filehash,sourcehash
from .experiments import evaluate


def freeze():
    cfg=Config.load('config/windows.json');s=np.load(Path(cfg.working_dir)/'entity_selection.npz')
    used=np.unique(np.concatenate([*[s[k] for k in s.files],np.load('work/improvements/confirmation_ids.npy'),np.load('work/campaign_0931/confirmation_v3_ids.npy'),np.load('work/campaign_0931/confirmation_v4_ids.npy'),np.load(ROOT/'confirmation_v5_ids.npy')]))
    ids=np.sort(np.random.default_rng(20260930).choice(np.setdiff1d(np.arange(1,2206822,dtype=np.uint32),used),5000,replace=False));np.save(ROOT/'confirmation_v6_ids.npy',ids)
    recipes=[]
    for report,name in [('neural_selection','neural_stack_d3'),('joint_selection','joint_stack_d3')]:
        r=next(r for r in json.loads((OUT/(report+'.json')).read_text(encoding='utf-8'))['experiments'] if r['model']==name);recipes.append(r)
    paths=['artifacts/improvements/catboost_d10_evidence.joblib','artifacts/campaign_0931/oof_compact_d9.joblib',*[str(ART/(n+'.joblib')) for n in ['raw_minimal_d7','raw_reverse_d7','neural_stack_d3','joint_stack_d3']],*[str(ART/'neural_minilm'/n) for n in ['model.safetensors','config.json','sentencepiece.bpe.model','special_tokens_map.json','tokenizer_config.json']]]
    f={'ids_hash':filehash(ROOT/'confirmation_v6_ids.npy'),'excluded_anchors':len(used),'seed':20260930,'recipes':recipes,'models':{p:filehash(p) for p in paths},'code':{p:sourcehash('src/'+p) for p in ['raw_evidence.py','campaign_reverse.py']},'neural_batch':64,'neural_precision':'bfloat16','baseline_threshold':.7250000000000003,'rule':'Promote neural only if paired 97.5% interval versus raw is positive. Prefer joint only if its 97.5% interval versus raw and 95% interval versus neural are positive.'}
    path=OUT/'frozen_v6.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==f
    else:path.write_text(json.dumps(f,indent=2),encoding='utf-8')
    return ids,f


def prepare(ids):
    dest=ROOT/'v6_base.joblib'
    if dest.exists():return joblib.load(dest)
    cfg=replace(Config.load('config/windows.json'),workers=1);folder=ROOT/'v6_base_batches';folder.mkdir(exist_ok=True);initialize_worker(cfg.to_dict(),'train','work/windows_v1/feature_engineer.joblib')
    blocker=_WORKER['blocker'];blocker.native.lookup=SortedLookup(blocker.native)
    if blocker.extra:blocker.extra.lookup=SortedLookup(blocker.extra)
    chunks=[];stats={};truth={}
    for start in range(0,len(ids),100):
        path=folder/f'{start:05d}.joblib'
        if not path.exists():
            result=process_batch(ids[start:start+100]);tmp=path.with_suffix('.partial');joblib.dump(result,tmp);tmp.replace(path)
        d=joblib.load(path);chunks.append(d);aggregate(stats,d[2]);truth.update(d[3]);print('V6_BASE',start+100,len(ids),flush=True)
    part={'pairs':pd.concat([d[0] for d in chunks],ignore_index=True),'features':pd.concat([d[1] for d in chunks],ignore_index=True),'truth_counts':truth,'stats':enrich_stats(stats,10320219)}
    tmp=dest.with_suffix('.partial');joblib.dump(part,tmp);tmp.replace(dest);blocker.close();_WORKER.clear();del blocker,chunks;gc.collect();return part


def main():
    ids,frozen=freeze();part=prepare(ids)
    import torch
    from transformers import AutoModelForSequenceClassification,XLMRobertaTokenizer
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);network=AutoModelForSequenceClassification.from_pretrained(str(ART/'neural_minilm')).cuda().eval();tokenizer=XLMRobertaTokenizer.from_pretrained(str(ART/'neural_minilm'))
    con=connect('work/windows_v1/train/records.sqlite',True);stats=make_stats(Config.load('config/windows.json'),'train');rev=Reverse('train')
    base=joblib.load('artifacts/improvements/catboost_d10_evidence.joblib');meta=joblib.load('artifacts/campaign_0931/oof_compact_d9.joblib');minimal=joblib.load(ART/'raw_minimal_d7.joblib');reverse=joblib.load(ART/'raw_reverse_d7.joblib');stacks={r['model']:joblib.load(ART/(r['model']+'.joblib')) for r in frozen['recipes']}
    pairs=part['pairs'];a=pairs.anchor_rid.to_numpy();starts=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,len(a)];folder=ROOT/'v6_predictions';folder.mkdir(exist_ok=True);blocks=[]
    for g in range(0,len(starts)-1,100):
        path=folder/f'{g:05d}.joblib';lo=starts[g];hi=starts[min(g+100,len(starts)-1)]
        if not path.exists():
            b=pairs.iloc[lo:hi];X=pd.concat([part['features'].iloc[lo:hi],enrich(b,con,stats)],axis=1);bp=predict(base,X);X=pd.concat([context(b,bp),peers(b,bp,con),details(b,con,stats)],axis=1);p=predict(meta,X);mask=(p>=GATE[0])&(p<GATE[1]);result={'raw':p.copy(),**{n:p.copy() for n in stacks}}
            if mask.any():
                small=b.loc[mask];rx=raw(small,con);rx['current_probability']=p[mask];rp=predict(minimal,rx[minimal['feature_names']]);result['raw'][mask]=rp
                targets=fetch_records(con,'targets',small.target_rid.unique());comp=rev.candidates(targets);cx=reverse_evidence(small,comp,con);xx=pd.concat([rx,cx],axis=1);vp=predict(reverse,xx[reverse['feature_names']])
                anc=fetch_records(con,'anchors',small.anchor_rid.unique())
                def serial(r):return 'name: '+str(r.business_name)+'; address: '+str(r.business_address)+'; country: '+str(r.country)
                aa={r.rid:serial(r) for r in anc.itertuples(index=False)};tt={r.rid:serial(r) for r in targets.itertuples(index=False)}
                tokens=tokenizer([aa[r] for r in small.anchor_rid],[tt[r] for r in small.target_rid],max_length=192,truncation=True,padding='max_length',return_tensors='np')['input_ids'];npred=np.zeros(len(tokens))
                with torch.inference_mode():
                    for i in range(0,len(tokens),64):
                        x=torch.as_tensor(tokens[i:i+64],device='cuda',dtype=torch.long)
                        with torch.autocast('cuda',dtype=torch.bfloat16):logits=network(input_ids=x,attention_mask=x.ne(tokenizer.pad_token_id)).logits
                        npred[i:i+len(x)]=logits.float().softmax(-1)[:,1].cpu().numpy()
                def logit(q):q=np.clip(q,1e-6,1-1e-6);return np.log(q/(1-q))
                sx=pd.DataFrame({'current':logit(p[mask]),'raw':logit(rp),'neural':logit(npred),'reverse':logit(vp)})
                for n,m in stacks.items():result[n][mask]=m['estimator'].predict_proba(sx[m['features']],thread_count=1)[:,1]
            tmp=path.with_suffix('.partial');joblib.dump({'lo':int(lo),'hi':int(hi),'predictions':result},tmp);tmp.replace(path)
        d=joblib.load(path);assert (d['lo'],d['hi'])==(lo,hi);blocks.append(d['predictions']);print('V6_PREDICT',g,flush=True)
    predictions={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};thresholds={'raw':frozen['baseline_threshold'],**{r['model']:r['threshold'] for r in frozen['recipes']}};values={k:scores(part,p,thresholds[k]) for k,p in predictions.items()};report={'metrics':{k:evaluate(part,p,thresholds[k]) for k,p in predictions.items()},'comparisons':{},'frozen':frozen}
    for k,p in predictions.items():np.save(ROOT/('v6_'+k+'_p.npy'),p)
    for name,left,right,alpha in [('neural_vs_raw','neural_stack_d3','raw',.0125),('joint_vs_raw','joint_stack_d3','raw',.0125),('joint_vs_neural','joint_stack_d3','neural_stack_d3',.025)]:
        delta=values[left]-values[right];rng=np.random.default_rng(2942);boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(4000)]);report['comparisons'][name]={'delta':float(delta.mean()),'ci':np.quantile(boot,[alpha,1-alpha]).tolist()}
    comp=report['comparisons'];chosen=None
    if comp['neural_vs_raw']['ci'][0]>0:chosen='neural_stack_d3'
    if comp['joint_vs_raw']['ci'][0]>0 and comp['joint_vs_neural']['ci'][0]>0:chosen='joint_stack_d3'
    report['chosen']=chosen;report['promotion_supported']=chosen is not None;(OUT/'confirmation_v6.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('V6_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
