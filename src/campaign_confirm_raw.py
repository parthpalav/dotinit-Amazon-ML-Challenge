"""Fresh, frozen confirmation; cache atomic batches for interruption recovery."""
from pathlib import Path
from dataclasses import replace
import gc, json, logging
import joblib,numpy as np,pandas as pd
from .campaign_raw import ROOT,OUT,ART,GATE
from .config import Config
from .real_pipeline import feature_subset
from .real_pipeline import initialize_worker,process_batch,_WORKER,aggregate,enrich_stats
from .disk_store import connect
from .evidence import make_stats,enrich
from .context_experiments import context
from .peer_evidence import features as peers
from .detail_evidence import features as details
from .raw_evidence import features as raw
from .campaign_scoring import predict
from .rescoring import filehash
from .experiments import evaluate
from .sorted_lookup import SortedLookup


def scores(part,p,t):
    truth=part['truth_counts'];pos={k:i for i,k in enumerate(truth)}
    ix=np.array([pos[x] for x in part['pairs'].source1_entity_id]);keep=p>=t
    tp=np.bincount(ix,weights=keep*part['pairs'].label.to_numpy(),minlength=len(truth))
    pred=np.bincount(ix,weights=keep,minlength=len(truth));den=.25*np.array(list(truth.values()))+pred
    return np.divide(1.25*tp,den,out=np.ones(len(truth)),where=den>0)


def main():
    logging.basicConfig(level=logging.INFO)
    cfg=replace(Config.load('config/windows.json'),workers=1,reports_dir=str(OUT))
    sel=np.load(Path(cfg.working_dir)/'entity_selection.npz')
    used=np.unique(np.concatenate([*[sel[k] for k in sel.files],np.load('work/improvements/confirmation_ids.npy'),np.load('work/campaign_0931/confirmation_v3_ids.npy'),np.load('work/campaign_0931/confirmation_v4_ids.npy')]))
    eligible=np.setdiff1d(np.arange(1,2206822,dtype=np.uint32),used)
    ids=np.sort(np.random.default_rng(20260929).choice(eligible,5000,replace=False));np.save(ROOT/'confirmation_v5_ids.npy',ids)
    recipe=max(json.loads((OUT/'raw_selection.json').read_text(encoding='utf-8'))['experiments'],key=lambda r:r['f0.5'])
    mp=ART/(recipe['model']+'.joblib')
    freeze={'recipe':recipe,'model_hash':filehash(mp),'gate':list(GATE),'ids_hash':filehash(ROOT/'confirmation_v5_ids.npy'),'excluded_anchors':len(used),
            'feature_hash':filehash('src/raw_evidence.py'),'main_hash':filehash('artifacts/campaign_0931/oof_compact_d9.joblib')}
    fp=OUT/'raw_frozen_v5.json'
    if fp.exists():assert json.loads(fp.read_text(encoding='utf-8'))==freeze
    else:fp.write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    destination=ROOT/'v5_base.joblib'
    if destination.exists():part=joblib.load(destination)
    else:
        folder=ROOT/'v5_base_batches';folder.mkdir(exist_ok=True)
        initialize_worker(cfg.to_dict(),'train',str(Path(cfg.working_dir)/'feature_engineer.joblib'))
        blocker=_WORKER['blocker'];blocker.native.lookup=SortedLookup(blocker.native)
        if blocker.extra:blocker.extra.lookup=SortedLookup(blocker.extra)
        chunks=[];stats={};truth={}
        for start in range(0,len(ids),100):
            path=folder/f'{start:05d}.joblib'
            if not path.exists():
                result=process_batch(ids[start:start+100]);tmp=path.with_suffix('.partial');joblib.dump(result,tmp);tmp.replace(path)
            result=joblib.load(path);chunks.append(result);aggregate(stats,result[2]);truth.update(result[3]);print('RAW_CONFIRM_BASE',start+100,len(ids),flush=True)
        part={'pairs':pd.concat([d[0] for d in chunks],ignore_index=True),'features':pd.concat([d[1] for d in chunks],ignore_index=True),'truth_counts':truth,'stats':enrich_stats(stats,10320219)}
        tmp=destination.with_suffix('.partial');joblib.dump(part,tmp);tmp.replace(destination);del chunks
        _WORKER['blocker'].close();_WORKER.clear();gc.collect()
    con=connect(Path(cfg.working_dir)/'train/records.sqlite',True);stats=make_stats(cfg,'train');pairs=part['pairs']
    base=joblib.load('artifacts/improvements/catboost_d10_evidence.joblib');meta=joblib.load('artifacts/campaign_0931/oof_compact_d9.joblib');residual=joblib.load(mp)
    folder=ROOT/'v5_parts';folder.mkdir(exist_ok=True);anchors=pairs.anchor_rid.to_numpy();starts=np.r_[0,np.flatnonzero(anchors[1:]!=anchors[:-1])+1,len(anchors)]
    pp=[];qq=[]
    for g in range(0,len(starts)-1,100):
        path=folder/f'{g:05d}.joblib';lo=starts[g];hi=starts[min(g+100,len(starts)-1)]
        if not path.exists():
            b=pairs.iloc[lo:hi];X=pd.concat([part['features'].iloc[lo:hi],enrich(b,con,stats)],axis=1);bp=predict(base,X)
            X=pd.concat([context(b,bp),peers(b,bp,con),details(b,con,stats)],axis=1);p=predict(meta,X);mask=(p>=GATE[0])&(p<GATE[1]);q=p.copy()
            if mask.any():
                xx=X.loc[mask].copy();xx['current_probability']=p[mask];xx=pd.concat([xx,raw(b.loc[mask],con)],axis=1);q[mask]=predict(residual,xx)
            q=p*(1-recipe['weight'])+q*recipe['weight'];tmp=path.with_suffix('.partial');joblib.dump({'p':p,'q':q,'lo':int(lo),'hi':int(hi)},tmp);tmp.replace(path)
        d=joblib.load(path);assert (d['lo'],d['hi'])==(lo,hi);pp.append(d['p']);qq.append(d['q']);print('RAW_CONFIRM',g,flush=True)
    p=np.concatenate(pp);q=np.concatenate(qq);np.save(ROOT/'v5_main_p.npy',p);np.save(ROOT/'v5_raw_p.npy',q)
    a=scores(part,p,.6000000000000002);b=scores(part,q,recipe['threshold']);delta=b-a;rng=np.random.default_rng(1942)
    boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(3000)]);ci=np.quantile(boot,[.025,.975]).tolist()
    report={'baseline':evaluate(part,p,.6000000000000002),'raw':evaluate(part,q,recipe['threshold']),'delta':float(delta.mean()),'paired_95ci':ci,'promotion_supported':bool(ci[0]>0),'frozen':freeze}
    (OUT/'raw_confirmation_v5.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('RAW_CONFIRMATION',json.dumps(report),flush=True);con.close()

if __name__=='__main__':main()
