"""Unlabelled reverse reference index and globally competing owner evidence."""
from pathlib import Path
import argparse, hashlib,json,time,gc
import joblib,numpy as np,pandas as pd
from rapidfuzz import process,fuzz
from .disk_store import NativeIndex,ENTRY,WIDTH,COLUMNS,connect,fetch_records
from .disk_blocking import rules_text
from .preprocessing import preprocess
from .evidence import enrich,make_stats
from .config import Config
from .campaign_scoring import predict
ROOT=Path('work/campaign_0942')


def build(split):
    folder=ROOT/('reverse_'+split);folder.mkdir(parents=True,exist_ok=True)
    con=connect(f'work/windows_v1/{split}/records.sqlite',True);n=con.execute('select count(*) from anchors').fetchone()[0]
    sig={'anchors':n,'native':hashlib.sha256(Path('src/native/index.cpp').read_bytes()).hexdigest(),'store':hashlib.sha256(Path(f'work/windows_v1/{split}/store_manifest.json').read_bytes()).hexdigest()}
    manifest=folder/'signature.json'
    if manifest.exists():assert json.loads(manifest.read_text(encoding='utf-8'))==sig
    else:manifest.write_text(json.dumps(sig,indent=2),encoding='utf-8')
    output=folder/'index.bin'
    if (folder/'COMPLETE.json').exists():return output
    native=NativeIndex('work/windows_v1/native');paths=[]
    for start in range(0,n,100000):
        path=folder/f'keys_{start:08d}.npy';paths.append(path)
        if path.exists():continue
        frame=fetch_records(con,'anchors',range(start+1,min(n,start+100000)+1));keys=native.keys(frame).ravel();a=np.empty(len(keys),ENTRY);a['key']=keys;a['row']=np.repeat(frame.rid.to_numpy(),WIDTH);a=a[keys!=0]
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.save(f,a)
        tmp.replace(path);print('REVERSE_KEYS',split,start+n*0,flush=True)
    with output.with_suffix('.partial').open('wb') as stream:
        for bucket in range(16):
            chunks=[]
            for path in paths:
                a=np.load(path,mmap_mode='r');chunks.append(a[(a['key']>>60)==bucket])
            a=np.concatenate(chunks);a.sort(order=['key','row']);keep=np.r_[True,(a['key'][1:]!=a['key'][:-1])|(a['row'][1:]!=a['row'][:-1])];a[keep].tofile(stream)
            print('REVERSE_SORT',split,bucket,flush=True)
    output.with_suffix('.partial').replace(output);(folder/'COMPLETE.json').write_text(json.dumps(sig),encoding='utf-8');con.close();native.close();return output


class Reverse:
    def __init__(self,split):
        self.con=connect(f'work/windows_v1/{split}/records.sqlite',True)
        self.native=NativeIndex('work/windows_v1/native',ROOT/f'reverse_{split}/index.bin')
        from .sorted_lookup import SortedLookup
        self.native.lookup=SortedLookup(self.native)

    def candidates(self,targets,topk=4):
        rr,mm,oo=self.native.lookup(self.native.keys(targets),240)
        anc=fetch_records(self.con,'anchors',np.unique(rr));ai=np.searchsorted(anc.rid.to_numpy(),rr);ti=np.repeat(np.arange(len(targets)),np.diff(oo))
        if not len(rr):return pd.DataFrame(columns=['anchor_rid','target_rid','source1_entity_id','candidate_entity_id','blocking_rules'])
        ns=process.cpdist(anc.name_norm.to_numpy()[ai],targets.name_norm.to_numpy()[ti],scorer=fuzz.ratio,dtype=np.float32,workers=1)/100
        ads=process.cpdist(anc.address_norm.to_numpy()[ai],targets.address_norm.to_numpy()[ti],scorer=fuzz.token_sort_ratio,dtype=np.float32,workers=1)/100
        missing=(anc.address_norm.to_numpy()[ai]=='')|(targets.address_norm.to_numpy()[ti]=='');ads[missing]=0
        rank=np.maximum(ns,ads)+.6*np.minimum(ns,ads);rank[anc.country_norm.to_numpy()[ai]!=targets.country_norm.to_numpy()[ti]]=-1
        selected=[]
        for lo,hi in zip(oo[:-1],oo[1:]):
            ii=np.arange(lo,hi);ii=ii[rank[ii]>=0];selected.extend(ii[np.argsort(-rank[ii],kind='stable')[:topk]].tolist())
        k=np.array(selected,dtype=int)
        return pd.DataFrame({'anchor_rid':rr[k],'target_rid':targets.rid.to_numpy()[ti[k]],'source1_entity_id':anc.entity_id.to_numpy()[ai[k]],'candidate_entity_id':targets.entity_id.to_numpy()[ti[k]],'blocking_rules':[rules_text(x) for x in mm[k]],'reverse_name':ns[k],'reverse_address':ads[k],'reverse_strength':rank[k]})

def prepare():
    rev=Reverse('train')
    for split in ['selection','v3','v4']:
        data=joblib.load(ROOT/(split+'_raw.joblib'));pairs=data['part']['pairs'].iloc[data['rows']];folder=ROOT/('competitors_'+split);folder.mkdir(exist_ok=True)
        ids=np.sort(pairs.target_rid.unique());blocks=[]
        for start in range(0,len(ids),400):
            path=folder/f'{start:07d}.joblib'
            if not path.exists():
                t=fetch_records(rev.con,'targets',ids[start:start+400]);b=rev.candidates(t)
                tmp=path.with_suffix('.partial');joblib.dump(b,tmp);tmp.replace(path)
            blocks.append(joblib.load(path))
            if start%4000==0:print('REVERSE_SCORED',split,start,len(ids),flush=True)
        b=pd.concat(blocks,ignore_index=True);joblib.dump(b,ROOT/(split+'_competitors.joblib'));del data,pairs,b,blocks;gc.collect()


def evidence(pairs,competitors,con):
    """Compare each proposed owner with other unlabelled references, excluding itself."""
    a=fetch_records(con,'anchors',pairs.anchor_rid.unique());t=fetch_records(con,'targets',pairs.target_rid.unique())
    ai=np.searchsorted(a.rid.to_numpy(),pairs.anchor_rid);ti=np.searchsorted(t.rid.to_numpy(),pairs.target_rid)
    ns=process.cpdist(a.name_norm.to_numpy()[ai],t.name_norm.to_numpy()[ti],scorer=fuzz.ratio,dtype=np.float32,workers=1)/100
    ads=process.cpdist(a.address_norm.to_numpy()[ai],t.address_norm.to_numpy()[ti],scorer=fuzz.token_sort_ratio,dtype=np.float32,workers=1)/100
    ads[(a.address_norm.to_numpy()[ai]=='')|(t.address_norm.to_numpy()[ti]=='')]=0
    strength=np.maximum(ns,ads)+.6*np.minimum(ns,ads)
    groups={int(k):v for k,v in competitors.groupby('target_rid',sort=False)};rows=[]
    for i,r in enumerate(pairs.itertuples(index=False)):
        b=groups.get(int(r.target_rid));b=b[b.anchor_rid!=r.anchor_rid] if b is not None else None
        if b is None or not len(b):rows.append([0,0,0,0,float(strength[i]),float(ns[i]),float(ads[i])]);continue
        winner=b.iloc[int(np.argmax(b.reverse_strength.to_numpy()))];best=float(winner.reverse_strength)
        rows.append([len(b),best,float(winner.reverse_name),float(winner.reverse_address),float(strength[i]-best),float(ns[i]-b.reverse_name.max()),float(ads[i]-b.reverse_address.max())])
    return pd.DataFrame(rows,index=pairs.index,columns=['reverse_count','reverse_best_strength','reverse_best_name','reverse_best_address','reverse_strength_margin','reverse_name_margin','reverse_address_margin'],dtype=np.float32)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['index','prepare']);p.add_argument('--split',default='train',choices=['train','test']);args=p.parse_args()
    if args.stage=='index':build(args.split)
    else:prepare()
