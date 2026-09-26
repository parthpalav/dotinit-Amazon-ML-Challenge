"""Additional identity evidence derived solely from supplied text.

Corpus statistics use unlabeled S1 records for each split (transductive,
country-scoped); no identities, websites or geocoding are looked up.
"""
import argparse,collections,hashlib,json,math,re,time,unicodedata
from pathlib import Path
import joblib,numpy as np,pandas as pd
from rapidfuzz.fuzz import ratio,token_sort_ratio,partial_ratio
from .disk_store import connect,fetch_records

VERSION=1
GENERIC=set('and the of company corporation group services international inc ltd limited private pvt llc llp corp co sarl sas sasu sci eurl sa association'.split())

def fold(text):
    # Accent folding is an additional view. Original scripts are never replaced.
    return ''.join(c for c in unicodedata.normalize('NFKD',text) if not unicodedata.combining(c))

def tokens(text):return set(fold(text).split())

def components(name,address):
    n=fold(name);a=fold(address);nt=set(n.split());at=set(a.split())
    numbers=set(re.findall(r'\d+',a)); numbers={str(int(v)) for v in numbers}
    small={x for x in numbers if len(x)<5};large=numbers-small
    words={x for x in at if not any(c.isdigit() for c in x)}
    return n,a,nt,at,nt-GENERIC,numbers,small,large,words

def make_stats(config,split):
    path=Path('work/improvements')/('evidence_stats_'+split+'.joblib');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return joblib.load(path)
    con=connect(Path(config.working_dir)/split/'records.sqlite',True)
    counts=collections.Counter();df=collections.Counter();sizes=collections.Counter()
    for country,name in con.execute('SELECT country_norm,name_norm FROM anchors'):
        f=fold(name);counts[(country,f)]+=1;sizes[country]+=1
        for token in set(f.split()):df[(country,token)]+=1
    con.close()
    result={'version':VERSION,'counts':{k:v for k,v in counts.items() if v>1},'df':dict(df),'sizes':dict(sizes)}
    joblib.dump(result,path);return result

def enrich(pairs,con,stats):
    anchors=fetch_records(con,'anchors',pairs.anchor_rid.unique());targets=fetch_records(con,'targets',pairs.target_rid.unique())
    records={}
    for row in pd.concat([anchors,targets]).itertuples(index=False):
        records[row.entity_id]=(components(row.name_norm,row.address_norm),row.country_norm)
    rows=[]
    for p in pairs.itertuples(index=False):
        (n,a,nt,at,core,nums,small,large,words),country=records[p.source1_entity_id]
        (m,b,mt,bt,mcore,mnums,msmall,mlarge,mwords),other=records[p.candidate_entity_id]
        d={}
        for field,left,right in [('name',nt,mt),('core',core,mcore),('address',at,bt),('address_words',words,mwords),('numbers',nums,mnums),('small_numbers',small,msmall),('large_numbers',large,mlarge)]:
            overlap=len(left&right);union=len(left|right)
            d[field+'_intersection_v2']=overlap;d[field+'_left_coverage_v2']=overlap/max(1,len(left));d[field+'_right_coverage_v2']=overlap/max(1,len(right))
            d[field+'_conflict_v2']=int(bool(left and right) and not overlap)
            d[field+'_left_only_v2']=len(left-right);d[field+'_right_only_v2']=len(right-left)
        d['name_fold_ratio_v2']=ratio(n,m)/100 if n and m else 0
        d['name_fold_token_sort_v2']=token_sort_ratio(n,m)/100 if n and m else 0
        d['core_name_ratio_v2']=ratio(' '.join(sorted(core)),' '.join(sorted(mcore)))/100 if core and mcore else 0
        d['address_fold_ratio_v2']=ratio(a,b)/100 if a and b else 0
        d['address_partial_ratio_v2']=partial_ratio(a,b)/100 if a and b else 0
        d['name_anchor_frequency_v2']=math.log1p(stats['counts'].get((country,n),1))
        d['name_target_frequency_v2']=math.log1p(stats['counts'].get((other,m),1))
        weights={t:math.log1p(stats['sizes'].get(country,1)/(1+stats['df'].get((country,t),0))) for t in nt|mt}
        d['name_idf_jaccard_v2']=sum(weights[t] for t in nt&mt)/max(1e-9,sum(weights.values()))
        d['name_idf_left_coverage_v2']=sum(weights[t] for t in nt&mt)/max(1e-9,sum(weights[t] for t in nt))
        d['name_shared_max_idf_v2']=max((weights[t] for t in nt&mt),default=0)
        d['name_unshared_max_idf_v2']=max((weights[t] for t in nt^mt),default=0)
        d['name_nonlatin_left_v2']=float(any(ord(c)>591 for c in n));d['name_nonlatin_right_v2']=float(any(ord(c)>591 for c in m))
        rows.append(d)
    return pd.DataFrame(rows,dtype=np.float32,index=pairs.index)

def prepare(args):
    from .config import Config
    cfg=Config.load('config/windows.json');stats=make_stats(cfg,'train')
    con=connect(Path(cfg.working_dir)/'train/records.sqlite',True)
    for split in args.splits:
        part=joblib.load(Path(cfg.working_dir)/'features'/(split+'.joblib'));pairs=part['pairs'];blocks=[];started=time.time()
        for i in range(0,len(pairs),10000):
            blocks.append(enrich(pairs.iloc[i:i+10000],con,stats))
            if i%100000==0:print(split,i,len(pairs),round(time.time()-started,1),flush=True)
        features=pd.concat(blocks).reset_index(drop=True)
        dest=Path('work/improvements')/(split+'_extra.joblib')
        joblib.dump({'version':VERSION,'pair_ids':pairs[['source1_entity_id','candidate_entity_id']],'features':features},dest)
        print('SAVED',split,features.shape,flush=True)
    con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--splits',nargs='+',default=['fit','calibration','validation']);prepare(p.parse_args())
