"""Offline script-independent text and ordered-number evidence."""
import argparse,gc,hashlib,json,re,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from anyascii import anyascii
from rapidfuzz.fuzz import ratio,token_sort_ratio,token_set_ratio,partial_ratio
from .disk_store import connect,fetch_records
from .evidence import GENERIC,make_stats
from .config import Config
VERSION=1

def skeleton(s):
 s=anyascii(s).lower()
 for a,b in [('ph','f'),('bh','b'),('dh','d'),('th','t'),('kh','k'),('sh','s'),('w','v')]:s=s.replace(a,b)
 s=re.sub('[aeiou]','',s);return re.sub(r'(.)\1+',r'\1',s)
STOP_SKEL={skeleton(t) for t in GENERIC}

def record(row):
 n=anyascii(row.name_norm).lower();a=anyascii(row.address_norm).lower();nt=n.split();core=[t for t in nt if t not in GENERIC];nums=[str(int(t)) for t in re.findall(r'\d+',a)]
 phon=' '.join(skeleton(t) for t in nt);pcore=' '.join(skeleton(t) for t in core if skeleton(t) not in STOP_SKEL)
 return n,a,nt,core,nums,phon,pcore,row.country_norm

def sim(a,b,fn=ratio):return fn(a,b)/100 if a and b else 0.

def features(pairs,con,stats):
 anchors=fetch_records(con,'anchors',pairs.anchor_rid.unique());targets=fetch_records(con,'targets',pairs.target_rid.unique());texts={r.entity_id:record(r) for r in pd.concat([anchors,targets]).itertuples(index=False)};rows=[]
 for row in pairs.itertuples(index=False):
  n,a,nt,core,nums,ph,pc,country=texts[row.source1_entity_id];m,b,mt,mc,mnums,mph,mpc,other=texts[row.candidate_entity_id];d={}
  for prefix,x,y in [('ascii_name',n,m),('ascii_core',' '.join(core),' '.join(mc)),('phonetic_name',ph,mph),('phonetic_core',pc,mpc),('ascii_address',a,b)]:
   d[prefix+'_ratio']=sim(x,y);d[prefix+'_sort']=sim(x,y,token_sort_ratio);d[prefix+'_set']=sim(x,y,token_set_ratio)
  d['ascii_name_partial']=sim(n,m,partial_ratio);d['ascii_address_partial']=sim(a,b,partial_ratio)
  left=sorted(core,key=lambda t:(-len(t),t))[:12];right=sorted(mc,key=lambda t:(-len(t),t))[:12]
  soft=np.array([[ratio(x,y)/100 for y in right] for x in left],dtype=np.float32) if left and right else np.zeros((len(left),len(right)))
  ls=soft.max(axis=1) if left and right else np.zeros(len(left));rs=soft.max(axis=0) if left and right else np.zeros(len(right))
  for prefix,values,tokens in [('left',ls,left),('right',rs,right)]:
   d['soft_'+prefix+'_mean']=float(values.mean()) if len(values) else 0;d['soft_'+prefix+'_min']=float(values.min()) if len(values) else 0;d['soft_'+prefix+'_unmatched']=int((values<.65).sum());d['soft_'+prefix+'_weighted']=sum(len(t)*v for t,v in zip(tokens,values))/max(1,sum(map(len,tokens)))
  d['number_sequence_ratio']=sim(' '.join(nums),' '.join(mnums));d['number_sorted_ratio']=sim(' '.join(sorted(nums)),' '.join(sorted(mnums)));d['number_first_equal']=int(bool(nums and mnums) and nums[0]==mnums[0]);d['number_last_equal']=int(bool(nums and mnums) and nums[-1]==mnums[-1]);d['number_left_count']=len(nums);d['number_right_count']=len(mnums)
  na=set(nums);nb=set(mnums);d['number_left_only']=len(na-nb);d['number_right_only']=len(nb-na);d['number_jaccard']=len(na&nb)/max(1,len(na|nb));d['number_subset']=int(bool(na and nb) and (na<=nb or nb<=na))
  d['left_name_length']=len(n);d['right_name_length']=len(m);d['left_address_length']=len(a);d['right_address_length']=len(b);d['left_core_count']=len(core);d['right_core_count']=len(mc)
  d['left_name_frequency']=np.log1p(stats['counts'].get((country,n),1));d['right_name_frequency']=np.log1p(stats['counts'].get((other,m),1));d['source3']=int(row.candidate_entity_id.startswith('S3'))
  rows.append(d)
 return pd.DataFrame(rows,index=pairs.index,dtype=np.float32).add_prefix('detail_')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--splits',nargs='+',default=['calibration','validation']);args=parser.parse_args();cfg=Config.load('config/windows.json');stats=make_stats(cfg,'train');con=connect(Path(cfg.working_dir)/'train/records.sqlite',True);root=Path('work/campaign_0931')
 for split in args.splits:
  base=Path('work/real_v1/features') if split in ['fit','calibration','validation'] else Path(cfg.working_dir)/'features';part=joblib.load(base/(split+'.joblib'));pairs=part['pairs'];del part;gc.collect();blocks=[];started=time.time()
  signature={'version':VERSION,'pairs':hashlib.sha256(pairs[['anchor_rid','target_rid']].to_numpy(dtype='<u4').tobytes()).hexdigest()}
  for lo in range(0,len(pairs),10000):
   blocks.append(features(pairs.iloc[lo:lo+10000],con,stats))
   if lo%100000==0:print('DETAIL',split,lo,round(time.time()-started,1),flush=True)
  joblib.dump({'signature':signature,'features':pd.concat(blocks)},root/('detail_'+split+'.joblib'));print('SAVED',split,flush=True)
 con.close()

if __name__=='__main__':main()
