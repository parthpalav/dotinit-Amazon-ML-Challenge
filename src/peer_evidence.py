"""Evidence between candidates for the same anchor, without using peer labels."""
from pathlib import Path
import argparse,gc,hashlib,json,time
import joblib,numpy as np,pandas as pd
from rapidfuzz.fuzz import ratio,token_sort_ratio,token_set_ratio
from .disk_store import connect,fetch_records
from .evidence import components
VERSION=1
NAMES=['peer_name_ratio','peer_name_sort','peer_core_ratio','peer_address_ratio','peer_address_set','peer_address_words','peer_numbers_jaccard','peer_number_equal','peer_exact_name','peer_exact_address','peer_joint','peer_name_with_address','peer_address_with_name','peer_name_weighted','peer_address_weighted','peer_joint_weighted','peer_same_source_joint','peer_other_source_joint','peer_strong_count','peer_support_count','peer_number_contradiction_count','peer_max_probability']

def features(pairs,prob,con):
 ids=pairs.target_rid.to_numpy();targets=fetch_records(con,'targets',ids)
 texts={r.rid:components(r.name_norm,r.address_norm) for r in targets.itertuples(index=False)}
 source=np.array([s.startswith('S3-') for s in pairs.candidate_entity_id]);anchor=pairs.anchor_rid.to_numpy();starts=np.r_[0,np.flatnonzero(anchor[1:]!=anchor[:-1])+1,len(anchor)]
 out=np.zeros((len(pairs),len(NAMES)),np.float32)
 for lo,hi in zip(starts[:-1],starts[1:]):
  peers=np.arange(lo,hi)[np.argsort(-prob[lo:hi],kind='stable')[:6]];peers=peers[prob[peers]>=.15]
  for i in range(lo,hi):
   n,a,nt,at,core,nums,small,large,words=texts[ids[i]]
   for j in peers:
    if i==j:continue
    m,b,mt,bt,mc,mnums,ms,ml,mwords=texts[ids[j]];p=prob[j]
    nr=ratio(n,m)/100 if n and m else 0;ns=token_sort_ratio(n,m)/100 if n and m else 0;cr=token_sort_ratio(' '.join(core),' '.join(mc))/100 if core and mc else 0
    ar=ratio(a,b)/100 if a and b else 0;aset=token_set_ratio(a,b)/100 if a and b else 0;aw=token_sort_ratio(' '.join(words),' '.join(mwords))/100 if words and mwords else 0
    nj=len(nums&mnums)/max(1,len(nums|mnums));ne=float(bool(nums) and nums==mnums);en=float(bool(n) and n==m);ea=float(bool(a) and a==b)
    joint=min(ns,ar) if a and b else ns*.8
    value=[nr,ns,cr,ar,aset,aw,nj,ne,en,ea,joint,ns if ar>.85 else 0,ar if ns>.85 else 0,p*ns,p*ar,p*joint,p*joint if source[i]==source[j] else 0,p*joint if source[i]!=source[j] else 0]
    out[i,:18]=np.maximum(out[i,:18],value)
    out[i,18]+=p>=.9;out[i,19]+=p>=.9 and joint>=.85;out[i,20]+=p>=.9 and bool(nums and mnums) and bool(nums-mnums) and bool(mnums-nums);out[i,21]=max(out[i,21],p)
 return pd.DataFrame(out,columns=NAMES,index=pairs.index)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--splits',nargs='+',default=['calibration','validation']);args=parser.parse_args()
 con=connect('work/windows_v1/train/records.sqlite',True);root=Path('work/campaign_0931');root.mkdir(exist_ok=True)
 for split in args.splits:
  location='work/windows_v1/features' if split=='confirmation' else 'work/real_v1/features';part=joblib.load(Path(location)/(split+'.joblib'));pairs=part['pairs'];del part;gc.collect()
  p=np.load(root/(split+'_champion_p.npy'));signature={'version':VERSION,'pairs':hashlib.sha256(pairs[['anchor_rid','target_rid']].to_numpy(dtype='<u4').tobytes()).hexdigest(),'probability':hashlib.sha256(p.tobytes()).hexdigest()}
  destination=root/('peer_'+split+'.joblib')
  if destination.exists():
   if joblib.load(destination)['signature']!=signature:raise ValueError('Peer cache mismatch')
   continue
  anchor=pairs.anchor_rid.to_numpy();starts=np.r_[0,np.flatnonzero(anchor[1:]!=anchor[:-1])+1,len(anchor)];blocks=[];started=time.time()
  for g in range(0,len(starts)-1,250):
   lo=starts[g];hi=starts[min(g+250,len(starts)-1)];blocks.append(features(pairs.iloc[lo:hi],p[lo:hi],con))
   if g%1000==0:print('PEERS',split,g,round(time.time()-started,1),flush=True)
  joblib.dump({'signature':signature,'features':pd.concat(blocks)},destination);print('SAVED',split,flush=True)
 con.close()

if __name__=='__main__':main()
