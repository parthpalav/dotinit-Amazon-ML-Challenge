"""Audit exact normalized S1 overlap across supplied splits, with collision checks."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from .disk_store import connect,fetch_records
ROOT=Path('work/campaign_0931');OUT=Path('reports/campaign_0931')
def key(row):return '\0'.join(row)
def digest(text):return int.from_bytes(hashlib.blake2b(text.encode(),digest_size=8).digest(),'little')
def main():
 started=time.time();train=connect('work/windows_v1/train/records.sqlite',True);test=connect('work/windows_v1/test/records.sqlite',True);cache=ROOT/'train_anchor_text_hashes.npz'
 if cache.exists():
  z=np.load(cache);keys=z['keys'];rids=z['rids']
 else:
  n=train.execute('select count(*) from anchors').fetchone()[0];keys=np.empty(n,np.uint64);rids=np.empty(n,np.uint32)
  for i,(rid,n,a,c) in enumerate(train.execute('select rid,name_norm,address_norm,country_norm from anchors')):keys[i]=digest(key((n,a,c)));rids[i]=rid
  order=np.argsort(keys);keys=keys[order];rids=rids[order];np.savez(cache,keys=keys,rids=rids);print('TRAIN_HASHED',len(keys),round(time.time()-started,1),flush=True)
 hits=[]
 for rid,n,a,c in test.execute('select rid,name_norm,address_norm,country_norm from anchors'):
  text=key((n,a,c));h=digest(text);p=int(np.searchsorted(keys,np.uint64(h)))
  if p<len(keys) and keys[p]==h:hits.append((rid,int(rids[p]),text,c))
 actual=[]
 for i in range(0,len(hits),500):
  rows=hits[i:i+500];records=fetch_records(train,'anchors',[r[1] for r in rows]);texts={r.rid:key((r.name_norm,r.address_norm,r.country_norm)) for r in records.itertuples(index=False)}
  actual.extend((a,b,c) for a,b,text,c in rows if texts[b]==text)
 from collections import Counter
 report={'exact_normalized_anchor_overlap':len(actual),'by_country':dict(Counter(c for _,_,c in actual)),'hash_hits':len(hits),'seconds':time.time()-started,'note':'Overlap is verified by full normalized name/address/country equality; no test labels are assumed or inferred by this audit.'}
 np.save(ROOT/'train_test_exact_anchor_overlap.npy',np.array([(a,b) for a,b,c in actual],dtype=np.uint32).reshape(-1,2));(OUT/'train_test_overlap.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
