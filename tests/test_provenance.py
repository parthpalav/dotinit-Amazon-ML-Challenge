"""Exact provenance replay must honor each rule's posting limit."""
import numpy as np,pandas as pd
from src.provenance import PairProvenance,frequent_keys
from src.disk_store import NativeIndex,ENTRY
from src.preprocessing import preprocess

def test_frequent_keys_spanning_chunk_boundary(tmp_path):
 a=np.zeros(1000004,dtype=ENTRY);a['key'][:999999]=7;a['key'][999999:]=9
 path=tmp_path/'index.bin';a.tofile(path);r=frequent_keys(path,3)
 assert r['keys'].tolist()==[7,9] and r['counts'].tolist()==[999999,5]

def test_direct_masks_equal_native_posting_lookup(tmp_path):
 targets=preprocess(pd.DataFrame([['S2-1','Acme','7 Rue Jean','France'],['S2-2','Acme','7 Rue Jean','France'],['S3-1','Acme','9 Rue Jean','France'],['S3-2','Different','Somewhere Else','US']],columns=['entity_id','business_name','business_address','country']))
 anchors=targets.iloc[[0,2,3]].reset_index(drop=True);expected=np.zeros((3,4),np.uint16);counts={}
 for name,extra in [('base',False),('extra',True)]:
  native=NativeIndex(tmp_path/'native',supplemental=extra);keys=native.keys(targets)
  a=np.zeros(keys.size,dtype=ENTRY);a['key']=keys.ravel();a['row']=np.repeat(np.arange(1,5),keys.shape[1]);a=a[a['key']!=0];a.sort(order=['key','row']);path=tmp_path/(name+'.bin');a.tofile(path)
  counts[name]=frequent_keys(path,1);indexed=NativeIndex(tmp_path/'native',path,supplemental=extra);r,m,o=indexed.lookup(indexed.keys(anchors),1)
  for i in range(3):expected[i,r[o[i]:o[i+1]]-1]|=m[o[i]:o[i+1]]
  indexed.close()
 actual=PairProvenance(tmp_path/'native',counts,1).masks(anchors,targets,np.repeat(np.arange(3),4),np.tile(np.arange(4),3))
 assert np.array_equal(actual,expected.ravel())
