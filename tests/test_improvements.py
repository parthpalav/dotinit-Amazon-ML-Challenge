"""Regression coverage for portable indexes and additional identity evidence."""
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd
from src.disk_store import NativeIndex,ENTRY,rss_mb
from src.preprocessing import preprocess
from src.evidence import components

def test_native_unicode_path_and_empty_index(tmp_path):
    native=NativeIndex(tmp_path/'native')
    frame=preprocess(pd.DataFrame([['S2-1','Café Lumière','7 Rue Jean','France']],columns=['entity_id','business_name','business_address','country']))
    keys=native.keys(frame);nonzero=np.unique(keys[keys!=0])
    entries=np.zeros(len(nonzero),dtype=ENTRY);entries['key']=nonzero;entries['row']=1
    path=tmp_path/'café.bin';entries.tofile(path)
    indexed=NativeIndex(tmp_path/'native',path)
    rows,masks,offsets=indexed.lookup(keys,240)
    assert rows.tolist()==[1] and offsets.tolist()==[0,1] and masks[0]>0
    indexed.close();path.unlink()  # mmap must be released on Windows.
    empty=tmp_path/'empty.bin';empty.touch()
    indexed=NativeIndex(tmp_path/'native',empty)
    rows,_,offsets=indexed.lookup(keys,240)
    assert len(rows)==0 and offsets.tolist()==[0,0]
    indexed.close();assert rss_mb()>0

def test_numbers_preserve_conflict_and_normalize_leading_zero():
    a=components('café sarl','4 rue demaire')
    b=components('cafe sas','7 rue demaire')
    assert a[0]=='cafe sarl' and a[6]=={'4'} and b[6]=={'7'}
    assert components('x','Af-0684 street')[6]=={'684'}
    assert components('x','')[5]==set()
