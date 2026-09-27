import tempfile
from pathlib import Path
import numpy as np
import pytest
from src.disk_store import NativeIndex,ENTRY
from src.sorted_lookup import SortedLookup


@pytest.mark.parametrize('supplemental',[False,True])
def test_sorted_lookup_matches_original_with_duplicates_and_posting_limits(supplemental):
    rng=np.random.default_rng(1942)
    with tempfile.TemporaryDirectory() as tmp:
        a=np.empty(2000,ENTRY);a['key']=np.sort(rng.integers(1,100,2000,dtype=np.uint64));a['row']=rng.integers(1,400,2000,dtype=np.uint32);a.sort(order=['key','row']);path=Path(tmp)/'index.bin';a.tofile(path)
        native=NativeIndex('work/windows_v1/native',path,supplemental=supplemental);fast=SortedLookup(native);keys=rng.integers(0,110,(12,native.width),dtype=np.uint64)
        for limit in [1,5,20,240]:
            old=native.lookup(keys,limit);new=fast(keys,limit)
            for x,y in zip(old,new):np.testing.assert_array_equal(x,y)
        native.close()
