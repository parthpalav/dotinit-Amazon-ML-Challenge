"""Exact legacy retrieval with sorted access to memory-mapped posting pages."""
import ctypes,hashlib,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np


class SortedLookup:
    def __init__(self,native):
        self.native=native;source=Path(__file__).parent/'native/sorted_lookup.cpp'
        root=Path('work/campaign_0942');root.mkdir(exist_ok=True)
        suffix='.dll' if sys.platform=='win32' else '.so';path=root/('sorted_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
        compiler=shutil.which('g++') or shutil.which('clang++')
        if not path.exists():
            flags=['-O3','-std=c++17','-shared']+(['-static-libgcc','-static-libstdc++'] if sys.platform=='win32' else ['-fPIC'])
            subprocess.run([compiler,*flags,str(source),'-o',str(path)],check=True)
        self.dll_dir=os.add_dll_directory(str(Path(compiler).parent)) if sys.platform=='win32' else None
        self.lib=ctypes.CDLL(str(path.resolve()));self.fn=self.lib.sorted_lookup
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];self.fn.restype=ctypes.c_size_t

    def __call__(self,keys,max_posting):
        keys=np.ascontiguousarray(keys,dtype=np.uint64);n=len(keys);capacity=max(1000,n*2000)
        while True:
            rows=np.empty(capacity,np.uint32);masks=np.empty(capacity,np.uint16);offsets=np.empty(n+1,np.uint32)
            used=self.fn(self.native.mapping.ctypes.data,len(self.native.mapping),keys.ctypes.data,n,self.native.width,max_posting,rows.ctypes.data,masks.ctypes.data,offsets.ctypes.data,capacity)
            if used!=ctypes.c_size_t(-1).value:return rows[:used],masks[:used],offsets
            capacity*=2
