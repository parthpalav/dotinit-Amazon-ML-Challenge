"""Exact candidate provenance from supplied text and posting-count exclusions.

A single sequential index scan records over-limit keys. Per-batch replay then
uses key intersections, avoiding repeated random reads of multi-GB indexes.
"""
import ctypes,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import joblib,numpy as np
from .disk_store import ENTRY,NativeIndex

def frequent_keys(path,limit):
 blocks=[];pending=None;h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while raw:=f.read(ENTRY.itemsize*1000000):
   h.update(raw);data=np.frombuffer(raw,dtype=ENTRY);keys=data['key']
   starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1];unique=keys[starts];counts=np.diff(np.r_[starts,len(keys)]).astype(np.int64)
   if pending is not None:
    if unique[0]==pending[0]:counts[0]+=pending[1]
    elif pending[1]>limit:blocks.append((np.array([pending[0]],np.uint64),np.array([pending[1]],np.int64)))
   valid=counts[:-1]>limit
   if valid.any():blocks.append((unique[:-1][valid],counts[:-1][valid]))
   pending=(unique[-1],counts[-1])
 if pending is not None and pending[1]>limit:blocks.append((np.array([pending[0]],np.uint64),np.array([pending[1]],np.int64)))
 return {'keys':np.concatenate([b[0] for b in blocks]) if blocks else np.empty(0,np.uint64),'counts':np.concatenate([b[1] for b in blocks]) if blocks else np.empty(0,np.int64),'sha256':h.hexdigest()}

def prepare_counts(cfg):
 dest=Path('work/improvements/test_posting_counts.joblib');directory=Path(cfg.working_dir)/'test'
 paths=[directory/'blocking_index.bin',directory/'supplement_index.bin']
 signature={'limit':cfg.retrieval_posting_limit,'files':[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in paths]}
 if dest.exists():
  value=joblib.load(dest)
  if value['signature']!=signature:raise ValueError('Posting count cache changed; rebuild explicitly')
  return value
 value={'signature':signature,'base':frequent_keys(paths[0],cfg.retrieval_posting_limit),'extra':frequent_keys(paths[1],cfg.retrieval_posting_limit)}
 tmp=dest.with_suffix('.partial');joblib.dump(value,tmp);tmp.replace(dest);return value

class PairProvenance:
 def __init__(self,directory,counts,limit):
  self.base=NativeIndex(directory);self.extra=NativeIndex(directory,supplemental=True);self.counts=counts;self.limit=limit
  source=Path(__file__).parent/'native/provenance.cpp';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
  suffix='.dll' if sys.platform=='win32' else '.dylib' if sys.platform=='darwin' else '.so';path=Path(directory)/('provenance-'+digest+suffix)
  compiler=shutil.which('clang++') or shutil.which('g++')
  if not path.exists():
   if not compiler:raise RuntimeError('C++17 compiler required')
   flags=['-O3','-std=c++17','-dynamiclib' if sys.platform=='darwin' else '-shared']+(['-static-libgcc','-static-libstdc++'] if sys.platform=='win32' else ['-fPIC'])
   temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');subprocess.run([compiler,*flags,str(source),'-o',str(temporary)],check=True)
   try:temporary.replace(path)
   except PermissionError:
    if not path.exists():raise
    temporary.unlink()
  self.dll_dir=os.add_dll_directory(str(Path(compiler).parent)) if sys.platform=='win32' and compiler else None
  self.lib=ctypes.CDLL(str(path.resolve()));self.fn=self.lib.ber_pair_masks
  self.fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_size_t]*2+[ctypes.c_void_p];self.fn.restype=None
 def masks(self,anchors,targets,anchor_indices,target_indices):
  ai=np.ascontiguousarray(anchor_indices,dtype=np.uint32);ti=np.ascontiguousarray(target_indices,dtype=np.uint32);result=np.zeros(len(ai),np.uint16)
  for name,native in [('base',self.base),('extra',self.extra)]:
   a=native.keys(anchors);t=np.sort(native.keys(targets),axis=1);bad=self.counts[name];keys=bad['keys']
   if len(keys):
    pos=np.searchsorted(keys,a);safe=np.minimum(pos,len(keys)-1);count=np.where((pos<len(keys))&(keys[safe]==a),bad['counts'][safe],0)
    limits=np.full(a.shape[1],self.limit)
    if name=='base':limits[[0,21]]*=5
    a[count>limits]=0
   self.fn(a.ctypes.data,t.ctypes.data,ai.ctypes.data,ti.ctypes.data,len(ai),a.shape[1],result.ctypes.data)
  return result
