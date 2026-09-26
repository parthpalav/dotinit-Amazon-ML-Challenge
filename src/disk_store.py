"""Disk-backed records and compact memory-mapped blocking indexes for full data."""
from __future__ import annotations
import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .data_loader import check_required_files, parse_ids
from .normalization import normalize_business_name, normalize_address, normalize_text

LOG = logging.getLogger(__name__)
WIDTH = 24
ENTRY = np.dtype([('key','<u8'),('row','<u4')])
COLUMNS = ['entity_id','business_name','business_address','country','name_norm','address_norm','country_norm']


def rss_mb():
    if sys.platform == 'win32':
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, 'peak_wset', info.rss) / (1024 * 1024)
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(1024*1024) if sys.platform=='darwin' else value/1024


class NativeIndex:
    def __init__(self, directory: str | Path, index_path=None, supplemental=False):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        source=Path(__file__).parent/'native'/('supplement.cpp' if supplemental else 'index.cpp')
        digest=hashlib.sha256((source.read_text(encoding='utf-8')+(source.parent/'index.cpp').read_text(encoding='utf-8')).encode()).hexdigest()[:16]
        self.width=64 if supplemental else WIDTH
        suffix='.dll' if sys.platform=='win32' else '.dylib' if sys.platform=='darwin' else '.so'
        library=directory/f'libber-{digest}{suffix}'
        if not library.exists():
            compiler=shutil.which('clang++') or shutil.which('g++')
            if not compiler: raise RuntimeError('A C++17 compiler is required for the disk retrieval backend')
            temporary=library.with_suffix(library.suffix+f'.{os.getpid()}.tmp')
            flags=['-O3','-std=c++17','-dynamiclib' if sys.platform=='darwin' else '-shared']
            flags += ['-static-libgcc','-static-libstdc++'] if sys.platform=='win32' else ['-fPIC']
            subprocess.run([compiler,*flags,str(source),'-o',str(temporary)],check=True)
            try:
                temporary.replace(library)
            except PermissionError:
                if not library.exists():raise
                temporary.unlink()
        compiler=shutil.which('g++') or shutil.which('clang++')
        self._dll_directory = os.add_dll_directory(str(Path(compiler).parent)) if sys.platform=='win32' and compiler else None
        self.lib=ctypes.CDLL(str(library.resolve()))
        chars=ctypes.POINTER(ctypes.c_char_p)
        self.lib.ber_keys.argtypes=[chars,chars,chars,ctypes.c_size_t,ctypes.c_void_p]
        self.lib.ber_attach.argtypes=[ctypes.c_void_p,ctypes.c_size_t]; self.lib.ber_attach.restype=ctypes.c_void_p
        self.lib.ber_close.argtypes=[ctypes.c_void_p]
        self.lib.ber_lookup.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint32,
                                     ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        self.lib.ber_lookup.restype=ctypes.c_size_t
        self.key_function=self.lib.ber_keys
        self.lookup_function=self.lib.ber_lookup
        if supplemental:
            self.key_function=self.lib.ber_extra_keys
            self.key_function.argtypes=self.lib.ber_keys.argtypes
            self.lookup_function=self.lib.ber_extra_lookup
            self.lookup_function.argtypes=self.lib.ber_lookup.argtypes
            self.lookup_function.restype=ctypes.c_size_t
        self.handle=None
        self.mapping=None
        if index_path:
            size=Path(index_path).stat().st_size
            if size % ENTRY.itemsize:raise ValueError('Corrupt packed index length')
            self.mapping=np.memmap(index_path,dtype=ENTRY,mode='r') if size else np.empty(0,dtype=ENTRY)
            self.handle=self.lib.ber_attach(self.mapping.ctypes.data,size)
            if not self.handle: raise OSError(f'Unable to mmap index {index_path}')

    def keys(self, records):
        n=len(records)
        columns=[]
        for field in ('name_norm','address_norm','country_norm'):
            columns.append((ctypes.c_char_p*n)(*(x.encode('utf-8') for x in records[field])))
        result=np.zeros((n,self.width),dtype=np.uint64)
        self.key_function(*columns,n,result.ctypes.data)
        return result

    def lookup(self, keys, max_posting):
        if not self.handle: raise RuntimeError('Index not opened')
        keys=np.ascontiguousarray(keys,dtype=np.uint64)
        capacity=max(1000,len(keys)*256)
        while True:
            rows=np.empty(capacity,np.uint32); masks=np.empty(capacity,np.uint16); offsets=np.empty(len(keys)+1,np.uint32)
            used=self.lookup_function(self.handle,keys.ctypes.data,len(keys),max_posting,rows.ctypes.data,
                                     masks.ctypes.data,offsets.ctypes.data,capacity)
            if used != ctypes.c_size_t(-1).value:
                return rows[:used],masks[:used],offsets
            capacity*=2
            if capacity>len(keys)*max_posting*(self.width+16)+1000: raise RuntimeError('Unexpected retrieval capacity')

    def close(self):
        if getattr(self,'handle',None):
            self.lib.ber_close(self.handle);self.handle=None
        self.mapping=None

    def __del__(self):
        self.close()


def connect(path,readonly=False):
    con=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) if readonly else sqlite3.connect(path)
    con.execute('PRAGMA cache_size=-65536')
    con.execute('PRAGMA mmap_size=268435456')
    if not readonly:
        con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=NORMAL')
    return con


def _input_signature(root,split):
    paths=check_required_files(root,split)
    if split=='train':paths.append(Path(root)/'train/train_ground_truth.tsv')
    return {str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}


def build_store(config,split):
    """Stream every row; fail on invalid IDs. Do not change raw source files."""
    root=Path(config.dataset_dir); directory=Path(config.working_dir)/split
    directory.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/'store_manifest.json'
    signature=_input_signature(root,split)
    native_hash=hashlib.sha256((Path(__file__).parent/'native/index.cpp').read_text(encoding='utf-8').encode()).hexdigest()
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['inputs']!=signature or manifest['native_sha256']!=native_hash:
            raise ValueError('Cached store differs from inputs or key code; use a fresh working_dir')
        LOG.info('Reusing verified %s disk store',split);return manifest
    db=directory/'records.sqlite'
    if db.exists():
        raise RuntimeError(f'Incomplete store in {directory}; use a fresh working_dir to rebuild safely')
    started=time.time(); con=connect(db)
    definition='rid INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE, business_name TEXT, business_address TEXT, country TEXT, name_norm TEXT, address_norm TEXT, country_norm TEXT'
    con.execute(f'CREATE TABLE anchors ({definition})');con.execute(f'CREATE TABLE targets ({definition})')
    shards=[(directory/f'keys-{j:02d}.bin').open('wb') for j in range(16)]
    native=NativeIndex(Path(config.working_dir)/'native')
    counts={};country_counts={};rid=0
    try:
        for source,path in enumerate(check_required_files(root,split),1):
            table='anchors' if source==1 else 'targets'; total=0;countries={}
            for frame in pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,chunksize=config.io_chunk_size):
                if frame.columns.tolist()!=['entity_id','business_name','business_address','country']:
                    raise ValueError(f'Unexpected schema at {path}: {frame.columns.tolist()}')
                for c in frame:frame[c]=frame[c].str.strip()
                if not frame.entity_id.str.startswith(f'S{source}-').all() or frame.entity_id.str.contains(r'[,\s]',regex=True).any():
                    raise ValueError(f'Invalid IDs at {path}')
                frame['name_norm']=frame.business_name.map(normalize_business_name)
                frame['address_norm']=frame.business_address.map(normalize_address)
                frame['country_norm']=frame.country.map(normalize_text)
                begin=total if source==1 else rid
                rowids=np.arange(begin+1,begin+len(frame)+1,dtype=np.uint32)
                con.executemany(f'INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)',
                                ((int(r),*row) for r,row in zip(rowids,frame[COLUMNS].itertuples(index=False,name=None))))
                con.commit()
                if source!=1:
                    keys=native.keys(frame).ravel()
                    entries=np.empty(len(keys),dtype=ENTRY);entries['key']=keys;entries['row']=np.repeat(rowids,WIDTH)
                    groups=(keys>>60).astype(np.uint8)
                    for j,stream in enumerate(shards):
                        entries[(groups==j)&(keys!=0)].tofile(stream)
                    rid+=len(frame)
                total+=len(frame)
                for c,n in frame.country.value_counts().items():countries[c]=countries.get(c,0)+int(n)
                if total%500000==0:LOG.info('%s S%d stored=%d peak_RSS=%.0f MiB',split,source,total,rss_mb())
            counts[f'source{source}']=total;country_counts[f'source{source}']=countries
            LOG.info('%s S%d complete rows=%d',split,source,total)
    finally:
        for stream in shards:stream.close()
    index_path=directory/'blocking_index.bin'
    with index_path.open('wb') as output:
        for j in range(16):
            path=directory/f'keys-{j:02d}.bin'
            entries=np.fromfile(path,dtype=ENTRY)
            entries.sort(order=['key','row'],kind='quicksort')
            # A target can emit the same token key more than once; retain one posting.
            if len(entries)>1:
                keep=np.r_[True,(entries['key'][1:]!=entries['key'][:-1])|(entries['row'][1:]!=entries['row'][:-1])]
                entries=entries[keep]
            entries.tofile(output)
            LOG.info('%s sorted key shard %d/16 postings=%d peak_RSS=%.0f MiB',split,j+1,len(entries),rss_mb())
            path.unlink()
    if split=='train':
        con.execute('CREATE TABLE labels (entity_id TEXT PRIMARY KEY, matches TEXT NOT NULL, count INTEGER NOT NULL)')
        con.execute('CREATE TABLE truth_pairs (source_id TEXT NOT NULL, target_id TEXT PRIMARY KEY)')
        positives=singletons=rows=0
        for frame in pd.read_csv(root/'train/train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=config.io_chunk_size):
            if frame.columns.tolist()!=['source1_entity_id','matched_entity_ids']:raise ValueError('Unexpected ground truth schema')
            values=[];links=[]
            for source,cell in frame.itertuples(index=False,name=None):
                ids=parse_ids(cell)
                if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate ground-truth pair for {source}')
                values.append((source,cell,len(ids)));links.extend((source,target) for target in ids)
                positives+=len(ids);singletons+=not ids
            con.executemany('INSERT INTO labels VALUES (?,?,?)',values)
            # Unique target IDs enforce the deduplicated-reference semantics and reject conflicting labels.
            con.executemany('INSERT INTO truth_pairs VALUES (?,?)',links)
            con.commit();rows+=len(frame)
        missing=con.execute('SELECT COUNT(*) FROM anchors a LEFT JOIN labels l USING(entity_id) WHERE l.entity_id IS NULL').fetchone()[0]
        unknown=con.execute('SELECT COUNT(*) FROM labels l LEFT JOIN anchors a USING(entity_id) WHERE a.entity_id IS NULL').fetchone()[0]
        invalid=con.execute('SELECT COUNT(*) FROM truth_pairs t LEFT JOIN targets r ON t.target_id=r.entity_id WHERE r.entity_id IS NULL').fetchone()[0]
        if missing or unknown or invalid:raise ValueError(f'Ground-truth mismatch: missing={missing},unknown={unknown},invalid={invalid}')
        counts.update(ground_truth_rows=rows,positive_pairs=positives,singletons=singletons,
                      possible_negative_pairs=counts['source1']*(counts['source2']+counts['source3'])-positives,
                      duplicate_pairs=0,conflicting_target_labels=0)
    con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close()
    manifest={'inputs':signature,'native_sha256':native_hash,'counts':counts,'countries':country_counts,
              'seconds':time.time()-started,'peak_rss_mb':rss_mb(),'index_bytes':index_path.stat().st_size,
              'index_entries':index_path.stat().st_size//ENTRY.itemsize}
    manifest_path.write_text(json.dumps(manifest,indent=2), encoding='utf-8');return manifest


def fetch_records(con,table,ids):
    ids=sorted(set(map(int,ids)));rows=[]
    for start in range(0,len(ids),900):
        block=ids[start:start+900]
        rows.extend(con.execute(f'SELECT * FROM {table} WHERE rid IN ({",".join("?" for _ in block)})',block).fetchall())
    return pd.DataFrame(rows,columns=['rid',*COLUMNS])
