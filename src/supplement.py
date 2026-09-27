"""Additional high-recall address/number keys and memory-mapped normalized text."""
import hashlib
import json
import logging
import mmap
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd
from .disk_store import NativeIndex,ENTRY,COLUMNS,connect,rss_mb
from .native_schema import supplement_source
LOG=logging.getLogger(__name__)


def build_supplement(config,split):
    directory=Path(config.working_dir)/split;manifest_path=directory/'supplement_manifest.json'
    source=supplement_source()
    signature=hashlib.sha256((source.read_text(encoding='utf-8')+(source.parent/'index.cpp').read_text(encoding='utf-8')).encode()).hexdigest()
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing['native_sha256']!=signature:raise ValueError('Supplement changed; use a fresh working_dir')
        return existing
    # Packed text is rewritten below: reject symlinks/hardlinks to preserved assets.
    for name in ['entity_id.bin','name_norm.bin','address_norm.bin','entity_id_offsets.npy','name_norm_offsets.npy','address_norm_offsets.npy','countries.npy']:
        target=directory/name
        if target.is_symlink() or (target.exists() and target.stat().st_nlink>1):
            raise ValueError(f'Refusing to rewrite shared packed asset {target}; use a fresh working directory with private packed outputs')
    con=connect(directory/'records.sqlite',readonly=True);count=con.execute('SELECT COUNT(*) FROM targets').fetchone()[0]
    native=NativeIndex(Path(config.working_dir)/'native',supplemental=True);started=time.time()
    offsets={field:np.lib.format.open_memmap(directory/f'{field}_offsets.npy',mode='w+',dtype=np.uint64,shape=(count+1,)) for field in ('entity_id','name_norm','address_norm')}
    streams={field:(directory/f'{field}.bin').open('wb') for field in offsets};positions={field:0 for field in offsets}
    countries=np.lib.format.open_memmap(directory/'countries.npy',mode='w+',dtype=np.uint16,shape=(count,));vocabulary={}
    with tempfile.TemporaryDirectory(prefix='.extra-',dir=directory) as temporary:
        shards=[(Path(temporary)/f'{j:02d}.bin').open('wb') for j in range(16)]
        cursor=con.execute('SELECT * FROM targets ORDER BY rid');done=0
        while True:
            rows=cursor.fetchmany(config.io_chunk_size)
            if not rows:break
            frame=pd.DataFrame(rows,columns=['rid',*COLUMNS]);rowids=frame.rid.to_numpy(dtype=np.uint32)
            if rowids[0]!=done+1:raise ValueError('Noncontiguous internal target rows')
            for field in offsets:
                encoded=[value.encode('utf-8')+b'\0' for value in frame[field]]
                lengths=np.fromiter(map(len,encoded),dtype=np.uint64,count=len(encoded))
                offsets[field][done:done+len(frame)]=positions[field]+np.r_[np.uint64(0),np.cumsum(lengths[:-1])]
                streams[field].write(b''.join(encoded));positions[field]+=int(lengths.sum());offsets[field][done+len(frame)]=positions[field]
            countries[done:done+len(frame)]=[vocabulary.setdefault(c,len(vocabulary)) for c in frame.country_norm]
            keys=native.keys(frame).ravel();entries=np.empty(len(keys),dtype=ENTRY);entries['key']=keys;entries['row']=np.repeat(rowids,native.width)
            groups=(keys>>60).astype(np.uint8)
            for j,stream in enumerate(shards):entries[(groups==j)&(keys!=0)].tofile(stream)
            done+=len(frame)
            if done%500000==0:LOG.info('%s supplementary records=%d/%d peak_RSS=%.0f MiB',split,done,count,rss_mb())
        for stream in shards:stream.close()
        for stream in streams.values():stream.close()
        for array in offsets.values():array.flush()
        countries.flush()
        pending=Path(temporary)/'supplement_index.bin'
        with pending.open('wb') as destination:
            for j in range(16):
                values=np.fromfile(Path(temporary)/f'{j:02d}.bin',dtype=ENTRY);values.sort(order=['key','row'],kind='quicksort')
                if len(values)>1:values=values[np.r_[True,(values['key'][1:]!=values['key'][:-1])|(values['row'][1:]!=values['row'][:-1])]]
                values.tofile(destination);LOG.info('%s supplementary sorted shard=%d/16 entries=%d',split,j+1,len(values))
        pending.replace(directory/'supplement_index.bin')
    (directory/'country_vocabulary.json').write_text(json.dumps(vocabulary), encoding='utf-8')
    result={'records':count,'native_sha256':signature,'seconds':time.time()-started,'peak_rss_mb':rss_mb(),
            'index_bytes':(directory/'supplement_index.bin').stat().st_size}
    manifest_path.write_text(json.dumps(result,indent=2), encoding='utf-8');con.close();return result


class PackedText:
    def __init__(self,directory):
        directory=Path(directory);self.files={};self.maps={};self.offsets={}
        for field in ('entity_id','name_norm','address_norm'):
            self.files[field]=(directory/f'{field}.bin').open('rb')
            self.maps[field]=mmap.mmap(self.files[field].fileno(),0,access=mmap.ACCESS_READ)
            self.offsets[field]=np.load(directory/f'{field}_offsets.npy',mmap_mode='r')
        self.countries=np.load(directory/'countries.npy',mmap_mode='r')
        vocab=json.loads((directory/'country_vocabulary.json').read_text(encoding='utf-8'))
        self.vocabulary=np.array(sorted(vocab,key=vocab.get),dtype=object)

    def get(self,ids):
        ids=np.asarray(ids,dtype=np.int64)-1;result={}
        for field,mapping in self.maps.items():
            starts=self.offsets[field][ids];ends=self.offsets[field][ids+1]
            result[field]=np.array([mapping[int(a):int(b)-1].decode('utf-8') for a,b in zip(starts,ends)],dtype=object)
        result['country_norm']=self.vocabulary[self.countries[ids]]
        return result
