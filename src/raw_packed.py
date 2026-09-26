"""Compact, lossless raw-text storage for correction inference on 16GB RAM."""
from pathlib import Path
import json,mmap
import numpy as np
from .disk_store import connect
from .rescoring import filehash
ROOT=Path('work/campaign_0942/raw_packed_test')


def build():
    ROOT.mkdir(parents=True,exist_ok=True);con=connect('work/windows_v1/test/records.sqlite',True)
    sig={'store_manifest':filehash('work/windows_v1/test/store_manifest.json'),'fields':['business_name','business_address'],'version':1}
    for table in ['anchors','targets']:
        folder=ROOT/table;folder.mkdir(exist_ok=True);done=folder/'COMPLETE.json'
        if done.exists():
            assert json.loads(done.read_text(encoding='utf-8'))['signature']==sig;continue
        n=con.execute('select count(*) from '+table).fetchone()[0]
        offsets=[np.lib.format.open_memmap(folder/(field+'_offsets.partial.npy'),mode='w+',dtype='<u8',shape=(n+1,)) for field in sig['fields']]
        streams=[(folder/(field+'.partial.bin')).open('wb') for field in sig['fields']];pos=[0,0]
        for offset in offsets:offset[0]=0
        count=0
        for rid,name,address in con.execute('select rid,business_name,business_address from '+table+' order by rid'):
            count+=1
            if rid!=count:raise ValueError('Noncontiguous row IDs')
            for j,value in enumerate([name,address]):
                data=str(value or '').encode('utf-8');streams[j].write(data);pos[j]+=len(data);offsets[j][rid]=pos[j]
            if count%500000==0:print('RAW_PACKED',table,count,n,flush=True)
        for stream in streams:stream.close()
        for offset in offsets:offset.flush()
        del offset,offsets
        for field in sig['fields']:
            (folder/(field+'_offsets.partial.npy')).replace(folder/(field+'_offsets.npy'));(folder/(field+'.partial.bin')).replace(folder/(field+'.bin'))
        done.write_text(json.dumps({'signature':sig,'rows':n,'bytes':pos},indent=2),encoding='utf-8')
    con.close();(ROOT/'COMPLETE.json').write_text(json.dumps(sig,indent=2),encoding='utf-8')


class Cursor:
    def __init__(self,rows):self.rows=rows
    def fetchall(self):return self.rows


class RawConnection:
    """Restricted adapter for raw_evidence.features, not a general SQL adapter."""
    def __init__(self):
        if not (ROOT/'COMPLETE.json').exists():raise ValueError('Packed text incomplete')
        self.maps={};self.offsets={};self.files=[]
        for table in ['anchors','targets']:
            for field in ['business_name','business_address']:
                f=(ROOT/table/(field+'.bin')).open('rb');self.files.append(f);self.maps[table,field]=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);self.offsets[table,field]=np.load(ROOT/table/(field+'_offsets.npy'),mmap_mode='r')
    def execute(self,query,ids):
        table=query.split(' FROM ')[1].split(' WHERE ')[0]
        if table not in ('anchors','targets') or not query.startswith('SELECT * FROM '):raise ValueError('Unsupported query')
        rows=[]
        for rid in ids:
            values=[]
            for field in ['business_name','business_address']:
                offset=self.offsets[table,field]
                if rid<1 or rid>=len(offset):raise ValueError('Invalid packed row ID')
                values.append(self.maps[table,field][int(offset[rid-1]):int(offset[rid])].decode('utf-8'))
            rows.append((rid,'',*values,'','','',''))
        return Cursor(rows)


if __name__=='__main__':build()
