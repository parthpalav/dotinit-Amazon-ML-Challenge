"""Full test record adapter, preserving original country and normalized fields."""
from pathlib import Path
import json,mmap
import numpy as np
from .raw_packed import RawConnection,Cursor
from .supplement import PackedText
from .disk_store import connect
from .rescoring import filehash
ROOT=Path('work/campaign_0942/test_record_extras')


def build():
    ROOT.mkdir(parents=True,exist_ok=True)
    sig={'store':filehash('work/windows_v1/test/store_manifest.json'),'version':1}
    con=connect('work/windows_v1/test/records.sqlite',True)
    for table in ['anchors','targets']:
        folder=ROOT/table;folder.mkdir(exist_ok=True);done=folder/'COMPLETE.json'
        if done.exists():
            assert json.loads(done.read_text(encoding='utf-8'))['signature']==sig
            continue
        fields=['country'] if table=='targets' else ['entity_id','name_norm','address_norm','country','country_norm']
        n=con.execute('select count(*) from '+table).fetchone()[0]
        offsets=[np.lib.format.open_memmap(folder/(f+'_offsets.partial.npy'),mode='w+',dtype='<u8',shape=(n+1,)) for f in fields]
        streams=[(folder/(f+'.partial.bin')).open('wb') for f in fields];pos=[0]*len(fields)
        for o in offsets:o[0]=0
        for count,row in enumerate(con.execute('select rid,'+','.join(fields)+' from '+table+' order by rid'),1):
            assert row[0]==count
            for j,value in enumerate(row[1:]):
                data=str(value or '').encode('utf-8');streams[j].write(data);pos[j]+=len(data);offsets[j][count]=pos[j]
            if count%500000==0:print('TEST_EXTRAS',table,count,n,flush=True)
        for stream in streams:stream.close()
        for o in offsets:o.flush()
        del o,offsets
        for f in fields:
            (folder/(f+'_offsets.partial.npy')).replace(folder/(f+'_offsets.npy'));(folder/(f+'.partial.bin')).replace(folder/(f+'.bin'))
        done.write_text(json.dumps({'signature':sig,'rows':n,'fields':fields},indent=2),encoding='utf-8')
    con.close();(ROOT/'COMPLETE.json').write_text(json.dumps(sig,indent=2),encoding='utf-8')


class TestRecords(RawConnection):
    def __init__(self):
        super().__init__()
        if not (ROOT/'COMPLETE.json').exists():raise ValueError('Run campaign_test_records build')
        self.normal=PackedText('work/windows_v1/test')
        for table in ['anchors','targets']:
            meta=json.loads((ROOT/table/'COMPLETE.json').read_text(encoding='utf-8'))
            for field in meta['fields']:
                f=(ROOT/table/(field+'.bin')).open('rb');self.files.append(f);self.maps[table,field]=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);self.offsets[table,field]=np.load(ROOT/table/(field+'_offsets.npy'),mmap_mode='r')

    def execute(self,query,ids):
        table=query.split(' FROM ')[1].split(' WHERE ')[0]
        if table not in ('anchors','targets') or not query.startswith('SELECT * FROM '):raise ValueError('Unsupported query')
        normal=self.normal.get(ids) if table=='targets' else None;rows=[]
        for i,rid in enumerate(ids):
            values=[]
            for field in ['entity_id','business_name','business_address','country','name_norm','address_norm','country_norm']:
                if normal is not None and field in normal:values.append(normal[field][i]);continue
                offset=self.offsets[table,field]
                if rid<1 or rid>=len(offset):raise ValueError('Invalid row ID')
                values.append(self.maps[table,field][int(offset[rid-1]):int(offset[rid])].decode('utf-8'))
            rows.append((rid,*values))
        return Cursor(rows)


def parity():
    import pandas as pd
    from .disk_store import fetch_records
    con=connect('work/windows_v1/test/records.sqlite',True);packed=TestRecords();rng=np.random.default_rng(942);count=0
    for table,n in [('anchors',1732544),('targets',9969589)]:
        ids=np.r_[1,n,rng.choice(np.arange(1,n+1),2000,replace=False)]
        x=fetch_records(con,table,ids);y=fetch_records(packed,table,ids);pd.testing.assert_frame_equal(x,y,check_exact=True);count+=len(x)
    report={'rows':count,'all_fields_exact':True,'adapter_hash':filehash(__file__),'manifest_hash':filehash(ROOT/'COMPLETE.json')}
    Path('reports/campaign_0942/test_records_parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('TEST_RECORDS_PARITY',json.dumps(report),flush=True)


if __name__=='__main__':build();parity()
