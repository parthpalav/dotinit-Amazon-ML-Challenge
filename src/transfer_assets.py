"""Standard-library-only asset packaging/restoration; never starts ML jobs."""
import argparse,hashlib,json,os,shutil,time,zipfile
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def safe(root,name):
 p=(root/name).resolve()
 if not p.is_relative_to(root.resolve()):raise ValueError('Unsafe archive path: '+name)
 return p

def pack(output):
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise ValueError('Refusing to overwrite existing bundle')
 files=sorted(p for base in ['dataset','artifacts','work','outputs'] for p in Path(base).rglob('*') if p.is_file())
 files+=sorted(Path('reports/improvements').glob('*.npy'))
 seen={};entries=[];excluded=[];started=time.time();total=0
 temporary=output.with_suffix('.partial')
 with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
  for p in files:
   if p.suffix in ('.dll','.dylib','.so','.pyc','.partial','.tmp') or p.name.endswith(('-shm','-wal')) or '__pycache__' in p.parts:
    if p.name.endswith('-wal') and p.stat().st_size:raise ValueError('Nonempty SQLite WAL; checkpoint database before packing')
    excluded.append(p.as_posix());continue
   stat=p.stat();identity=(stat.st_dev,stat.st_ino);entry={'path':p.as_posix(),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
   if stat.st_ino and identity in seen:
    prior=seen[identity];entry.update(alias=prior['path'],sha256=prior['sha256'])
   else:
    h=hashlib.sha256()
    with p.open('rb') as source,archive.open(p.as_posix(),'w',force_zip64=True) as dest:
     for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block);dest.write(block)
    entry['sha256']=h.hexdigest();seen[identity]=entry;total+=stat.st_size
   entries.append(entry)
   if len(entries)%100==0 or stat.st_size>100000000:
    progress={'files':len(entries),'unique_bytes':total,'seconds':round(time.time()-started,1),'last':p.as_posix()};print(json.dumps(progress),flush=True)
    output.with_suffix('.progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
  manifest={'format':1,'entries':entries,'excluded_rebuildable_or_incomplete':excluded,'unique_bytes':total,'logical_bytes':sum(e['bytes'] for e in entries),'instructions':'Restore with python -m src.transfer_assets restore BUNDLE from the Git checkout root.'}
  archive.writestr('ASSET_MANIFEST.json',json.dumps(manifest,indent=2))
 temporary.replace(output)
 result={'archive':str(output),'archive_bytes':output.stat().st_size,'sha256':digest(output),'files':len(entries),'aliases':sum('alias' in e for e in entries),'unique_bytes':total,'logical_bytes':manifest['logical_bytes'],'seconds':time.time()-started}
 output.with_suffix('.sha256').write_text(result['sha256']+'  '+output.name+'\n',encoding='utf-8')
 Path('reports/improvements/transfer_bundle.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

def restore(bundle):
 bundle=Path(bundle);sidecar=bundle.with_suffix('.sha256')
 if sidecar.exists() and digest(bundle)!=sidecar.read_text().split()[0]:raise ValueError('Archive checksum mismatch')
 root=Path.cwd();start=time.time();verified={}
 with zipfile.ZipFile(bundle) as archive:
  manifest=json.loads(archive.read('ASSET_MANIFEST.json'))
  for i,e in enumerate(manifest['entries']):
   dest=safe(root,e['path']);dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    if dest.stat().st_size!=e['bytes'] or digest(dest)!=e['sha256']:raise ValueError('Existing file differs: '+str(dest))
   elif 'alias' in e:
    source=safe(root,e['alias'])
    if verified.get(e['alias'])!=e['sha256']:raise ValueError('Unverified alias target')
    try:os.link(source,dest)
    except OSError:shutil.copy2(source,dest)
   else:
    temp=dest.with_suffix(dest.suffix+'.transfer-partial');h=hashlib.sha256()
    with archive.open(e['path']) as source,temp.open('wb') as target:
     for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block);target.write(block)
    if temp.stat().st_size!=e['bytes'] or h.hexdigest()!=e['sha256']:raise ValueError('Asset checksum mismatch: '+str(dest))
    temp.replace(dest)
   os.utime(dest,ns=(e['mtime_ns'],e['mtime_ns']));verified[e['path']]=e['sha256']
   if i%100==0:print('VERIFIED',i+1,len(manifest['entries']),flush=True)
 Path('reports/improvements').mkdir(parents=True,exist_ok=True)
 Path('reports/improvements/asset_restore.json').write_text(json.dumps({'verified_files':len(verified),'seconds':time.time()-start,'bundle':str(bundle)},indent=2),encoding='utf-8')
 print('RESTORE COMPLETE',len(verified),'files; no ML process started',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['pack','restore']);p.add_argument('path');args=p.parse_args()
 (pack if args.action=='pack' else restore)(args.path)

if __name__=='__main__':main()
