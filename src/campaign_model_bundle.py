"""Create a verified weights-only add-on; never overwrite tracked files on restore."""
from pathlib import Path
import argparse,hashlib,json,zipfile


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def pack(output):
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 models=['artifacts/improvements/catboost_d10_evidence.joblib','artifacts/campaign_0931/oof_compact_d9.joblib']
 documents=['reports/CAMPAIGN_MODEL_CARD.md','reports/CAMPAIGN_MODEL_LICENSE.txt','reports/campaign_0931/frozen_confirmation_v4.json','reports/campaign_0931/alias_frozen.json','requirements-campaign-tested.txt']
 frozen=json.loads(Path(documents[2]).read_text());expected=next(x['sha256'] for x in frozen['models'] if x['name']=='oof')
 if digest(models[0])!='1e55e08605fc84b83ab3e0c5b59126d5e853c0dcd2f7b8de94f475363aea6f69':raise ValueError('Base model differs from the verified champion')
 if digest(models[1])!=expected:raise ValueError('Selected model differs from frozen confirmation')
 manifest={'kind':'Weights-only add-on; not a complete dataset/index/checkpoint backup','restore':'Extract into the parth repository root. Only ignored artifacts/ and transfer/ paths are written. Original dataset/index assets and the complete verified base-score cache are still required.','files':{}}
 for p in models+documents:
  name=p if p in models else 'transfer/campaign_model_bundle/'+p
  manifest['files'][name]={'source':p,'bytes':Path(p).stat().st_size,'sha256':digest(p)}
 temp=output.with_suffix('.partial')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for name,item in manifest['files'].items():z.write(item['source'],name)
  z.writestr('transfer/campaign_model_bundle/manifest.json',json.dumps(manifest,indent=2))
 with zipfile.ZipFile(temp) as z:
  for name,item in manifest['files'].items():
   if hashlib.sha256(z.read(name)).hexdigest()!=item['sha256']:raise ValueError('Bundle verification failed: '+name)
 temp.replace(output);manifest.update(zip_path=str(output),zip_bytes=output.stat().st_size,zip_sha256=digest(output),verified=True)
 Path('reports/campaign_0931/model_bundle.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'path':str(output),'bytes':output.stat().st_size,'verified':True}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='transfer/campaign-0931-models.zip');a=p.parse_args();pack(a.output)
