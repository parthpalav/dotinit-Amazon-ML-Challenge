"""Fail closed when experiment caches no longer match their inputs/settings."""
import hashlib,json
from pathlib import Path
from .native_schema import supplement_source

FIELDS={
 'selection':['seed','validation_fraction','calibration_fraction','fit_anchor_limit','calibration_anchor_limit','validation_anchor_limit'],
 'engineer':['seed','tfidf_max_features','fit_anchor_limit'],
 'ranker':['seed','retrieval_posting_limit','fit_anchor_limit'],
 'features':['seed','tfidf_max_features','retrieval_posting_limit','max_candidates_per_anchor'],
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def signature(config,stage,ids=None):
 root=Path(__file__).parent
 source_files=['normalization.py','preprocessing.py','contacts.py']
 if stage in ('engineer','features'):source_files+=['features.py']
 if stage in ('ranker','features'):source_files+=['disk_blocking.py','coarse_ranker.py','native/index.cpp','native/supplement.cpp']
 result={'stage':stage,'config':{k:getattr(config,k) for k in FIELDS[stage]},
  'code':{s:hashlib.sha256((supplement_source() if s=='native/supplement.cpp' else root/s).read_text(encoding='utf-8-sig').encode()).hexdigest() for s in source_files}}
 manifest=Path(config.working_dir)/'train/store_manifest.json'
 if manifest.exists():result['store']=json.loads(manifest.read_text(encoding='utf-8')).get('verified_input_sha256',json.loads(manifest.read_text(encoding='utf-8'))['inputs'])
 for name in (['candidate_ranker.joblib','feature_engineer.joblib'] if stage=='features' else ['entity_selection.npz'] if stage in ('engineer','ranker') else []):
  path=Path(config.working_dir)/name
  if path.exists():result[name]=sha(path)
 if ids is not None:result['ids_sha256']=hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()
 return result

def verify(path,config,stage,ids=None):
 path=Path(path)
 if not path.exists():return
 meta=path.with_suffix(path.suffix+'.meta.json')
 if not meta.exists() or json.loads(meta.read_text(encoding='utf-8'))!=signature(config,stage,ids):
  raise ValueError(f'Stale or unverified cache: {path}. Use a fresh working directory or explicit verified migration.')

def record(path,config,stage,ids=None):
 path=Path(path);path.with_suffix(path.suffix+'.meta.json').write_text(json.dumps(signature(config,stage,ids),sort_keys=True,indent=2), encoding='utf-8')
