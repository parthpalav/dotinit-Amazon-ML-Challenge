"""Verify and migrate legacy caches without rebuilding or copying 17 GB.

Hard links are used only for immutable data. Manifests are copied and changed
only after input SHA256, key-code and normalization equivalence checks pass.
"""
import hashlib,json,os,subprocess
from pathlib import Path
import joblib,numpy as np
from .artifact_compat import load_legacy
from .config import Config
from .disk_store import _input_signature,NativeIndex,fetch_records
from .disk_blocking import DiskBlocker

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def migrate():
    reportdir=Path('reports/improvements');reportdir.mkdir(parents=True,exist_ok=True)
    source=Path('src/native/index.cpp').read_text(encoding='utf-8')
    old=subprocess.check_output(['git','show','HEAD:src/native/index.cpp']).decode().replace('\r\n','\n')
    keypart=lambda s:s[s.index('static const int WIDTH'):s.index('struct Index {')].strip().split('// Python owns')[0].strip()
    if keypart(source)!=keypart(old):raise ValueError('Retrieval key algorithm changed; rebuild required')
    for name in ['src/normalization.py','src/native/supplement.cpp']:
        previous=subprocess.check_output(['git','show','HEAD:'+name]).decode().replace('\r\n','\n')
        if Path(name).read_text(encoding='utf-8')!=previous:raise ValueError('Normalization/extra keys changed')
    expected=json.loads(Path('reports/input_continuity.json').read_text(encoding='utf-8'))['current_tsv_sha256']
    actual={}
    for name,wanted in expected.items():
        actual[name]=sha(name)
        if actual[name]!=wanted:raise ValueError('Input content changed: '+name)
        print('VERIFIED',name,flush=True)
    config=json.loads(Path('config/real.json').read_text(encoding='utf-8'))
    config.update(working_dir='work/windows_v1',model_path='artifacts/real_model_windows.joblib',workers=1)
    dest=Path(config['working_dir']);dest.mkdir(parents=True,exist_ok=True)
    for split in ['train','test']:
        src=Path('work/real_v1')/split;dst=dest/split;dst.mkdir(exist_ok=True)
        for p in src.iterdir():
            if p.suffix=='.json' or p.name.endswith(('-wal','-shm')) or p.name.startswith('.'):continue
            target=dst/p.name
            if not target.exists():os.link(p,target)
        for p in src.glob('*.json'):
            contents=json.loads(p.read_text(encoding='utf-8'))
            if p.name=='store_manifest.json':
                contents['inputs']=_input_signature(Path(config['dataset_dir']),split)
                contents['native_sha256']=hashlib.sha256(source.encode()).hexdigest()
                contents['verified_input_sha256']={k:v for k,v in actual.items() if '/'+split+'/' in k}
            if p.name=='supplement_manifest.json':
                contents['native_sha256']=hashlib.sha256((Path('src/native/supplement.cpp').read_text(encoding='utf-8')+source).encode()).hexdigest()
            (dst/p.name).write_text(json.dumps(contents,indent=2), encoding='utf-8')
    artifact=load_legacy('artifacts/real_model.joblib')
    joblib.dump(artifact['retrieval_ranker'],dest/'candidate_ranker.joblib')
    for name in ['entity_selection.npz','feature_engineer.joblib']:
        p=dest/name
        if not p.exists():os.link(Path('work/real_v1')/name,p)
    (dest/'features').mkdir(exist_ok=True)
    for p in Path('work/real_v1/features').glob('*.joblib'):
        q=dest/'features'/p.name
        if not q.exists():os.link(p,q)
    cfg=Config(**config)
    blocker=DiskBlocker(cfg,'train')
    data=joblib.load('work/real_v1/features/validation.joblib')
    ids=np.load(dest/'entity_selection.npz')['validation'][:100]
    anchors=fetch_records(blocker.con,'anchors',ids)
    pairs,targets,stats,truth=blocker.retrieve(anchors,True)
    original=data['pairs'][data['pairs'].source1_entity_id.isin(anchors.entity_id)]
    columns=['source1_entity_id','candidate_entity_id','blocking_rules','label']
    sortcols=columns[:2]
    if not pairs[columns].sort_values(sortcols).reset_index(drop=True).equals(original[columns].sort_values(sortcols).reset_index(drop=True)):
        raise ValueError('Windows retrieval differs from saved baseline candidates')
    blocker.close()
    from . import cache_guard
    cache_guard.record(dest/'entity_selection.npz',cfg,'selection')
    cache_guard.record(dest/'feature_engineer.joblib',cfg,'engineer')
    cache_guard.record(dest/'candidate_ranker.joblib',cfg,'ranker')
    selection=np.load(dest/'entity_selection.npz')
    for split in selection.files:cache_guard.record(dest/'features'/(split+'.joblib'),cfg,'features',selection[split])
    artifact['config']=config;joblib.dump(artifact,config['model_path'])
    Path('config/windows.json').write_text(json.dumps(config,indent=2)+'\n', encoding='utf-8')
    result={'raw_hashes':actual,'windows_candidate_parity_anchors':len(anchors),'pairs':len(pairs),'status':'verified','config':'config/windows.json'}
    (reportdir/'windows_migration.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':migrate()
