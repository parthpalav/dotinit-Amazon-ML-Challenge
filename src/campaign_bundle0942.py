"""Verified weights-only backup of the three-model raw-text pipeline."""
from pathlib import Path
import hashlib,json,zipfile
from .rescoring import filehash


def main():
    frozen=json.loads(Path('reports/campaign_0942/minimal_frozen_v5.json').read_text(encoding='utf-8'))
    models=['artifacts/improvements/catboost_d10_evidence.joblib','artifacts/campaign_0931/oof_compact_d9.joblib','artifacts/campaign_0942/'+frozen['recipe']['model']+'.joblib']
    expected=['1e55e08605fc84b83ab3e0c5b59126d5e853c0dcd2f7b8de94f475363aea6f69','b770fd7bea838e53f69a6f3dcdbdd3917520034d46dbfa6a3b045f0fbff20651',frozen['model_hash']]
    for p,h in zip(models,expected):
        if filehash(p)!=h:raise ValueError('Frozen model mismatch: '+p)
    docs=['reports/CAMPAIGN_0942_MODEL_CARD.md','reports/CAMPAIGN_MODEL_LICENSE.txt','reports/campaign_0942/minimal_frozen_v5.json','reports/campaign_0942/minimal_confirmation_v5.json','requirements-campaign-tested.txt']
    files={p:p for p in models};files.update({'transfer/campaign_0942_bundle/'+p:p for p in docs})
    manifest={'kind':'Weights-only add-on; dataset/index/main-score cache not included','restore':'Extract into repo root on parth; only ignored artifacts/ and transfer/ paths are written. Follow tracked README and RETHINK.md for current commands.','files':{k:{'sha256':filehash(v),'bytes':Path(v).stat().st_size} for k,v in files.items()}}
    path=Path('transfer/campaign-0942-models.zip');tmp=path.with_suffix('.partial')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for k,v in files.items():z.write(v,k)
        z.writestr('transfer/campaign_0942_bundle/manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(tmp) as z:
        for k,v in manifest['files'].items():
            if hashlib.sha256(z.read(k)).hexdigest()!=v['sha256']:raise ValueError('ZIP verification failed')
    tmp.replace(path);manifest.update(zip_path=str(path),zip_sha256=filehash(path),verified=True)
    Path('reports/campaign_0942/model_bundle.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('BUNDLE_VERIFIED',path,flush=True)


if __name__=='__main__':main()
