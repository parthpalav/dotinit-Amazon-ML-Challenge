"""Verified portable weights for the frozen multilingual and owner ensemble."""
from pathlib import Path
import hashlib,json,zipfile
from .campaign_neural_test import verify_frozen,OUT
from .rescoring import filehash


def main():
    frozen=verify_frozen();files={Path(p).as_posix():p for p in frozen['models']}
    for p in ['reports/campaign_0942/frozen_v6.json','requirements-neural.txt','requirements-campaign-tested.txt','reports/CAMPAIGN_0942_MODEL_CARD.md','reports/CAMPAIGN_MODEL_LICENSE.txt']:
        files['transfer/campaign_0942_ensemble_bundle/'+p]=p
    for p in ['reports/campaign_0942/confirmation_v6.json','reports/campaign_0942/ensemble_production.json']:
        if Path(p).exists():files['transfer/campaign_0942_ensemble_bundle/'+p]=p
    manifest={'kind':'Weights only; dataset/index/score caches are separate','restore':'Extract ZIP at repo root on parth; ignored artifacts/ and transfer/ are written. Install requirements-neural.txt in addition to base environment. Follow RETHINK.md.','files':{k:{'sha256':filehash(v),'bytes':Path(v).stat().st_size} for k,v in files.items()}}
    path=Path('transfer/campaign-0942-ensemble-models.zip');tmp=path.with_suffix('.partial')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for k,v in files.items():z.write(v,k)
        z.writestr('transfer/campaign_0942_ensemble_bundle/manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(tmp) as z:
        for k,v in manifest['files'].items():
            with z.open(k) as f:
                h=hashlib.file_digest(f,'sha256').hexdigest()
            if h!=v['sha256']:raise ValueError('ZIP verification failed')
    tmp.replace(path);manifest.update(zip_path=str(path),zip_sha256=filehash(path),verified=True);(OUT/'ensemble_bundle.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('ENSEMBLE_BUNDLE_VERIFIED',path,flush=True)


if __name__=='__main__':main()
