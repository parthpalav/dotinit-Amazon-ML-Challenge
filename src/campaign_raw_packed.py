"""Lossless packed-text backend for the frozen raw correction algorithm."""
import argparse,json,os
from pathlib import Path
import joblib,numpy as np,pandas as pd
from . import campaign_raw_scoring as scoring
from .raw_packed import RawConnection,ROOT as PACKED
from .raw_evidence import features
from .disk_store import connect
from .campaign_scoring import predict
from .rescoring import filehash


def initialize(recipe):
    scoring.W['con']=RawConnection();scoring.W['model']=joblib.load('artifacts/campaign_0942/'+recipe['model']+'.joblib');scoring.W['recipe']=recipe


def parity():
    con=connect('work/windows_v1/test/records.sqlite',True);packed=RawConnection();recipe=json.loads(Path('reports/campaign_0942/minimal_frozen_v5.json').read_text(encoding='utf-8'))['recipe'];model=joblib.load('artifacts/campaign_0942/'+recipe['model']+'.joblib');checked=0
    for start in [0,400000,1000000,1700000]:
        a=np.load(scoring.MAIN/f'{start:08d}.npy');ix=np.flatnonzero((a['p']>=.005)&(a['p']<.9999))[:300]
        pairs=pd.DataFrame({'anchor_rid':a['anchor'][ix],'target_rid':a['target'][ix]});x=features(pairs,con);y=features(pairs,packed);pd.testing.assert_frame_equal(x,y,check_exact=True)
        for f in [x,y]:f['current_probability']=a['p'][ix]
        np.testing.assert_array_equal(predict(model,x[model['feature_names']]),predict(model,y[model['feature_names']]));checked+=len(ix)
    con.close();report={'pairs':checked,'features_bitwise_equal':True,'probabilities_bitwise_equal':True,'countries':'Samples across four widely separated test shards','packed_manifest_hash':filehash(PACKED/'COMPLETE.json'),'adapter_hash':filehash('src/raw_packed.py')}
    Path('reports/campaign_0942/packed_parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('PACKED_PARITY',json.dumps(report),flush=True)


def main(args):
    root=Path(args.work);root.mkdir(parents=True,exist_ok=True);checked=json.loads(Path('reports/campaign_0942/packed_parity.json').read_text(encoding='utf-8'))
    if checked['packed_manifest_hash']!=filehash(PACKED/'COMPLETE.json') or checked['adapter_hash']!=filehash('src/raw_packed.py'):raise ValueError('Packed backend changed after parity test')
    provenance={'adapter_hash':checked['adapter_hash'],'wrapper_hash':filehash(__file__),'parity':checked,'original_algorithm_hash':filehash('src/campaign_raw_scoring.py')}
    path=root/'backend.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==provenance
    else:path.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    if args.import_old:
        old=Path('work/campaign_0942/test_minimal');signature=json.loads((old/'signature.json').read_text(encoding='utf-8'));frozen=json.loads(Path('reports/campaign_0942/minimal_frozen_v5.json').read_text(encoding='utf-8'))
        if signature['model_sha256']!=frozen['model_hash'] or signature['scoring_sha256']!=provenance['original_algorithm_hash']:raise ValueError('Wrong import model/algorithm')
        for p in old.glob('*.npy'):
            receipt=p.with_suffix('.json')
            if not receipt.exists():continue
            r=json.loads(receipt.read_text(encoding='utf-8'))
            if r['output_hash']!=filehash(p) or r['input_hash']!=filehash(scoring.MAIN/p.name):raise ValueError('Import hash mismatch')
            for f in [p,receipt]:
                dest=root/f.name
                if not dest.exists():os.link(f,dest)
        (root/'import.json').write_text(json.dumps({'original_signature':signature,'reason':'Identical lossless text fields and exact feature/prediction parity; original cache provenance retained.'},indent=2),encoding='utf-8')
    scoring.initialize=initialize;scoring.run(args)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parity',action='store_true');p.add_argument('--import-old',action='store_true');p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int);p.add_argument('--work',default='work/campaign_0942/test_minimal_packed');args=p.parse_args()
    if args.parity:parity()
    else:main(args)
