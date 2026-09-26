"""Explicit recovery of trusted legacy Mac joblib artifacts.

XGBoost memory snapshots are not portable; capture the snapshot without calling
its native deserializer, then load only the standard UBJSON Model payload.
Never use on untrusted pickle files.
"""
from pathlib import Path
import json
from joblib.numpy_pickle import NumpyUnpickler

class Snapshot:
    def __setstate__(self,state): self.state=state

class LegacyUnpickler(NumpyUnpickler):
    def find_class(self,module,name):
        if module=='xgboost.core' and name=='Booster':return Snapshot
        return super().find_class(module,name)

def load_legacy(path, recover_ranker=True):
    path=Path(path)
    with path.open('rb') as f:
        artifact=LegacyUnpickler(str(path),f,ensure_native_byte_order=True).load()
    if recover_ranker:
        import xgboost as xgb
        ranker=artifact.get('retrieval_ranker',artifact)['model']
        snapshot=ranker._Booster
        if isinstance(snapshot,Snapshot):
            payload=bytes(snapshot.state['handle'])
            marker=b'L'+(5).to_bytes(8,'big')+b'Model'
            if not payload.startswith(b'{L') or not payload.endswith(b'}}') or payload.count(marker)!=1:
                raise ValueError('Unsupported legacy XGBoost snapshot envelope')
            model_start=payload.index(marker)+len(marker)
            booster=xgb.Booster()
            booster.load_model(bytearray(payload[model_start:-1]))
            ranker._Booster=booster
            ranker.set_params(n_jobs=1)
    return artifact
