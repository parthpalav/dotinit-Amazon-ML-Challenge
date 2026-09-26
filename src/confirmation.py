"""Create a reserved confirmation holdout, never used to tune thresholds."""
from dataclasses import replace
from pathlib import Path
import json,joblib,numpy as np
import logging
from .config import Config
from .real_pipeline import feature_subset

def main():
    logging.basicConfig(level=logging.INFO)
    cfg=Config.load('config/windows.json');out=Path('reports/improvements');out.mkdir(exist_ok=True)
    selection=np.load(Path(cfg.working_dir)/'entity_selection.npz')
    used=np.concatenate([selection[k] for k in selection.files]);n=2206821
    eligible=np.setdiff1d(np.arange(1,n+1,dtype=np.uint32),used)
    holdout=np.sort(np.random.default_rng(20260926).choice(eligible,5000,replace=False))
    (Path('work/improvements')).mkdir(exist_ok=True)
    np.save('work/improvements/confirmation_ids.npy',holdout)
    cfg=replace(cfg,reports_dir=str(out),workers=1)
    part=feature_subset(cfg,'confirmation',holdout,Path(cfg.working_dir)/'feature_engineer.joblib',10320219)
    print('CONFIRMATION PREPARED',part['features'].shape,flush=True)

if __name__=='__main__':main()
