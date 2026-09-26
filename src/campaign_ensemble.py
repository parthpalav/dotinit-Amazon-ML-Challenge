"""Select low-cost probability ensembles on the original selection set only."""
from pathlib import Path
import json,joblib,numpy as np
from .experiments import choose,evaluate
ROOT=Path('work/campaign_0931');OUT=Path('reports/campaign_0931');ART=Path('artifacts/campaign_0931')
class ProbabilityEnsemble:
 def __init__(self,members,weights):self.members=members;self.weights=weights
 def predict(self,features):return sum(w*m.predict(features) for w,m in zip(self.weights,self.members))
def main():
 from .campaign_ensemble import ProbabilityEnsemble as PortableEnsemble
 part=joblib.load('work/real_v1/features/validation.joblib');report=json.loads((OUT/'oof_selection.json').read_text());pred={d:np.load(ROOT/f'oof_compact_d{d}_validation_p.npy') for d in [5,7,9]};models={d:joblib.load(ART/f'oof_compact_d{d}.joblib') for d in pred}
 choices=[([7,9],[w,1-w]) for w in [.25,.5,.75]]+[([5,7,9],[1/3]*3),([5,9],[.5,.5])]
 for i,(depths,weights) in enumerate(choices):
  p=sum(w*pred[d] for d,w in zip(depths,weights));t,_=choose(part,p);stem=f'oof_ensemble_{i}';row={'model':stem,'depths':depths,'weights':weights,'threshold':t,**evaluate(part,p,t)};report.append(row)
  joblib.dump({'matcher':PortableEnsemble([models[d]['matcher'] for d in depths],weights),'feature_names':models[depths[0]]['feature_names'],'training_scope':'Ensemble of cross-fitted context models, weights selected on original validation'},ART/(stem+'.joblib'));np.save(ROOT/(stem+'_validation_p.npy'),p);print('ENSEMBLE',json.dumps(row),flush=True)
 (OUT/'oof_final_selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('BEST',json.dumps(max(report,key=lambda r:r['f0.5'])),flush=True)
if __name__=='__main__':main()
