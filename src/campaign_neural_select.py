"""Test whether multilingual predictions add information beyond raw correction."""
import json
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from .campaign_raw import ROOT,OUT,ART,subset
from .experiments import evaluate,choose


def main():
    d=joblib.load(ROOT/'neural_data.joblib');part=d['part'];rows=d['rows'];current=d['p'];neural=np.load(ROOT/'neural_development_p.npy')
    raw=np.load(ROOT/'raw_minimal_d7_development_p.npy');source=part['pairs'].iloc[rows].source1_entity_id
    ids=np.array(list(part['truth_counts']),dtype=object);np.random.default_rng(20260929).shuffle(ids)
    cal=d['split']==1;selection_ids=set(ids[:3000]);select=part['pairs'].source1_entity_id.isin(selection_ids).to_numpy();sp=subset(part,select,selection_ids)
    assert len(sp['truth_counts'])==3000
    def logit(p):p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
    X=pd.DataFrame({'current':logit(current[rows]),'raw':logit(raw[rows]),'neural':logit(neural)})
    y=part['pairs'].iloc[rows].label.to_numpy();results={'raw_baseline':evaluate(sp,raw[select],.7250000000000003),'experiments':[]}
    recipes=[('neural_logistic',LogisticRegression(C=1,max_iter=1000)),('neural_stack_d3',CatBoostClassifier(iterations=350,depth=3,learning_rate=.035,l2_leaf_reg=12,loss_function='Logloss',random_seed=1942,task_type='GPU',devices='0',thread_count=2,allow_writing_files=False,verbose=100))]
    for name,model in recipes:
        model.fit(X.loc[cal],y[cal]);q=current.copy();q[rows]=model.predict_proba(X)[:,1];threshold,_=choose(sp,q[select]);row={'model':name,'threshold':threshold,**evaluate(sp,q[select],threshold)};results['experiments'].append(row)
        joblib.dump({'estimator':model,'features':list(X.columns),'gate':(.005,.9999),'neural_model_dir':str(ART/'neural_minilm'),'raw_model':'artifacts/campaign_0942/raw_minimal_d7.joblib'},ART/(name+'.joblib'));np.save(ROOT/(name+'_development_p.npy'),q);print('NEURAL_SELECTION',json.dumps(row),flush=True)
    (OUT/'neural_selection.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
