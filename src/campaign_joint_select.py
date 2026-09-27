"""Development-only comparison of neural + raw + global-owner evidence."""
import json
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from .campaign_raw import ROOT,OUT,ART,subset
from .experiments import evaluate,choose


def main():
    d=joblib.load(ROOT/'neural_data.joblib');part=d['part'];rows=d['rows'];p=d['p'];raw=np.load(ROOT/'raw_minimal_d7_development_p.npy');neural=np.load(ROOT/'neural_development_p.npy');reverse=np.load(ROOT/'raw_reverse_d7_development_p.npy')
    def logit(q):q=np.clip(q,1e-6,1-1e-6);return np.log(q/(1-q))
    ids=np.array(list(part['truth_counts']),dtype=object);np.random.default_rng(20260929).shuffle(ids)
    X=pd.DataFrame({'current':logit(p[rows]),'raw':logit(raw[rows]),'neural':logit(neural),'reverse':logit(reverse[rows])});cal=d['split']==1;selectids=set(ids[:3000]);sel=part['pairs'].source1_entity_id.isin(selectids).to_numpy();sp=subset(part,sel,selectids);y=part['pairs'].iloc[rows].label.to_numpy();results=[]
    assert len(sp['truth_counts'])==3000
    for name,model in [('joint_logistic',LogisticRegression(C=1,max_iter=1000)),('joint_stack_d3',CatBoostClassifier(iterations=350,depth=3,learning_rate=.035,l2_leaf_reg=12,loss_function='Logloss',random_seed=1942,task_type='GPU',devices='0',thread_count=2,allow_writing_files=False,verbose=100))]:
        model.fit(X.loc[cal],y[cal]);q=p.copy();q[rows]=model.predict_proba(X)[:,1];threshold,_=choose(sp,q[sel]);r={'model':name,'threshold':threshold,**evaluate(sp,q[sel],threshold)};results.append(r);joblib.dump({'estimator':model,'features':list(X.columns),'gate':(.005,.9999),'reverse_model':'artifacts/campaign_0942/raw_reverse_d7.joblib'},ART/(name+'.joblib'));np.save(ROOT/(name+'_development_p.npy'),q);print('JOINT_SELECTION',json.dumps(r),flush=True)
    (OUT/'joint_selection.json').write_text(json.dumps({'experiments':results},indent=2),encoding='utf-8')


if __name__=='__main__':main()
