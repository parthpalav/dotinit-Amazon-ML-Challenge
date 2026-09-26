import sqlite3
import numpy as np
import pandas as pd
from src.context_experiments import context
from src.peer_evidence import features as peers
from src.detail_evidence import features as details
from src.campaign_scoring import tasks,DTYPE

def database():
 con=sqlite3.connect(':memory:')
 for table in ['anchors','targets']:
  con.execute(f'create table {table}(rid integer,entity_id text,business_name text,business_address text,country text,name_norm text,address_norm text,country_norm text)')
 def put(table,rid,e,n,a):con.execute(f'insert into {table} values(?,?,?,?,?,?,?,?)',(rid,e,n,a,'India',n,a,'india'))
 put('anchors',1,'S1-1','cafe alpha','12 main road');put('anchors',2,'S1-2','beta stores','13 main road')
 put('targets',1,'S2-1','cafe alpha','12 main road');put('targets',2,'S3-2','cafe alpha','12 main road');put('targets',3,'S2-3','beta stores','13 main road')
 return con

def pairs():
 return pd.DataFrame({'anchor_rid':[1,1,2],'target_rid':[1,2,3],'source1_entity_id':['S1-1','S1-1','S1-2'],'candidate_entity_id':['S2-1','S3-2','S2-3'],'label':[1,1,1]},index=[10,11,12])

def test_peers_exclude_self_and_ignore_labels():
 con=database();p=pairs();a=peers(p,np.array([.95,.9,.99]),con);p['label']=0;b=peers(p,np.array([.95,.9,.99]),con)
 pd.testing.assert_frame_equal(a,b)
 assert a.loc[10,'peer_exact_name']==1
 assert a.loc[10,'peer_other_source_joint']==np.float32(.9)
 assert (a.loc[12]==0).all()  # An isolated candidate cannot corroborate itself.

def test_features_invariant_to_complete_anchor_batching():
 con=database();p=pairs();q=np.array([.95,.9,.99]);stats={'counts':{}}
 for fn in [lambda x,y:context(x,y),lambda x,y:peers(x,y,con),lambda x,y:details(x,con,stats)]:
  whole=fn(p,q);batched=pd.concat([fn(p.iloc[:2],q[:2]),fn(p.iloc[2:],q[2:])]);pd.testing.assert_frame_equal(whole,batched)
 assert context(p,q).loc[12,'group_sum_others']==0

def test_task_boundaries_preserve_empty_anchors_and_all_pairs():
 a=np.array([(1,1,.9),(1,2,.8),(3,1,.5),(5,3,.2)],dtype=DTYPE)
 assert list(tasks(a,6,2))==[(0,2,0,2),(2,4,2,3),(4,6,3,4)]
 assert list(tasks(a,6,2,3))==[(0,2,0,2),(2,3,2,3)]

def test_deployment_predict_and_portable_ensemble(tmp_path):
 import joblib,pytest
 from catboost import CatBoostClassifier
 from src.model import CalibratedMatcher
 from src.campaign_ensemble import ProbabilityEnsemble
 from src.campaign_scoring import predict
 X=pd.DataFrame({'a':np.arange(20,dtype=float),'b':np.arange(20,dtype=float)%3});y=(X.a>8).astype(int)
 est=CatBoostClassifier(iterations=3,depth=2,verbose=False,thread_count=1,allow_writing_files=False);est.fit(X,y)
 m=CalibratedMatcher(est);m.calibrate(X,y);artifact={'matcher':ProbabilityEnsemble([m,m],[.25,.75]),'feature_names':list(X.columns)}
 path=tmp_path/'model.joblib';joblib.dump(artifact,path);loaded=joblib.load(path)
 np.testing.assert_allclose(predict(loaded,X),m.predict(X),rtol=0,atol=1e-15)
 with pytest.raises(ValueError,match='schema'):predict(loaded,X[['b','a']])
