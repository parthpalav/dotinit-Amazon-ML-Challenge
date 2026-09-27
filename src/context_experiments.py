"""Explore entity-context policies and a reranker without changing the champion.
Meta fitting/calibration use only the old calibration anchors. Old validation is
explicitly a selection set; a new confirmation set is required for promotion.
"""
from pathlib import Path
import argparse,gc,json,time
import joblib,numpy as np,pandas as pd
from .experiments import evaluate,choose
from .model import CalibratedMatcher
ROOT=Path('work/campaign_0931');OUT=Path('reports/campaign_0931');ART=Path('artifacts/campaign_0931')

def context(pairs,prob):
 group=pairs.source1_entity_id;d=pd.DataFrame({'p':prob,'g':group});g=d.groupby('g',sort=False).p
 result=pd.DataFrame({'base_probability':prob,'base_logit':np.log(np.clip(prob,1e-6,1-1e-6)/np.clip(1-prob,1e-6,1))},index=pairs.index)
 result['probability_rank']=g.rank(method='min',ascending=False)
 result['group_max']=g.transform('max');result['group_sum_others']=g.transform('sum')-prob
 result['gap_to_max']=result.group_max-prob
 for t in [.1,.3,.5,.675,.9,.99]:result['others_above_'+str(t)]=(d.p>=t).groupby(group,sort=False).transform('sum')-(d.p>=t)
 result['group_sum_squares_others']=(d.p**2).groupby(group,sort=False).transform('sum')-prob**2
 return result.astype(np.float32)

def load(split,model):
 location='work/windows_v1/features' if split=='confirmation' else 'work/real_v1/features'
 part=joblib.load(Path(location)/(split+'.joblib'));extra=joblib.load(Path('work/improvements')/(split+'_extra.joblib'))
 if not extra['pair_ids'].equals(part['pairs'][['source1_entity_id','candidate_entity_id']]):raise ValueError('Feature alignment')
 part['features']=pd.concat([part['features'],extra['features']],axis=1);del extra
 path=ROOT/(split+'_champion_p.npy')
 if path.exists():prob=np.load(path)
 else:prob=model['matcher'].predict(part['features']);np.save(path,prob)
 part['features']=pd.concat([part['features'],context(part['pairs'],prob)],axis=1)
 return part,prob

def policies(part,p):
 truth=part['truth_counts'];pos={x:i for i,x in enumerate(truth)};ix=np.array([pos[x] for x in part['pairs'].source1_entity_id]);n=len(truth)
 order=np.lexsort((-p,ix));sortedp=p[order];sortedix=ix[order];offset=np.r_[0,np.cumsum(np.bincount(ix,minlength=n))];rank=np.arange(len(p))-np.repeat(offset[:-1],np.diff(offset));matrix=np.zeros((n,int(rank.max())+1));matrix[sortedix,rank]=sortedp
 results=[]
 for temperature in [.6,.8,1.,1.2]:
  q=1/(1+np.exp(-np.log(np.clip(matrix,1e-9,1-1e-9)/np.clip(1-matrix,1e-9,1))/temperature))
  for multiplier in [.7,1.,1.3,1.6,2.]:
   expected=1.25*q.cumsum(axis=1)/(.25*q.sum(axis=1,keepdims=True)+np.arange(1,q.shape[1]+1))
   empty=np.prod(1-q,axis=1)*multiplier;k=np.argmax(np.c_[empty,expected],axis=1)
   selected=np.zeros(len(p));selected[order]=(rank<k[sortedix]);metrics=evaluate(part,selected,.5)
   results.append({'temperature':temperature,'singleton_multiplier':multiplier,**metrics})
 pd.DataFrame(results).to_csv(OUT/'expected_f_policies.tsv',sep='\t',index=False)
 print('BEST_EXPECTED_F',json.dumps(max(results,key=lambda r:r['f0.5'])),flush=True)

def main():
 from catboost import CatBoostClassifier
 parser=argparse.ArgumentParser();parser.add_argument('--peers',action='store_true');parser.add_argument('--detail',action='store_true');parser.add_argument('--compact',action='store_true');args=parser.parse_args();prefix=('context_peer' if args.peers else 'context')+('_detail' if args.detail else '')+('_compact' if args.compact else '')
 for x in [ROOT,OUT,ART]:x.mkdir(parents=True,exist_ok=True)
 model=joblib.load('artifacts/improvements/catboost_d10_evidence.joblib')
 cal,cp=load('calibration',model);val,vp=load('validation',model);del model;gc.collect()
 if args.peers:
  for split,part in [('calibration',cal),('validation',val)]:part['features']=pd.concat([part['features'],joblib.load(ROOT/('peer_'+split+'.joblib'))['features']],axis=1)
 if args.detail:
  for split,part in [('calibration',cal),('validation',val)]:part['features']=pd.concat([part['features'],joblib.load(ROOT/('detail_'+split+'.joblib'))['features']],axis=1)
 if args.compact:
  for part in [cal,val]:part['features']=part['features'].iloc[:,113:]
 policies(val,vp)
 anchors=np.asarray(cal['pairs'].source1_entity_id.unique(),dtype=object).copy();rng=np.random.default_rng(20260926);rng.shuffle(anchors);held=set(anchors[:1000]);fit=~cal['pairs'].source1_entity_id.isin(held);report=[]
 for depth in [5,7,9]:
  started=time.time();est=CatBoostClassifier(iterations=1200,depth=depth,learning_rate=.04,l2_leaf_reg=8,loss_function='Logloss',task_type='GPU',random_seed=43,thread_count=3,allow_writing_files=False,verbose=200)
  est.fit(cal['features'][fit],cal['pairs'].label[fit],eval_set=(cal['features'][~fit],cal['pairs'].label[~fit]),early_stopping_rounds=100)
  matcher=CalibratedMatcher(est);matcher.calibrate(cal['features'][~fit],cal['pairs'].label[~fit]);p=matcher.predict(val['features']);np.save(ROOT/f'{prefix}_d{depth}_validation_p.npy',p)
  joblib.dump({'matcher':matcher,'feature_names':cal['features'].columns.tolist(),'training_scope':'4000 original calibration anchors; sigmoid calibration on remaining 1000'},ART/f'{prefix}_d{depth}.joblib')
  for weight in [.25,.5,.75,1.]:
   blend=(1-weight)*vp+weight*p;threshold,_=choose(val,blend);metrics=evaluate(val,blend,threshold);row={'depth':depth,'weight':weight,'threshold':threshold,'trees':est.tree_count_,'seconds':time.time()-started,**metrics};report.append(row);print('RESULT',json.dumps(row),flush=True)
  (OUT/(prefix+'_selection.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('BEST',json.dumps(max(report,key=lambda r:r['f0.5'])),flush=True)

if __name__=='__main__':main()
