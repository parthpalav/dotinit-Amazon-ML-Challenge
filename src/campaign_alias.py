"""Evaluate one-hop candidate expansion from high-confidence predicted name aliases.
Ground truth is used only after proposing/scoring candidates for evaluation.
"""
from pathlib import Path
import argparse,gc,hashlib,json,time
import joblib,numpy as np,pandas as pd
from .config import Config
from .disk_store import connect,fetch_records,COLUMNS
from .preprocessing import preprocess
from .evidence import make_stats,enrich
from .context_experiments import context
from .peer_evidence import features as peers
from .detail_evidence import features as details
from .experiments import evaluate
ROOT=Path('work/campaign_0931');OUT=Path('reports/campaign_0931')
def namehash(name,country):return np.uint64(int.from_bytes(hashlib.blake2b((country+'\0'+name).encode(),digest_size=8).digest(),'little'))
def name_index(con,split):
 path=ROOT/(split+'_target_name_index.npz')
 if path.exists():z=np.load(path);return z['keys'],z['rids']
 n=con.execute('select count(*) from targets').fetchone()[0];keys=np.empty(n,np.uint64);rids=np.empty(n,np.uint32);started=time.time()
 for i,(rid,name,country) in enumerate(con.execute('select rid,name_norm,country_norm from targets')):
  keys[i]=namehash(name,country);rids[i]=rid
 order=np.argsort(keys);keys=keys[order];rids=rids[order];tmp=path.with_suffix('.partial')
 with tmp.open('wb') as f:np.savez(f,keys=keys,rids=rids)
 tmp.replace(path);print('NAME_INDEX',split,n,round(time.time()-started,1),flush=True);return keys,rids

def proposals(pairs,p,con,keys,rids):
 strong=pairs[p>=.95];texts=fetch_records(con,'targets',strong.target_rid);text={r.rid:(r.name_norm,r.country_norm) for r in texts.itertuples(index=False)};aliases={};existing={s:set(g.target_rid) for s,g in pairs.groupby('source1_entity_id',sort=False)};rid_to_source={r.anchor_rid:r.source1_entity_id for r in pairs.itertuples(index=False)}
 memo={};wanted={}
 for row in strong.itertuples(index=False):
  name,country=text[row.target_rid]
  if not name:continue
  aliases.setdefault(row.source1_entity_id,set()).add((name,country));h=namehash(name,country)
  if h not in memo:
   lo=int(np.searchsorted(keys,h,'left'));hi=int(np.searchsorted(keys,h,'right'));memo[h]=rids[lo:hi] if hi-lo<=20 else []
  wanted.setdefault(row.anchor_rid,set()).update(int(x) for x in memo[h] if int(x) not in existing[row.source1_entity_id])
 all_ids=set().union(*wanted.values()) if wanted else set();records=fetch_records(con,'targets',all_ids);by_rid={r.rid:r for r in records.itertuples(index=False)};rows=[]
 for ar,ids in wanted.items():
  source=rid_to_source[ar]
  for tr in sorted(ids):
   r=by_rid[tr]
   if (r.name_norm,r.country_norm) in aliases[source]:rows.append((ar,tr,source,r.entity_id,'predicted_name_alias'))
 return pd.DataFrame(rows,columns=['anchor_rid','target_rid','source1_entity_id','candidate_entity_id','blocking_rules'])

def unique_prob(pairs,p):
 frame=pd.DataFrame({'target':pairs.candidate_entity_id.to_numpy(),'p':p});best=frame.groupby('target',sort=False).p.transform('max');win=p==best;count=win.groupby(frame.target,sort=False).transform('sum');return np.where(win&(count==1),p,0.)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--split',default='validation',choices=['validation','confirmation_v4']);args=parser.parse_args();split=args.split;cfg=Config.load('config/windows.json');con=connect(Path(cfg.working_dir)/'train/records.sqlite',True);keys,rids=name_index(con,'train')
 location=Path('work/real_v1/features') if split=='validation' else Path(cfg.working_dir)/'features';part=joblib.load(location/(split+'.joblib'));old=part['pairs'];del part['features'];gc.collect();bp=np.load(ROOT/(split+'_champion_p.npy'));op=np.load(ROOT/('oof_compact_d9_validation_p.npy' if split=='validation' else 'confirmation_v4_oof_p.npy'))
 new=proposals(old,op,con,keys,rids);del keys,rids;gc.collect();print('ALIAS_PROPOSALS',split,len(new),new.anchor_rid.nunique(),flush=True)
 if not len(new):return
 # Label only after candidate proposal, with no labels available to feature builders.
 owner={};ids=new.candidate_entity_id.tolist()
 for i in range(0,len(ids),800):
  b=ids[i:i+800];owner.update(con.execute('select target_id,source_id from truth_pairs where target_id in ('+','.join('?' for _ in b)+')',b))
 new['label']=[int(owner.get(t)==s) for s,t in zip(new.source1_entity_id,new.candidate_entity_id)]
 stats=make_stats(cfg,'train');engineer=joblib.load(Path(cfg.working_dir)/'feature_engineer.joblib');champ=joblib.load('artifacts/improvements/catboost_d10_evidence.joblib');meta=joblib.load('artifacts/campaign_0931/oof_compact_d9.joblib');groups=new.anchor_rid.unique();blocks=[];pred=[];started=time.time()
 for i in range(0,len(groups),100):
  group=groups[i:i+100];added=new[new.anchor_rid.isin(group)].copy();a=fetch_records(con,'anchors',group);t=fetch_records(con,'targets',added.target_rid);records=preprocess(pd.concat([a[COLUMNS[:4]],t[COLUMNS[:4]]],ignore_index=True));X=engineer.transform(added,engineer.prepare(records));X=pd.concat([X.set_axis(added.index),enrich(added,con,stats)],axis=1);added['_p']=champ['matcher'].predict(X);added['_new']=True
  mask=old.anchor_rid.isin(group);previous=old[mask].copy();previous['_p']=bp[mask];previous['_new']=False;b=pd.concat([previous,added]).sort_values('anchor_rid',kind='stable').reset_index(drop=True);q=b['_p'].to_numpy();features=pd.concat([context(b,q),peers(b,q,con),details(b,con,stats)],axis=1);fresh=b['_new'].to_numpy();p=meta['matcher'].predict(features.loc[fresh,meta['feature_names']]);blocks.append(b[fresh].drop(columns=['_p','_new']));pred.append(p);print('ALIAS_SCORED',i,round(time.time()-started,1),flush=True)
 added=pd.concat(blocks,ignore_index=True);ap=np.concatenate(pred);allpairs=pd.concat([old,added],ignore_index=True);allp=np.r_[op,ap];after={**part,'pairs':allpairs};t=.6000000000000002
 report={'split':split,'seed_threshold':.95,'max_name_posting':20,'new_candidates':len(added),'new_true_candidates':int(added.label.sum()),'old_pair':evaluate(part,op,t),'new_pair':evaluate(after,allp,t),'old_unique':evaluate(part,unique_prob(old,op),t),'new_unique':evaluate(after,unique_prob(allpairs,allp),t)}
 joblib.dump({'pairs':added,'p':ap},ROOT/('alias_'+split+'.joblib'));(OUT/('alias_'+split+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');print('ALIAS_RESULT',json.dumps(report),flush=True);con.close()
if __name__=='__main__':main()
