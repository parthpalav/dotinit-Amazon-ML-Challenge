"""Measure which missed true candidates could be recovered from predicted aliases."""
import json,collections,gc
from pathlib import Path
import joblib,numpy as np
from rapidfuzz.fuzz import token_sort_ratio
from .disk_store import connect,fetch_records
ROOT=Path('work/campaign_0931');OUT=Path('reports/campaign_0931')
def main():
 part=joblib.load('work/real_v1/features/validation.joblib');pairs=part['pairs'];del part;gc.collect();con=connect('work/windows_v1/train/records.sqlite',True);p=np.load(ROOT/'oof_compact_d9_validation_p.npy')
 grouped=pairs.groupby('source1_entity_id',sort=False);seen={s:set(g.candidate_entity_id) for s,g in grouped};names=list(seen);missing=[]
 for i in range(0,len(names),800):
  chunk=names[i:i+800]
  for s,cell in con.execute('select entity_id,matches from labels where entity_id in ('+','.join('?' for _ in chunk)+')',chunk):missing.extend((s,t) for t in cell.split(',') if t and t not in seen[s])
 strong=pairs[p>=.95];records=fetch_records(con,'targets',strong.target_rid);texts={r.entity_id:r for r in records.itertuples(index=False)};alias=collections.defaultdict(list)
 for r in strong.itertuples(index=False):alias[r.source1_entity_id].append(texts[r.candidate_entity_id])
 actual={}
 for i in range(0,len(missing),800):
  ids=[t for s,t in missing[i:i+800]]
  for row in con.execute('select entity_id,name_norm,address_norm,country_norm from targets where entity_id in ('+','.join('?' for _ in ids)+')',ids):actual[row[0]]=row[1:]
 counts=collections.Counter();examples=[]
 for s,t in missing:
  n,a,c=actual[t];counts['missing_true_pairs']+=1;counts['missing_'+c]+=1;counts['missing_address']+=not bool(a);counts['nonlatin_name']+=any(ord(x)>591 for x in n);same=[r for r in alias[s] if r.name_norm==n and n];counts['exact_predicted_alias']+=bool(same)
  if same:
   score=max((token_sort_ratio(a,r.address_norm)/100 if a and r.address_norm else 0) for r in same);counts['alias_with_address_similarity_085']+=score>=.85;counts['alias_missing_address']+=not bool(a)
   if len(examples)<12:examples.append({'source':s,'missed_target':t,'name':n,'address':a,'predicted_peer':same[0].entity_id,'peer_address':same[0].address_norm,'address_similarity':score})
 (OUT/'retrieval_opportunity.json').write_text(json.dumps({'counts':dict(counts),'examples':examples,'scope':'Labels identify omitted true pairs for audit only; aliases come from predictions >= .95 without peer labels.'},indent=2,ensure_ascii=False),encoding='utf-8');print(dict(counts),flush=True)
if __name__=='__main__':main()
