"""Readable error examples and subgroup comparisons for campaign selection."""
import json
from pathlib import Path
import joblib,numpy as np
from .disk_store import connect,fetch_records
ROOT=Path('work/campaign_0931');OUT=Path('reports/campaign_0931')
def main():
 part=joblib.load('work/real_v1/features/validation.joblib');pairs=part['pairs'];old=np.load(ROOT/'validation_champion_p.npy');new=np.load(ROOT/'context_peer_detail_compact_d5_validation_p.npy');y=pairs.label.to_numpy();con=connect('work/windows_v1/train/records.sqlite',True);results={}
 masks={'recovered_true':(y==1)&(old<.675)&(new>=.675),'removed_false':(y==0)&(old>=.675)&(new<.675),'new_false':(y==0)&(old<.675)&(new>=.675),'lost_true':(y==1)&(old>=.675)&(new<.675)}
 for name,mask in masks.items():
  ix=np.flatnonzero(mask);ix=ix[np.argsort(-np.abs(new[ix]-old[ix]))];examples=[]
  for i in ix[:12]:
   row=pairs.iloc[i];a=fetch_records(con,'anchors',[row.anchor_rid]).iloc[0];t=fetch_records(con,'targets',[row.target_rid]).iloc[0];owner=con.execute('select source_id from truth_pairs where target_id=?',(row.candidate_entity_id,)).fetchone();actual=None
   if owner:
    actual=con.execute('select business_name,business_address from anchors where entity_id=?',owner).fetchone()
   examples.append({'source':row.source1_entity_id,'target':row.candidate_entity_id,'country':a.country,'anchor_name':a.business_name,'anchor_address':a.business_address,'target_name':t.business_name,'target_address':t.business_address,'old_p':float(old[i]),'new_p':float(new[i]),'label':int(y[i]),'actual_owner':owner[0] if owner else None,'actual_owner_text':actual})
  results[name]={'count':int(mask.sum()),'examples':examples}
 (OUT/'compact_error_examples.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v['count'] for k,v in results.items()}));con.close()
if __name__=='__main__':main()
