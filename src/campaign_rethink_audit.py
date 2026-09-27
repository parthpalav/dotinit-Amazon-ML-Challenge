"""Audit current reranker errors and true competing owners; no model changes."""
from pathlib import Path
import collections,gc,json
import joblib,numpy as np,pandas as pd
from rapidfuzz.fuzz import token_sort_ratio
from .disk_store import connect,fetch_records
from .experiments import evaluate


def main():
    out=Path('reports/campaign_0942');out.mkdir(exist_ok=True);con=connect('work/windows_v1/train/records.sqlite',True);results={}
    for split,path,pfile in [('selection','work/real_v1/features/validation.joblib','work/campaign_0931/oof_compact_d9_validation_p.npy'),('confirmation_v4','work/windows_v1/features/confirmation_v4.joblib','work/campaign_0931/confirmation_v4_oof_p.npy')]:
        part=joblib.load(path);pairs=part['pairs'];del part['features'];p=np.load(pfile);y=pairs.label.to_numpy();assert len(y)==len(p);selected=p>=.6000000000000002;fp=pairs[selected&(y==0)].copy();fp['p']=p[selected&(y==0)];counts=collections.Counter();examples=[];known=set(part['truth_counts'])
        for row in fp.sort_values('p',ascending=False).itertuples(index=False):
            a=fetch_records(con,'anchors',[row.anchor_rid]).iloc[0];t=fetch_records(con,'targets',[row.target_rid]).iloc[0];owner=con.execute('select source_id from truth_pairs where target_id=?',(row.candidate_entity_id,)).fetchone();o=con.execute('select entity_id,business_name,business_address,name_norm,address_norm from anchors where entity_id=?',owner).fetchone() if owner else None
            counts['false_positive']+=1;counts['fp_'+a.country]+=1;counts['target_has_other_true_owner']+=o is not None;counts['owner_in_evaluated_anchors']+=bool(o and o[0] in known);counts['predicted_source_is_true_singleton']+=part['truth_counts'][row.source1_entity_id]==0;counts['missing_target_address']+=not bool(t.address_norm);counts['missing_source_address']+=not bool(a.address_norm);counts['same_exact_name_as_owner']+=bool(o and a.name_norm==o[3]);counts['same_exact_address_as_owner']+=bool(o and a.address_norm and a.address_norm==o[4]);counts['p_above_099']+=row.p>=.99
            if len(examples)<35:examples.append({'source':row.source1_entity_id,'target':row.candidate_entity_id,'country':a.country,'p':row.p,'source_name':a.business_name,'source_address':a.business_address,'target_name':t.business_name,'target_address':t.business_address,'true_owner':o[0] if o else None,'true_owner_name':o[1] if o else None,'true_owner_address':o[2] if o else None,'source_owner_name_similarity':token_sort_ratio(a.name_norm,o[3])/100 if o else None})
        bins=[]
        for lo,hi in [(0,.1),(.1,.3),(.3,.6),(.6,.8),(.8,.95),(.95,.99),(.99,1.00001)]:
            m=(p>=lo)&(p<hi);bins.append({'lo':lo,'hi':hi,'pairs':int(m.sum()),'positives':int(y[m].sum()),'mean_probability':float(p[m].mean()) if m.any() else None,'positive_fraction':float(y[m].mean()) if m.any() else None})
        counts['missing_true_candidates']=sum(part['truth_counts'].values())-int(y.sum());counts['in_pool_false_negatives']=int(((y==1)&~selected).sum())
        results[split]={'counts':dict(counts),'probability_bins':bins,'examples':examples,'metrics':evaluate(part,p,.6000000000000002),'note':'This set is now diagnostic/development data for subsequent choices; a new untouched confirmation is required.'};print(split,json.dumps(dict(counts)),flush=True);del part,pairs,fp;gc.collect()
    (out/'latest_error_audit.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8');con.close()

if __name__=='__main__':main()
