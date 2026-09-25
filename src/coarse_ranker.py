"""Small XGBoost retrieval ranker fitted only on fitting S1 entities.

This is a candidate-filtering stage; the final matching model sees and exports the
same retained candidates. All recall loss from this stage is measured explicitly.
"""
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rapidfuzz import process,fuzz
from rapidfuzz.distance import JaroWinkler

COARSE_NAMES=['name_ratio','name_wratio','name_token_set','name_token_sort','name_jaro',
              'address_ratio','address_wratio','address_token_set','address_token_sort',
              'name_length_ratio','address_length_ratio','country_match','name_missing','address_missing',
              'number_overlap','number_jaccard','name_token_jaccard','address_token_jaccard',
              'country_name','city_name','postal','name_ngram','address','strong_name','name_number','address_number','name_pair']


def grouped_similarities(left,right,boundaries,scorer):
    """Reuse each query's scorer setup over its already-retrieved targets.

    Each matrix is only 1 x the candidate count for one anchor, never an
    all-source cross product. Scores retain the original float32 semantics.
    """
    scores=np.empty(len(left),dtype=np.float32)
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        scores[lo:hi]=process.cdist([left[lo]],right[lo:hi],scorer=scorer,
                                   dtype=np.float32,workers=1)[0]
    return scores


def coarse_features(anchors,text,indexer,repeated,masks,similarity_threads=1):
    n=len(indexer)
    if not n:return np.empty((0,len(COARSE_NAMES)),dtype=np.float32)
    left_name=anchors.name_norm.to_numpy()[repeated];right_name=text['name_norm'][indexer]
    left_address=anchors.address_norm.to_numpy()[repeated];right_address=text['address_norm'][indexer]
    boundaries=np.r_[0,np.flatnonzero(np.diff(repeated))+1,n]
    jobs=[];missing_masks=[]
    for left,right,scorers in ((left_name,right_name,(fuzz.ratio,fuzz.WRatio,fuzz.token_set_ratio,fuzz.token_sort_ratio,JaroWinkler.normalized_similarity)),
                               (left_address,right_address,(fuzz.ratio,fuzz.WRatio,fuzz.token_set_ratio,fuzz.token_sort_ratio))):
        missing=(left=='')|(right=='')
        for scorer in scorers:
            jobs.append((left,right,boundaries,scorer));missing_masks.append(missing)
    if similarity_threads>1:
        with ThreadPoolExecutor(max_workers=similarity_threads) as pool:
            values=list(pool.map(lambda args:grouped_similarities(*args),jobs))
    else:values=[grouped_similarities(*args) for args in jobs]
    for score,args,missing in zip(values,jobs,missing_masks):
        if args[-1] is not JaroWinkler.normalized_similarity:score/=100
        score[missing]=0
    for left,right in ((left_name,right_name),(left_address,right_address)):
        ll=np.fromiter(map(len,left),np.float32,n);rl=np.fromiter(map(len,right),np.float32,n)
        values.append(np.minimum(ll,rl)/np.maximum(1,np.maximum(ll,rl)))
    values.extend([(anchors.country_norm.to_numpy()[repeated]==text['country_norm'][indexer]).astype(np.float32),
                   ((left_name=='')|(right_name=='')).astype(np.float32),((left_address=='')|(right_address=='')).astype(np.float32)])
    left_numbers=[set(re.findall(r'\d+',x)) for x in anchors.address_norm]
    right_numbers=[set(re.findall(r'\d+',x)) for x in text['address_norm']]
    intersection=np.fromiter((len(left_numbers[a]&right_numbers[b]) for a,b in zip(repeated,indexer)),np.float32,n)
    union=np.fromiter((len(left_numbers[a]|right_numbers[b]) for a,b in zip(repeated,indexer)),np.float32,n)
    values.extend([intersection,intersection/np.maximum(1,union)])
    for field in ('name_norm','address_norm'):
        ls=[set(x.split()) for x in anchors[field]];rs=[set(x.split()) for x in text[field]]
        values.append(np.fromiter((len(ls[a]&rs[b])/max(1,len(ls[a]|rs[b])) for a,b in zip(repeated,indexer)),np.float32,n))
    values.extend(((masks&(1<<bit))!=0).astype(np.float32) for bit in range(9))
    return np.column_stack(values).astype(np.float32)


def fit_candidate_ranker(config,selected):
    from pathlib import Path
    import json,time,logging,joblib
    from xgboost import XGBClassifier
    from .disk_store import fetch_records
    from .disk_blocking import DiskBlocker
    log=logging.getLogger(__name__);path=Path(config.working_dir)/'candidate_ranker.joblib'
    if path.exists():return joblib.load(path)
    rng=np.random.default_rng(config.seed+311);order=rng.permutation(selected['fit'])
    train_ids=order[:min(2000,len(order)//2)];holdout=order[len(train_ids):len(train_ids)+min(1000,len(order)-len(train_ids))]
    blocker=DiskBlocker(config,'train',load_ranker=False);features=[];labels=[];started=time.time()
    for start in range(0,len(train_ids),100):
        anchors=fetch_records(blocker.con,'anchors',train_ids[start:start+100]);truth=blocker.get_truth(anchors)
        rows,masks,offsets,text,indexer,repeated=blocker.raw_retrieve(anchors)
        x=coarse_features(anchors,text,indexer,repeated,masks)
        entity=anchors.entity_id.to_numpy()[repeated];target=text['entity_id'][indexer]
        y=np.fromiter((int(b in truth[a]) for a,b in zip(entity,target)),dtype=np.int8,count=len(rows))
        features.append(x);labels.append(y)
        log.info('Retrieval ranker fitting anchors=%d/%d pairs=%d elapsed=%.1fs',min(start+100,len(train_ids)),len(train_ids),sum(map(len,labels)),time.time()-started)
    x=np.concatenate(features);y=np.concatenate(labels);del features,labels
    model=XGBClassifier(n_estimators=160,max_depth=5,learning_rate=.08,subsample=.85,colsample_bytree=.9,
                        reg_lambda=5,tree_method='hist',objective='binary:logistic',eval_metric='logloss',
                        n_jobs=config.workers,random_state=config.seed+2,scale_pos_weight=20)
    model.fit(x,y)
    model.set_params(n_jobs=1)
    artifact={'model':model,'feature_names':COARSE_NAMES,'fit_source1_ids':train_ids.tolist(),
              'fit_pairs':len(y),'positive_pairs':int(y.sum()),'holdout_source1_ids':holdout.tolist(),
              'config':{'posting_limit':config.retrieval_posting_limit},'runtime_seconds':time.time()-started}
    joblib.dump(artifact,path)
    del x,y
    stats={'true_pairs':0,'raw_retained':0,'anchors':0,'raw_candidates':0};caps=(8,16,32,64,128)
    for cap in caps:stats[f'retained_top_{cap}']=0
    for start in range(0,len(holdout),100):
        anchors=fetch_records(blocker.con,'anchors',holdout[start:start+100]);truth=blocker.get_truth(anchors)
        rows,masks,offsets,text,indexer,repeated=blocker.raw_retrieve(anchors)
        probabilities=model.predict_proba(coarse_features(anchors,text,indexer,repeated,masks))[:,1]
        for i,anchor in enumerate(anchors.itertuples(index=False)):
            lo,hi=offsets[i:i+2];actual=truth[anchor.entity_id]
            good=np.asarray([target in actual for target in text['entity_id'][indexer[lo:hi]]],dtype=bool)
            stats['true_pairs']+=len(actual);stats['raw_retained']+=int(good.sum());stats['anchors']+=1;stats['raw_candidates']+=int(hi-lo)
            order=np.argsort(-probabilities[lo:hi],kind='stable')
            for cap in caps:stats[f'retained_top_{cap}']+=int(good[order[:cap]].sum())
    stats['recall']={str(cap):stats[f'retained_top_{cap}']/max(1,stats['true_pairs']) for cap in caps}
    stats['raw_recall']=stats['raw_retained']/max(1,stats['true_pairs'])
    stats['split_scope']='Held-out fitting entities; no calibration/validation/test labels used'
    stats['fit_pairs']=artifact['fit_pairs'];stats['fit_positives']=artifact['positive_pairs']
    (Path(config.reports_dir)/'retrieval_ranker_pilot.json').write_text(json.dumps(stats,indent=2))
    log.info('Retrieval ranker held-out fitting results: %s',stats)
    blocker.close();return artifact
