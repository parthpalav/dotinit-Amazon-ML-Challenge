"""Real-data orchestration reusing the existing normalization, features and four models.

The disk backend bounds memory and reports entity sampling/top-k retrieval explicitly.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import joblib
import numpy as np
import pandas as pd
from .config import Config
from .disk_store import build_store,connect,fetch_records,rss_mb,COLUMNS
from .disk_blocking import DiskBlocker
from .features import FeatureEngineer
from .model import model_candidates,CalibratedMatcher
from .preprocessing import preprocess
from .real_metrics import tune,entity_scores,probability_diagnostics

LOG=logging.getLogger(__name__)
_WORKER={}


def initialize_worker(config_dict,split,engineer_path=None,model_path=None):
    config=Config(**config_dict)
    _WORKER['config']=config;_WORKER['blocker']=DiskBlocker(config,split)
    if engineer_path:_WORKER['engineer']=joblib.load(engineer_path)
    if model_path:
        _WORKER['artifact']=joblib.load(model_path)
        _WORKER['engineer']=_WORKER['artifact']['feature_engineer']
        estimator=_WORKER['artifact']['matcher'].estimator
        # Parallelize anchors across processes; avoid a nested forest/XGBoost
        # thread pool in every inference worker. Predictions are unchanged.
        if hasattr(estimator,'get_params') and 'n_jobs' in estimator.get_params(deep=False):
            estimator.set_params(n_jobs=1)


def process_batch(ids,mode='features'):
    blocker=_WORKER['blocker'];anchors=fetch_records(blocker.con,'anchors',ids)
    pairs,targets,stats,truth=blocker.retrieve(anchors,with_truth=mode!='infer',fetch_targets=mode!='audit')
    stats['worker_peak_rss_mb']=rss_mb()
    if mode=='audit':return stats
    if len(pairs):
        records=preprocess(pd.concat([anchors[COLUMNS[:4]],targets[COLUMNS[:4]]],ignore_index=True))
        engineer=_WORKER['engineer'];features=engineer.transform(pairs,engineer.prepare(records))
    else:
        features=pd.DataFrame(columns=_WORKER['engineer'].feature_names or [])
    if mode=='features':
        return pairs,features,stats,[(a,len(truth[a])) for a in anchors.entity_id]
    artifact=_WORKER['artifact']
    probabilities=artifact['matcher'].predict(features)
    if len(features) and features.columns.tolist()!=artifact['feature_names']:raise ValueError('Feature schema changed')
    selected=probabilities>=artifact['threshold']
    grouped={}
    for (source,target),probability in zip(zip(pairs.source1_entity_id,pairs.candidate_entity_id),probabilities):
        grouped.setdefault(source,[]).append((target,float(probability)))
    matching=[];candidates=[]
    for source in anchors.entity_id:
        values=grouped.get(source,[])
        accepted=sorted(((t,p) for t,p in values if p>=artifact['threshold']),key=lambda v:(-v[1],v[0]))
        matching.append(source+'\t'+','.join(t for t,p in accepted)+'\n')
        candidates.append(source+'\t'+','.join(t for t,p in values)+'\n')
    stats['counts']['predicted_matches']=int(selected.sum())
    stats['counts']['predicted_nonmatches']=int((~selected).sum())
    stats['counts']['predicted_singletons']=sum(line.rstrip('\n').endswith('\t') for line in matching)
    stats['counts']['scored_pairs']=len(probabilities)
    return ''.join(matching),''.join(candidates),stats


def pool_results(config,split,batches,mode,engineer_path=None,model_path=None):
    with ProcessPoolExecutor(max_workers=config.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(config.to_dict(),split,engineer_path,model_path)) as pool:
        # Bound submitted work as well as batch size. Executor.map eagerly queues
        # all batches on Python <=3.13 and can retain large completed results.
        batches=iter(batches)
        pending=deque()
        for _ in range(config.workers*2):
            batch=next(batches,None)
            if batch is None:break
            pending.append(pool.submit(process_batch,batch,mode))
        while pending:
            yield pending.popleft().result()
            batch=next(batches,None)
            if batch is not None:pending.append(pool.submit(process_batch,batch,mode))


def aggregate(acc,item):
    for section in ('counts','raw_rule_true','final_rule_true','rule_candidates'):
        dest=acc.setdefault(section,{})
        for key,value in item.get(section,{}).items():
            if key.startswith('max_'):dest[key]=max(dest.get(key,0),int(value))
            else:dest[key]=dest.get(key,0)+int(value)
    acc['worker_peak_rss_mb']=max(acc.get('worker_peak_rss_mb',0),item.get('worker_peak_rss_mb',0))


def enrich_stats(stats,target_count):
    counts=stats['counts'];n=counts.get('anchors',0);total=counts.get('true_pairs',0)
    stats['candidate_recall']=counts.get('retained_true',0)/total if total else None
    stats['raw_candidate_recall']=counts.get('raw_retained_true',0)/total if total else None
    stats['average_candidates']=counts.get('candidate_pairs',0)/n if n else 0
    stats['candidate_reduction_ratio']=1-counts.get('candidate_pairs',0)/(n*target_count) if n and target_count else 1
    stats['possible_pairs_removed']=n*target_count-counts.get('candidate_pairs',0)
    return stats


def prepare(config):
    manifests={split:build_store(config,split) for split in ('train','test')}
    from .supplement import build_supplement
    for split in ('train','test'):build_supplement(config,split)
    return manifests


def choose_entities(config,manifest):
    path=Path(config.working_dir)/'entity_selection.npz'
    if path.exists():return dict(np.load(path))
    n=manifest['counts']['source1'];rng=np.random.default_rng(config.seed)
    order=rng.permutation(n)+1
    nv=int(n*config.validation_fraction);nc=int(n*config.calibration_fraction)
    groups={'validation':order[:nv],'calibration':order[nv:nv+nc],'fit':order[nv+nc:]}
    limits={'fit':config.fit_anchor_limit,'calibration':config.calibration_anchor_limit,'validation':config.validation_anchor_limit}
    selected={name:np.sort(ids[:limits[name]]).astype(np.uint32) for name,ids in groups.items()}
    np.savez(path,**selected)
    details={name:{'assigned_entities':len(ids),'sampled_entities':len(selected[name])} for name,ids in groups.items()}
    Path(config.reports_dir).mkdir(parents=True,exist_ok=True)
    (Path(config.reports_dir)/'entity_sampling.json').write_text(json.dumps({'seed':config.seed,'groups':details,
        'method':'Seeded permutation of every training S1; disjoint group assignment then bounded S1 sampling. Every retrieved candidate of each sampled S1 retained.'},indent=2))
    con=connect(Path(config.working_dir)/'train/records.sqlite',readonly=True)
    # Complete manifest: no training record silently disappears from the audit.
    codes=np.empty(n,dtype=np.uint8);used=np.zeros(n,dtype=bool)
    for code,(name,ids) in enumerate(groups.items()):codes[ids-1]=code;used[selected[name]-1]=True
    names=list(groups)
    with (Path(config.reports_dir)/'split_manifest.tsv').open('w',encoding='utf-8') as stream:
        stream.write('source1_entity_id\tsplit\tused_for_estimator_or_selection\n')
        for rid,entity in con.execute('SELECT rid,entity_id FROM anchors ORDER BY rid'):
            stream.write(f'{entity}\t{names[codes[rid-1]]}\t{int(used[rid-1])}\n')
    con.close();LOG.info('Entity sampling: %s',details);return selected


def fit_engineer(config,selected,manifest):
    path=Path(config.working_dir)/'feature_engineer.joblib'
    if path.exists():return path
    con=connect(Path(config.working_dir)/'train/records.sqlite',readonly=True)
    anchors=fetch_records(con,'anchors',selected['fit'])
    rng=np.random.default_rng(config.seed+71)
    count=manifest['counts']['source2']+manifest['counts']['source3']
    target_ids=rng.choice(count,size=min(70000,count),replace=False)+1
    targets=fetch_records(con,'targets',target_ids)
    records=preprocess(pd.concat([anchors[COLUMNS[:4]],targets[COLUMNS[:4]]],ignore_index=True))
    engineer=FeatureEngineer(config.tfidf_max_features).fit(records)
    # Initialize stable column order using a schema-only pair; it is never labeled or trained on.
    if len(anchors) and len(targets):
        tiny=records.iloc[[0,len(anchors)]]
        schema_pair=pd.DataFrame([[anchors.entity_id.iloc[0],targets.entity_id.iloc[0],'strong_name']],
            columns=['source1_entity_id','candidate_entity_id','blocking_rules'])
        engineer.transform(schema_pair,engineer.prepare(tiny))
    joblib.dump(engineer,path)
    (Path(config.reports_dir)/'tfidf_corpus.json').write_text(json.dumps({'fit_s1_rows':len(anchors),'random_training_reference_rows':len(targets),
        'test_rows':0,'vocabularies':{k:len(v.vocabulary_) if v is not None else 0 for k,v in engineer.vectorizers.items()}},indent=2))
    con.close();return path


def feature_subset(config,name,ids,engineer_path,target_count):
    directory=Path(config.working_dir)/'features';directory.mkdir(exist_ok=True)
    path=directory/f'{name}.joblib'
    if path.exists():return joblib.load(path)
    batches=[ids[start:start+config.anchor_batch_size] for start in range(0,len(ids),config.anchor_batch_size)]
    all_pairs=[];all_features=[];truth_counts=[];stats={};started=time.time()
    for i,(pairs,features,summary,counts) in enumerate(pool_results(config,'train',batches,'features',str(engineer_path))):
        all_pairs.append(pairs);all_features.append(features);truth_counts.extend(counts);aggregate(stats,summary)
        if (i+1)%10==0:LOG.info('%s feature batches=%d/%d pairs=%d elapsed=%.1fs',name,i+1,len(batches),stats['counts']['candidate_pairs'],time.time()-started)
    result={'pairs':pd.concat(all_pairs,ignore_index=True),'features':pd.concat(all_features,ignore_index=True),
            'truth_counts':dict(truth_counts),'stats':enrich_stats(stats,target_count)}
    joblib.dump(result,path)
    (Path(config.reports_dir)/f'{name}_blocking.json').write_text(json.dumps(result['stats'],indent=2))
    return result


def train_real(config):
    started=time.time();reports=Path(config.reports_dir);reports.mkdir(parents=True,exist_ok=True)
    manifest=build_store(config,'train');target_count=manifest['counts']['source2']+manifest['counts']['source3']
    from .supplement import build_supplement
    from .coarse_ranker import fit_candidate_ranker
    build_supplement(config,'train')
    selected=choose_entities(config,manifest)
    retrieval_ranker=fit_candidate_ranker(config,selected)
    engineer_path=fit_engineer(config,selected,manifest)
    data={name:feature_subset(config,name,ids,engineer_path,target_count) for name,ids in selected.items()}
    for name,part in data.items():
        recall=part['stats']['candidate_recall']
        if recall is not None and recall<config.minimum_candidate_recall:
            message=f'{name} candidate recall {recall:.4%} is below target {config.minimum_candidate_recall:.2%}'
            if config.fail_on_low_candidate_recall:raise ValueError(message)
            LOG.warning(message)
    labels={name:part['pairs'].label.to_numpy(dtype=np.int8) for name,part in data.items()}
    if len(np.unique(labels['fit']))<2:raise ValueError('Real fitting candidates need both classes')
    validation=data['validation'];truth=validation['truth_counts'];positions={entity:i for i,entity in enumerate(truth)}
    anchor_indices=np.asarray([positions[x] for x in validation['pairs'].source1_entity_id],dtype=np.int32)
    truth_counts=np.asarray(list(truth.values()),dtype=np.int32)
    experiments=[];calibration_rows=[];fitted={};feature_names=data['fit']['features'].columns.tolist()
    for name,estimator in model_candidates(config,labels['fit']).items():
        model_started=time.time();LOG.info('Fitting %s on %d real pairs (%d positive, %d negative)',name,len(labels['fit']),int(labels['fit'].sum()),int((labels['fit']==0).sum()))
        estimator.fit(data['fit']['features'],labels['fit'])
        matcher=CalibratedMatcher(estimator)
        raw=matcher.predict(validation['features'])
        if name!='exact_rule':matcher.calibrate(data['calibration']['features'],labels['calibration'])
        calibrated=matcher.predict(validation['features'])
        options=[]
        for variant,probabilities in (('raw',raw),('calibrated',calibrated)):
            threshold,table=tune(anchor_indices,labels['validation'],probabilities,truth_counts,target_count)
            metrics=entity_scores(anchor_indices,labels['validation'],probabilities,truth_counts,threshold,target_count)
            diagnostics=probability_diagnostics(labels['validation'],probabilities,threshold)
            row={'model':name,'probability_variant':variant,'threshold':threshold,**metrics,**diagnostics}
            calibration_rows.append(row);options.append(row)
            table.to_csv(reports/f'thresholds_{name}_{variant}.tsv',sep='\t',index=False)
        # Retain calibration only when validation operating point improves, or reliability improves on tied F0.5.
        chosen=sorted(options,key=lambda r:(-r['f0.5'],-r['precision'],r['brier'],r['probability_variant']))[0]
        if chosen['probability_variant']=='raw':matcher.calibrator=None;matcher.calibration_status+=';raw_selected_on_validation'
        experiments.append({**chosen,'candidate_recall':validation['stats']['candidate_recall'],
            'average_candidates':validation['stats']['average_candidates'],'runtime_seconds':time.time()-model_started})
        fitted[name]={'matcher':matcher,'threshold':chosen['threshold']}
        pd.DataFrame(experiments).to_csv(reports/'model_comparison.tsv',sep='\t',index=False)
        LOG.info('%s macro F0.5=%.6f threshold=%.6f precision=%.6f recall=%.6f',name,chosen['f0.5'],chosen['threshold'],chosen['precision'],chosen['recall'])
    best=sorted(experiments,key=lambda r:(-r['f0.5'],-r['precision'],r['model']))[0];winner=best['model']
    pd.DataFrame(calibration_rows).to_csv(reports/'calibration_comparison.tsv',sep='\t',index=False)
    artifact={'artifact_version':2,'backend':'disk','model_name':winner,'config':config.to_dict(),
              'feature_engineer':joblib.load(engineer_path),'feature_names':feature_names,'uses_embeddings':False,
              'retrieval_ranker':retrieval_ranker,
              'validation_metrics':best,**fitted[winner]}
    destination=Path(config.model_path);destination.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(artifact,destination)
    run={'selected_model':winner,'selected_threshold':best['threshold'],'validation':best,'config':config.to_dict(),
         'rows':manifest['counts'],'training_candidate_labels':{n:{'positive':int(y.sum()),'negative':int((y==0).sum())} for n,y in labels.items()},
         'features':feature_names,'runtime_seconds':time.time()-started,'peak_parent_rss_mb':rss_mb(),
         'versions':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','xgboost','rapidfuzz','joblib')},
         'data_origin':'Only real competition training records; no synthetic data, test labels, embeddings or external lookups',
         'validation_scope':'Disjoint S1 validation sample, used for model/calibration/threshold selection; not an untouched generalization estimate'}
    (reports/'real_run_manifest.json').write_text(json.dumps(run,indent=2))
    (reports/'feature_names.json').write_text(json.dumps(feature_names,indent=2))
    LOG.info('Selected %s using validation F0.5; artifact=%s',winner,destination);return run


def audit_blocking(config):
    manifest=build_store(config,'train');total=manifest['counts']['source1'];target_count=manifest['counts']['source2']+manifest['counts']['source3']
    batches=[np.arange(start+1,min(start+config.anchor_batch_size,total)+1,dtype=np.uint32) for start in range(0,total,config.anchor_batch_size)]
    stats={};started=time.time()
    for i,summary in enumerate(pool_results(config,'train',batches,'audit')):
        aggregate(stats,summary)
        if (i+1)%50==0:LOG.info('Full training blocking audit anchors=%d/%d recall=%.4f elapsed=%.1fs',stats['counts']['anchors'],total,
            stats['counts']['retained_true']/max(1,stats['counts']['true_pairs']),time.time()-started)
    enrich_stats(stats,target_count);stats['seconds']=time.time()-started
    (Path(config.reports_dir)/'full_training_blocking.json').write_text(json.dumps(stats,indent=2))
    return stats


def infer_real(config):
    artifact=joblib.load(config.model_path)
    if artifact.get('artifact_version')!=2:raise ValueError('Expected a disk-backend artifact')
    # Preserve all model/retrieval settings from training; runtime location overrides only.
    trained=Config(**artifact['config'])
    config=replace(trained,dataset_dir=config.dataset_dir,working_dir=config.working_dir,output_dir=config.output_dir,
                   reports_dir=config.reports_dir,model_path=config.model_path,resource_dir=config.resource_dir,workers=config.workers)
    manifest=build_store(config,'test');total=manifest['counts']['source1']
    from .supplement import build_supplement
    build_supplement(config,'test')
    joblib.dump(artifact['retrieval_ranker'],Path(config.working_dir)/'candidate_ranker.joblib')
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    batches=[np.arange(start+1,min(start+config.anchor_batch_size,total)+1,dtype=np.uint32) for start in range(0,total,config.anchor_batch_size)]
    matching=output/'matching_results.tsv.partial';candidate=output/'candidate_pairs.tsv.partial';stats={};started=time.time()
    with matching.open('w',encoding='utf-8',newline='') as mw,candidate.open('w',encoding='utf-8',newline='') as cw:
        mw.write('source1_entity_id\tmatched_entity_ids\n');cw.write('source1_entity_id\tcandidate_entity_ids\n')
        for i,(match_lines,candidate_lines,summary) in enumerate(pool_results(config,'test',batches,'infer',model_path=str(Path(config.model_path).resolve()))):
            mw.write(match_lines);cw.write(candidate_lines);aggregate(stats,summary)
            if (i+1)%50==0:
                mw.flush();cw.flush();LOG.info('Test inference anchors=%d/%d scored=%d matches=%d elapsed=%.1fs',stats['counts']['anchors'],total,
                    stats['counts']['scored_pairs'],stats['counts']['predicted_matches'],time.time()-started)
    if stats['counts']['anchors']!=total or stats['counts']['scored_pairs']!=stats['counts']['candidate_pairs']:
        raise ValueError('Inference/export coverage mismatch')
    matching.replace(output/'matching_results.tsv');candidate.replace(output/'candidate_pairs.tsv')
    enrich_stats(stats,manifest['counts']['source2']+manifest['counts']['source3']);stats['seconds']=time.time()-started
    (Path(config.reports_dir)/'test_inference.json').write_text(json.dumps(stats,indent=2));return stats


def official_validate(config):
    validator=Path(config.resource_dir or Path(config.dataset_dir).parent)/'utils/validate_submission.py'
    if not validator.is_file():raise FileNotFoundError(f'Official validator not found: {validator}')
    command=[sys.executable,str(validator),'--matching',str(Path(config.output_dir)/'matching_results.tsv'),
             '--candidate',str(Path(config.output_dir)/'candidate_pairs.tsv'),'--test-dir',str(Path(config.dataset_dir)/'test'),'--check-ids']
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path('.').resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    report=Path(config.reports_dir)/'official_validator.log';report.write_text(result.stdout)
    (Path(config.reports_dir)/'official_validator_command.json').write_text(json.dumps({'command':command,'exit_code':result.returncode,
        'validator_sha256':hashlib.sha256(validator.read_bytes()).hexdigest()},indent=2))
    LOG.info('Official validator exit=%d\n%s',result.returncode,result.stdout)
    if result.returncode:raise RuntimeError(f'Official validator failed; see {report}')
    return result.stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['quality','prepare','train','audit','infer','validate','run'])
    parser.add_argument('--config',default='config/real.json');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    config=Config.load(args.config);Path(config.reports_dir).mkdir(parents=True,exist_ok=True)
    if args.command in ('quality','train','run'):
        from .quality import quality_report
        quality_report(config)
    if args.command in ('prepare','run'):prepare(config)
    if args.command in ('train','run'):train_real(config)
    if args.command in ('audit','run'):audit_blocking(config)
    if args.command in ('infer','run'):infer_real(config)
    if args.command in ('validate','run'):official_validate(config)
    if args.command=='run':
        from .real_report import generate
        generate(config)


if __name__=='__main__':main()
