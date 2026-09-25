"""Disk candidate union, optional trained retrieval ranking, and audited top-k."""
from collections import Counter
import ctypes
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import process,fuzz
from .blocking import RULES
from .coarse_ranker import coarse_features
from .disk_store import NativeIndex,connect,fetch_records
from .supplement import PackedText


def rules_text(mask):
    return '|'.join(rule for bit,rule in enumerate(RULES) if int(mask)&(1<<bit))


class DiskBlocker:
    def __init__(self,config,split,load_ranker=True):
        self.config=config;self.split=split;self.directory=Path(config.working_dir)/split
        self.con=connect(self.directory/'records.sqlite',readonly=True)
        self.native=NativeIndex(Path(config.working_dir)/'native',self.directory/'blocking_index.bin')
        self.extra=None;self.packed=None;self.ranker=None
        if (self.directory/'supplement_manifest.json').exists():
            self.extra=NativeIndex(Path(config.working_dir)/'native',self.directory/'supplement_index.bin',supplemental=True)
            self.packed=PackedText(self.directory)
            self.merge=self.extra.lib.ber_merge
            self.merge.argtypes=[ctypes.c_void_p]*6+[ctypes.c_size_t]+[ctypes.c_void_p]*3
            self.merge.restype=ctypes.c_size_t
        path=Path(config.working_dir)/'candidate_ranker.joblib'
        if load_ranker and path.exists():self.ranker=joblib.load(path)['model']

    def raw_retrieve(self,anchors):
        rows,masks,offsets=self.native.lookup(self.native.keys(anchors),self.config.retrieval_posting_limit)
        if self.extra:
            er,em,eo=self.extra.lookup(self.extra.keys(anchors),self.config.retrieval_posting_limit)
            capacity=len(rows)+len(er);rr=np.empty(capacity,np.uint32);mm=np.empty(capacity,np.uint16);oo=np.empty(len(anchors)+1,np.uint32)
            used=self.merge(rows.ctypes.data,masks.ctypes.data,offsets.ctypes.data,er.ctypes.data,em.ctypes.data,eo.ctypes.data,
                            len(anchors),rr.ctypes.data,mm.ctypes.data,oo.ctypes.data)
            rows,masks,offsets=rr[:used],mm[:used],oo
        unique,indexer=np.unique(rows,return_inverse=True)
        if self.packed:text=self.packed.get(unique)
        else:
            frame=fetch_records(self.con,'targets',unique).set_index('rid').loc[unique]
            text={key:frame[key].to_numpy() for key in ('entity_id','name_norm','address_norm','country_norm')}
        text['rid']=unique
        repeated=np.repeat(np.arange(len(anchors)),np.diff(offsets))
        return rows,masks,offsets,text,indexer,repeated

    def get_truth(self,anchors):
        ids=anchors.entity_id.tolist();truth={}
        for start in range(0,len(ids),900):
            block=ids[start:start+900]
            truth.update((a,set(b.split(',')) if b else set()) for a,b in self.con.execute(
                f'SELECT entity_id,matches FROM labels WHERE entity_id IN ({",".join("?" for _ in block)})',block))
        if len(truth)!=len(anchors):raise ValueError('Missing anchor labels')
        return truth

    def retrieve(self,anchors,with_truth=False,fetch_targets=True):
        rids,masks,offsets,text,indexer,repeated=self.raw_retrieve(anchors)
        if len(rids) and self.ranker is not None:
            rank=self.ranker.predict_proba(coarse_features(anchors,text,indexer,repeated,masks,
                similarity_threads=min(2,self.config.threads)))[:,1]
        elif len(rids):
            na=anchors.name_norm.to_numpy()[repeated];nb=text['name_norm'][indexer]
            aa=anchors.address_norm.to_numpy()[repeated];ab=text['address_norm'][indexer]
            ns=process.cpdist(na,nb,scorer=fuzz.ratio,dtype=np.float32,workers=1)/100
            ads=process.cpdist(aa,ab,scorer=fuzz.token_set_ratio,dtype=np.float32,workers=1)/100
            ns[(na=='')|(nb=='')]=0;ads[(aa=='')|(ab=='')]=0
            rank=np.maximum(ns,ads)+.35*np.minimum(ns,ads)
        else:rank=np.empty(0)
        truth=self.get_truth(anchors) if with_truth else {}
        selected=[];raw_counts=[];raw_true=Counter();final_true=Counter();rule_counts=Counter();summary=Counter()
        for i,anchor in enumerate(anchors.itertuples(index=False)):
            lo,hi=int(offsets[i]),int(offsets[i+1]);raw_counts.append(hi-lo);keep=np.arange(lo,hi)
            if hi-lo>self.config.max_candidates_per_anchor:
                keep=keep[np.argsort(-rank[lo:hi],kind='stable')[:self.config.max_candidates_per_anchor]]
            keep=np.sort(keep);selected.extend(keep.tolist())
            summary['anchors']+=1;summary['raw_candidates']+=hi-lo;summary['candidate_pairs']+=len(keep)
            summary['no_candidate']+=not len(keep);summary['one_candidate']+=len(keep)==1;summary['multiple_candidates']+=len(keep)>1
            summary['topk_truncated_anchors']+=hi-lo>len(keep)
            if with_truth:
                actual=truth[anchor.entity_id];summary['true_pairs']+=len(actual);summary['true_singletons']+=not actual
                is_true=np.asarray([entity in actual for entity in text['entity_id'][indexer[lo:hi]]],dtype=bool)
                summary['raw_retained_true']+=int(is_true.sum());summary['retained_true']+=int(is_true[keep-lo].sum())
                for bit,rule in enumerate(RULES):
                    rule_raw=(masks[lo:hi]&(1<<bit))!=0
                    rule_counts[rule]+=int(rule_raw.sum());raw_true[rule]+=int((rule_raw&is_true).sum())
                    final_true[rule]+=int((rule_raw[keep-lo]&is_true[keep-lo]).sum())
        selected=np.asarray(selected,dtype=int)
        pairs=pd.DataFrame({'anchor_rid':anchors.rid.to_numpy()[repeated[selected]],'target_rid':rids[selected],
            'source1_entity_id':anchors.entity_id.to_numpy()[repeated[selected]],'candidate_entity_id':text['entity_id'][indexer[selected]],
            'blocking_rules':[rules_text(mask) for mask in masks[selected]]})
        if with_truth:pairs['label']=[int(b in truth[a]) for a,b in zip(pairs.source1_entity_id,pairs.candidate_entity_id)]
        summary['max_raw_candidates']=max(raw_counts,default=0)
        summary['max_candidates']=min(summary['max_raw_candidates'],self.config.max_candidates_per_anchor)
        stats={'counts':dict(summary),'raw_rule_true':dict(raw_true),'final_rule_true':dict(final_true),'rule_candidates':dict(rule_counts)}
        targets=fetch_records(self.con,'targets',rids[selected]) if fetch_targets else None
        return pairs,targets,stats,truth

    def close(self):
        self.native.close();self.con.close()
        if self.extra:self.extra.close()
