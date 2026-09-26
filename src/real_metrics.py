"""Vectorized entity metrics and held-out probability diagnostics."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss, confusion_matrix


def entity_scores(anchor_indices,labels,probabilities,truth_counts,threshold,target_count):
    selected=np.asarray(probabilities)>=threshold
    n=len(truth_counts)
    predicted=np.bincount(anchor_indices[selected],minlength=n)
    tp=np.bincount(anchor_indices[selected & (labels==1)],minlength=n)
    actual=np.asarray(truth_counts)
    both_empty=(actual==0)&(predicted==0)
    precision=np.divide(tp,predicted,out=np.zeros(n,dtype=float),where=predicted>0)
    recall=np.divide(tp,actual,out=np.zeros(n,dtype=float),where=actual>0)
    precision[both_empty]=1;recall[both_empty]=1
    f05=np.divide(1.25*tp,.25*actual+predicted,out=np.zeros(n,dtype=float),where=(.25*actual+predicted)>0)
    f1=np.divide(2*tp,actual+predicted,out=np.zeros(n,dtype=float),where=(actual+predicted)>0)
    f05[both_empty]=1;f1[both_empty]=1
    tp_total=int(tp.sum());fp=int(predicted.sum()-tp_total);fn=int(actual.sum()-tp_total)
    singleton=actual==0;negatives=n*target_count-int(actual.sum())
    return {'precision':float(precision.mean()),'recall':float(recall.mean()),'f0.5':float(f05.mean()),'f1':float(f1.mean()),
            'singleton_accuracy':float(both_empty[singleton].mean()) if singleton.any() else None,
            'true_positive':tp_total,'false_positive':fp,'false_negative':fn,
            'true_negative_all_possible':negatives-fp,'false_positive_rate':fp/negatives if negatives else 0,
            'predicted_positive':int(selected.sum()),'predicted_negative_candidates':int((~selected).sum()),
            'predicted_positive_rate':float(selected.mean()) if len(selected) else 0}


def tune(anchor_indices,labels,probabilities,truth_counts,target_count):
    thresholds=np.r_[np.arange(.05,.951,.01),.975,.99,.995,.999,1.,np.nextafter(1.,2.)]
    table=pd.DataFrame([{'threshold':float(t),**entity_scores(anchor_indices,labels,probabilities,truth_counts,t,target_count)} for t in thresholds])
    best=table.sort_values(['f0.5','precision','threshold'],ascending=False,kind='stable').iloc[0]
    return float(best.threshold),table


def probability_diagnostics(labels,probabilities,threshold):
    labels=np.asarray(labels);probabilities=np.asarray(probabilities,dtype=float)
    if not len(labels):return {'pr_auc':None,'roc_auc':None,'brier':None,'log_loss':None,'confusion_matrix':[[0,0],[0,0]]}
    both=len(np.unique(labels))==2
    return {'pr_auc':float(average_precision_score(labels,probabilities)) if labels.sum() else None,
            'roc_auc':float(roc_auc_score(labels,probabilities)) if both else None,
            'brier':float(brier_score_loss(labels,probabilities)),
            'log_loss':float(log_loss(labels,np.clip(probabilities,1e-8,1-1e-8),labels=[0,1])),
            'confusion_matrix':confusion_matrix(labels,probabilities>=threshold,labels=[0,1]).tolist()}
