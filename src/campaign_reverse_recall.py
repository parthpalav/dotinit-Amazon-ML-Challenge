"""Positive-only retrieval coverage diagnostic, not an end-to-end F score."""
import json
import joblib,numpy as np,pandas as pd
from .campaign_reverse import Reverse,ROOT
from .disk_store import COLUMNS


def main():
    rev=Reverse('train');part=joblib.load('work/windows_v1/features/confirmation_v4.joblib');owners={};ids=list(part['truth_counts'])
    for lo in range(0,len(ids),800):
        b=ids[lo:lo+800]
        for source,cell in rev.con.execute('select entity_id,matches from labels where entity_id in ('+','.join('?' for _ in b)+')',b):
            for target in cell.split(',') if cell else []:owners[target]=source
    wanted=sorted(owners);found=set();best=set();forward=set(part['pairs'].loc[part['pairs'].label==1,'candidate_entity_id']);counts={}
    folder=ROOT/'reverse_recall_v4';folder.mkdir(exist_ok=True)
    for lo in range(0,len(wanted),400):
        path=folder/f'{lo:07d}.joblib';b=wanted[lo:lo+400]
        if not path.exists():
            t=pd.DataFrame(rev.con.execute('select * from targets where entity_id in ('+','.join('?' for _ in b)+') order by rid',b).fetchall(),columns=['rid',*COLUMNS]);candidates=rev.candidates(t)
            joblib.dump(candidates,path)
        candidates=joblib.load(path)
        for r in candidates.itertuples(index=False):
            if owners[r.candidate_entity_id]==r.source1_entity_id:found.add(r.candidate_entity_id)
        top=candidates.sort_values('reverse_strength',ascending=False,kind='stable').drop_duplicates('candidate_entity_id')
        for r in top.itertuples(index=False):
            if owners[r.candidate_entity_id]==r.source1_entity_id:best.add(r.candidate_entity_id)
        if lo%4000==0:print('REVERSE_RECALL',lo,len(wanted),flush=True)
    report={'scope':'Positive-only reverse-retrieval diagnostic on consumed v4. Labels choose evaluation queries and measure recovery, never influence retrieved candidates. Not an end-to-end metric or deployable candidate set.',
            'true_links':len(wanted),'forward_retained':len(forward),'reverse_top4_retained':len(found),'reverse_top1_retained':len(best),'union_retained':len(forward|found),'forward_omissions_recovered':len(found-forward),'still_missing':len(set(wanted)-(forward|found))}
    from pathlib import Path
    Path('reports/campaign_0942/reverse_recall_v4.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('REVERSE_RECALL_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
