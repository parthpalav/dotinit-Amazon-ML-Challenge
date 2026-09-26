"""Bounded, resumable import of read-only reference scores and graph audit."""
import argparse,json,sqlite3,time
from pathlib import Path
import numpy as np
DTYPE=np.dtype([('anchor','<u4'),('target','<u4'),('p','<f8')])
ROOT=Path('work/campaign_0931');REPORT=Path('reports/campaign_0931')

def main():
 ROOT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 source=Path('F:/dotinit-Amazon-ML-Challenge/work/improvements/test_scores_d10_direct')
 n=55431940;path=ROOT/'test_scores.npy';state=ROOT/'score_import.json'
 if state.exists():info=json.loads(state.read_text());start=info['next_anchor'];offset=info['pairs']
 else:start=0;offset=0
 a=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=DTYPE,shape=(n,));started=time.time()
 for anchor in range(start,1732544,100):
  block=np.load(source/f'{anchor:08d}.npy')
  if block.dtype!=DTYPE or not np.isfinite(block['p']).all():raise ValueError('Invalid reference shard')
  a[offset:offset+len(block)]=block;offset+=len(block)
  if (anchor//100+1)%100==0 or anchor+100>=1732544:
   a.flush();info={'next_anchor':min(anchor+100,1732544),'pairs':offset,'seconds_this_run':time.time()-started,'source':str(source)}
   state.write_text(json.dumps(info,indent=2));print('IMPORTED',json.dumps(info),flush=True)
 if offset!=n:raise ValueError('Incomplete pair coverage')
 (REPORT/'score_import.json').write_text(json.dumps(info,indent=2))
 con=sqlite3.connect('file:work/windows_v1/test/records.sqlite?mode=ro',uri=True)
 countries=np.zeros(1732545,np.uint8);vocab={'US':1,'India':2,'France':3}
 for rid,country in con.execute('select rid,country from anchors'):countries[rid]=vocab[country]
 np.save(ROOT/'test_anchor_countries.npy',countries);con.close()
 best=np.zeros(9969590);second=np.zeros_like(best);ties=np.zeros(len(best),np.uint16)
 for lo in range(0,n,1000000):
  b=a[lo:lo+1000000];np.maximum.at(best,b['target'],b['p'])
 for lo in range(0,n,1000000):
  b=a[lo:lo+1000000];win=b['p']==best[b['target']];np.add.at(ties,b['target'][win],1);np.maximum.at(second,b['target'][~win],b['p'][~win])
 np.save(ROOT/'target_best.npy',best);np.save(ROOT/'target_second.npy',second);np.save(ROOT/'target_ties.npy',ties)
 counts={c:{'anchors':int((countries==k).sum()),'pair_matches':0,'unique_matches':0,'small_margin_unique':0,'pair_empty':0,'unique_empty':0} for c,k in vocab.items()}
 pair_counts=np.zeros(len(countries),np.uint16);unique_counts=np.zeros(len(countries),np.uint16)
 for lo in range(0,n,1000000):
  b=a[lo:lo+1000000];accept=b['p']>=.6750000000000003;unique=accept&(b['p']==best[b['target']])&(ties[b['target']]==1)
  np.add.at(pair_counts,b['anchor'][accept],1);np.add.at(unique_counts,b['anchor'][unique],1)
  for country,k in vocab.items():
   mask=countries[b['anchor']]==k;counts[country]['pair_matches']+=int((accept&mask).sum());counts[country]['unique_matches']+=int((unique&mask).sum());counts[country]['small_margin_unique']+=int((unique&mask&((b['p']-second[b['target']])<.1)).sum())
 for country,k in vocab.items():counts[country]['pair_empty']=int(((countries==k)&(pair_counts==0)).sum());counts[country]['unique_empty']=int(((countries==k)&(unique_counts==0)).sum())
 (REPORT/'full_graph_audit.json').write_text(json.dumps(counts,indent=2));print('GRAPH',json.dumps(counts),flush=True)

if __name__=='__main__':main()
