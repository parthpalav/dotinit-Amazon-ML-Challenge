"""Compare two complete submissions without treating changed predictions as gains."""
import argparse,collections,json
from pathlib import Path
from .disk_store import connect
from .config import Config


def compare(baseline,current,output):
 cfg=Config.load('config/windows.json')
 con=connect(Path(cfg.working_dir)/'test/records.sqlite',True)
 counters=collections.defaultdict(collections.Counter)
 with Path(baseline).open(encoding='utf-8') as before,Path(current).open(encoding='utf-8') as after:
  expected='source1_entity_id\tmatched_entity_ids'
  if next(before).rstrip('\r\n')!=expected or next(after).rstrip('\r\n')!=expected:raise ValueError('Unexpected header')
  for source,country in con.execute('select entity_id,country from anchors order by rid'):
   a=next(before).rstrip('\r\n').split('\t');b=next(after).rstrip('\r\n').split('\t')
   if a[0]!=source or b[0]!=source or len(a)!=2 or len(b)!=2:raise ValueError('Rows must have matching source order')
   old=set(a[1].split(',')) if a[1] else set();new=set(b[1].split(',')) if b[1] else set()
   for c in (counters[country],counters['ALL']):
    c['anchors']+=1;c['old_matches']+=len(old);c['new_matches']+=len(new);c['retained_matches']+=len(old&new);c['added_matches']+=len(new-old);c['removed_matches']+=len(old-new);c['changed_anchors']+=old!=new;c['old_singletons']+=not old;c['new_singletons']+=not new;c['singleton_to_matched']+=not old and bool(new);c['matched_to_singleton']+=bool(old) and not new
  if next(before,None) is not None or next(after,None) is not None:raise ValueError('Extra rows')
 con.close()
 result={'baseline':str(baseline),'current':str(current),'interpretation':'Prediction changes only. Test labels are unavailable; added/removed matches are not measured true/false positives. France has no labeled confirmation estimate.','by_country':dict(counters)}
 Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--current',required=True);p.add_argument('--output',required=True);a=p.parse_args();compare(a.baseline,a.current,a.output)
