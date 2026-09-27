"""Run the two validated pipelines sequentially to avoid16GB RAM contention."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
import psutil
OUT=Path('reports/campaign_0931')
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ready(path):
 try:
  p=json.loads(path.read_text());v=p['validation'];root=Path(p['output'])
  return p['state']=='ready_for_amazon_evaluation' and digest(root/'matching_results.tsv')==v['matching_sha256'] and digest(root/'candidate_pairs.tsv')==v['candidate_sha256']
 except (OSError,ValueError,KeyError):return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--main-workers',type=int,default=6);p.add_argument('--alias-workers',type=int,default=4);args=p.parse_args()
 if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='parth':raise ValueError('Run only on parth')
 state=OUT/'coordinator.json'
 for name,module,plan in [('main','src.campaign_finish',OUT/'production_plan.json'),('alias','src.campaign_alias_finish',OUT/'alias_production_plan.json')]:
  if ready(plan):print('ALREADY_VALIDATED',name,flush=True);continue
  workers=args.main_workers if name=='main' else min(args.alias_workers,max(1,int((psutil.virtual_memory().available/1e9-1.5)/.8)))
  status={'stage':name,'workers':workers,'state':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};state.write_text(json.dumps(status,indent=2),encoding='utf-8');print('COORDINATOR_START',json.dumps(status),flush=True)
  code=subprocess.call([sys.executable,'-u','-m',module,'--workers',str(workers)],env={**os.environ,'PYTHONUTF8':'1'})
  if code:
   status.update(state='failed',exit_code=code);state.write_text(json.dumps(status,indent=2),encoding='utf-8');raise SystemExit(code)
  time.sleep(2)
 state.write_text(json.dumps({'state':'ready_for_amazon_evaluation','outputs':['outputs/campaign_0931_oof_unique','outputs/campaign_0931_oof_alias_unique']},indent=2),encoding='utf-8');print('BOTH_SUBMISSIONS_READY',flush=True)
if __name__=='__main__':main()
