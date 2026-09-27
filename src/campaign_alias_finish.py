"""Finish alias scoring and export; wait for validated main output if needed."""
import argparse,json,time
from pathlib import Path
from .campaign_finish import run_stage
from .rescoring import filehash
OUT=Path('reports/campaign_0931')
def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1);p.add_argument('--export-only',action='store_true');args=p.parse_args();root=Path.cwd().resolve();output=Path('outputs/campaign_0931_oof_alias_unique');state=OUT/'alias_production_plan.json';plan={'work':'work/campaign_0931/test_alias_v2','output':str(output),'state':'scoring' if not args.export_only else 'waiting_for_scoring','preserves_existing_matches':True};state.write_text(json.dumps(plan,indent=2),encoding='utf-8')
 if not args.export_only:run_stage('alias_production_scoring',['-m','src.campaign_alias_scoring','--workers',str(args.workers)])
 print('Waiting for completed alias scores and officially validated main output',flush=True);started=time.time()
 while True:
  try:
   main_plan=json.loads((OUT/'production_plan.json').read_text());complete=json.loads(Path('work/campaign_0931/test_alias_v2/SCORING_COMPLETE.json').read_text())
   if main_plan['state']=='ready_for_amazon_evaluation' and complete['anchors']==1732544:break
  except (OSError,ValueError,KeyError):pass
  if time.time()-started>21600:raise RuntimeError('Upstream scoring did not finish within6hours; inspect logs and resume')
  time.sleep(2)
 plan['state']='export';state.write_text(json.dumps(plan,indent=2),encoding='utf-8');valid=output/'validation.json'
 if valid.exists():
  report=json.loads(valid.read_text())
  if filehash(output/'matching_results.tsv')!=report['matching_sha256'] or filehash(output/'candidate_pairs.tsv')!=report['candidate_sha256']:raise ValueError('Existing alias export changed')
 else:
  if output.exists() and any(output.iterdir()):
   archive=Path('work/campaign_0931/interrupted_exports')/(output.name+'_'+str(time.time_ns()))
   if not output.resolve().is_relative_to(root) or not archive.resolve().is_relative_to(root):raise ValueError('Archive paths must stay in repository')
   archive.parent.mkdir(parents=True,exist_ok=True);output.rename(archive)
  run_stage('alias_production_export',['-m','src.campaign_alias_export'])
 plan['state']='official_validation';state.write_text(json.dumps(plan,indent=2),encoding='utf-8');run_stage('alias_official_validation',['vendor/student_resource/utils/validate_submission.py','--matching',str(output/'matching_results.tsv'),'--test-dir','dataset/test','--check-ids']);plan['state']='ready_for_amazon_evaluation';plan['validation']=json.loads(valid.read_text());state.write_text(json.dumps(plan,indent=2),encoding='utf-8');print('READY',str(output/'matching_results.tsv'),flush=True)
if __name__=='__main__':main()
