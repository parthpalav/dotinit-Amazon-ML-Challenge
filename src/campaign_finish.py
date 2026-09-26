"""Complete the confirmed campaign: resumable scoring, export, official validation.
Run only after confirmation_v4 completes. Each stage logs and prints progress.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from .rescoring import filehash
OUT=Path('reports/campaign_0931')
def run_stage(name,args):
 print('START',name,flush=True);log=OUT/(name+'.log');started=time.time()
 with log.open('a',encoding='utf-8') as f:
  f.write('\nRESUME/RUN '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');f.flush()
  proc=subprocess.Popen([sys.executable,'-u',*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env={**os.environ,'PYTHONUTF8':'1'})
  for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
  code=proc.wait()
 if code:raise RuntimeError(f'{name} failed with code {code}; see {log}')
 print('DONE',name,round(time.time()-started,1),'seconds',flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);args=p.parse_args();root=Path.cwd().resolve()
 if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='parth':raise ValueError('Campaign execution requires parth')
 results=json.loads((OUT/'confirmation_v4_results.json').read_text())
 name='oof' if results['oof']['ci_vs_compact'][0]>0 else 'compact'
 frozen=OUT/'frozen_confirmation_v4.json';recipe=next(r for r in json.loads(frozen.read_text())['models'] if r['name']==name)
 if results[name]['paired_95ci'][0]<=0 or filehash(recipe['path'])!=recipe['sha256']:raise ValueError('Model is not confirmed')
 work=Path('work/campaign_0931')/('test_final_'+name);output=Path('outputs')/('campaign_0931_'+name+'_unique');plan={'name':name,'model':recipe,'work':str(work),'output':str(output),'reason':'OOF selected only if its paired 95% CI versus compact is strictly positive; otherwise use confirmed compact.','state':'scoring'}
 state=OUT/'production_plan.json'
 if state.exists():
  old=json.loads(state.read_text())
  if old['name']!=name or old['model']!=recipe:raise ValueError('A different production recipe already exists')
 state.write_text(json.dumps(plan,indent=2),encoding='utf-8')
 run_stage('production_scoring',['-m','src.campaign_scoring','--workers',str(args.workers),'--work',str(work),'--model-name',name,'--frozen',str(frozen),'--confirmation',str(OUT/'confirmation_v4_results.json')])
 plan['state']='export';state.write_text(json.dumps(plan,indent=2),encoding='utf-8')
 valid=output/'validation.json'
 if valid.exists():
  report=json.loads(valid.read_text())
  if report['model_sha256']!=recipe['sha256'] or report['matching_sha256']!=filehash(output/'matching_results.tsv'):raise ValueError('Existing export changed')
 else:
  if output.exists() and any(output.iterdir()):
   archive=Path('work/campaign_0931/interrupted_exports')/(output.name+'_'+str(time.time_ns()))
   if not output.resolve().is_relative_to(root) or not archive.resolve().is_relative_to(root):raise ValueError('Archive paths must stay in the working repository')
   archive.parent.mkdir(parents=True,exist_ok=True);output.rename(archive)
  run_stage('production_export',['-m','src.finalize_improved','--work',str(work),'--selection',str(work/'selection.json'),'--output',str(output),'--unique-owner'])
 plan['state']='official_validation';state.write_text(json.dumps(plan,indent=2),encoding='utf-8')
 run_stage('production_official_validation',['vendor/student_resource/utils/validate_submission.py','--matching',str(output/'matching_results.tsv'),'--test-dir','dataset/test','--check-ids'])
 plan['state']='ready_for_amazon_evaluation';plan['validation']=json.loads(valid.read_text());state.write_text(json.dumps(plan,indent=2),encoding='utf-8');print('READY',str(output/'matching_results.tsv'),flush=True)
if __name__=='__main__':main()
