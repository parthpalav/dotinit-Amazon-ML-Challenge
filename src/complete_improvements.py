"""Resume confirmed inference, export separate variants, and record completion/failure."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,traceback

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);parser.add_argument('--full-validator',action='store_true');args=parser.parse_args()
 if args.workers<1:raise ValueError('workers must be positive')
 root=Path('reports/improvements');root.mkdir(parents=True,exist_ok=True)
 status=root/'completion_status.json'
 def record(stage,state,**extra):
  value={'stage':stage,'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
  status.write_text(json.dumps(value,indent=2),encoding='utf-8')
  with Path('reports/IMPROVEMENT_STATUS.md').open('a',encoding='utf-8') as f:f.write('\n- Automated continuation: '+json.dumps(value)+'\n')
 def run(stage,args):
  record(stage,'running');log=root/(stage+'.log')
  with log.open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT,check=True,env={**os.environ,'PYTHONUTF8':'1'})
  record(stage,'passed',log=str(log))
 try:
  run('full_rescoring',['-m','src.rescoring','--model','artifacts/improvements/catboost_d10_evidence.joblib','--work','work/improvements/test_scores_d10_direct','--workers',str(args.workers),'--direct-provenance'])
  for suffix,flags in [('pair_threshold',[]),('unique_owner',['--unique-owner'])]:
   output=Path('outputs')/('improved_'+suffix)
   if (output/'validation.json').exists():
    existing=json.loads((output/'validation.json').read_text(encoding='utf-8'));frozen=json.loads((root/'frozen_selection.json').read_text(encoding='utf-8'))
    if existing['model_sha256']!=frozen['sha256']:raise ValueError('Existing export belongs to a different model')
   if not (output/'validation.json').exists():run('export_'+suffix,['-m','src.finalize_improved','--work','work/improvements/test_scores_d10_direct','--output',str(output),*flags])
   # Candidate TSV is exhaustively validated by the streaming exporter above.
   # The official checker additionally checks matching coverage/IDs. Avoid its
   # 55-million-string candidate dictionary on this 16 GB machine.
   run('official_matching_'+suffix,['vendor/student_resource/utils/validate_submission.py','--matching',str(output/'matching_results.tsv'),'--candidate',str(output/'candidate_pairs.tsv') if args.full_validator else str(root/'not_loaded_by_official_checker.tsv'),'--test-dir','dataset/test','--check-ids'])
  record('all','complete',scope='Two exported variants; streaming validation plus official '+('matching/candidate' if args.full_validator else 'matching-only')+' ID validation. Amazon score pending submission.')
 except Exception as exc:
  record('continuation','failed',error=str(exc));traceback.print_exc();raise

if __name__=='__main__':main()
