"""Resume confirmed correction, export unique owners, and officially validate."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from .rescoring import filehash
ROOT=Path('reports/campaign_0942')


def stage(name,argv):
    print('START',name,flush=True)
    with (ROOT/(name+'.log')).open('a',encoding='utf-8') as f:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-u',*argv],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
        if proc.wait():raise RuntimeError(name+' failed; see log')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='parth':raise ValueError('Requires parth branch')
    report=json.loads((ROOT/'minimal_confirmation_v5.json').read_text(encoding='utf-8'))
    if not report['promotion_supported']:raise ValueError('Confirmation failed')
    work=Path('work/campaign_0942/test_minimal');out=Path('outputs/campaign_0942_raw_unique');state=ROOT/'production_plan.json'
    plan={'state':'scoring','output':str(out),'work':str(work),'frozen':report['frozen'],'amazon_score':None}
    state.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    stage('raw_production',['-m','src.campaign_raw_scoring','--workers',str(args.workers)])
    validation=out/'validation.json'
    if validation.exists():
        v=json.loads(validation.read_text(encoding='utf-8'))
        if v['matching_sha256']!=filehash(out/'matching_results.tsv') or v['model_sha256']!=report['frozen']['model_hash']:raise ValueError('Export differs from confirmed model')
    else:
        if out.exists() and any(out.iterdir()):
            archive=Path('work/campaign_0942/interrupted_exports')/(out.name+'_'+str(time.time_ns()))
            if not out.resolve().is_relative_to(Path.cwd().resolve()) or not archive.resolve().is_relative_to(Path.cwd().resolve()):raise ValueError('Unsafe archive location')
            archive.parent.mkdir(parents=True,exist_ok=True);out.rename(archive)
        stage('raw_export',['-m','src.finalize_improved','--work',str(work),'--selection',str(work/'selection.json'),'--output',str(out),'--unique-owner'])
    stage('raw_official_validation',['vendor/student_resource/utils/validate_submission.py','--matching',str(out/'matching_results.tsv'),'--test-dir','dataset/test','--check-ids'])
    plan.update(state='ready_for_amazon_evaluation',validation=json.loads(validation.read_text(encoding='utf-8')));state.write_text(json.dumps(plan,indent=2),encoding='utf-8');print('READY',str(out/'matching_results.tsv'),flush=True)


if __name__=='__main__':main()
