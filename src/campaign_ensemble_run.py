"""Resume the confirmed neural pipeline through final validated submissions."""
from pathlib import Path
import argparse,json,subprocess,time
from .campaign_raw_finish import stage
from .campaign_neural_test import ROOT,OUT,verify_frozen
from .rescoring import filehash


def export(kind):
    work=ROOT/('test_'+kind);out=Path('outputs')/('campaign_0942_'+kind+'_unique');validation=out/'validation.json';selection=json.loads((work/'selection.json').read_text(encoding='utf-8'))
    if validation.exists():
        v=json.loads(validation.read_text(encoding='utf-8'))
        if v['model_sha256']!=selection['sha256'] or v['matching_sha256']!=filehash(out/'matching_results.tsv'):raise ValueError('Completed output changed')
    else:
        if out.exists() and any(out.iterdir()):
            archive=ROOT/'interrupted_exports'/(out.name+'_'+str(time.time_ns()))
            if not out.resolve().is_relative_to(Path.cwd().resolve()) or not archive.resolve().is_relative_to(Path.cwd().resolve()):raise ValueError('Unsafe archive paths')
            archive.parent.mkdir(parents=True,exist_ok=True);out.rename(archive)
        stage(kind+'_export',['-m','src.finalize_improved','--work',str(work),'--selection',str(work/'selection.json'),'--output',str(out),'--unique-owner'])
    stage(kind+'_official_validation',['vendor/student_resource/utils/validate_submission.py','--matching',str(out/'matching_results.tsv'),'--test-dir','dataset/test','--check-ids'])
    print('READY',str(out/'matching_results.tsv'),flush=True)
    return {'output':str(out),'validation':json.loads(validation.read_text(encoding='utf-8')),'official_validator':'PASS','amazon_score':None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='parth':raise ValueError('Requires parth')
    if not (OUT/'confirmation_v6.json').exists():stage('confirmation_v6',['-m','src.campaign_confirm_v6'])
    frozen=verify_frozen();result=json.loads((OUT/'confirmation_v6.json').read_text(encoding='utf-8'))
    if result['frozen']!=frozen or not result['promotion_supported']:raise ValueError('No promoted ensemble')
    state={'state':'preparing','chosen':result['chosen'],'outputs':[]};path=OUT/'ensemble_production.json'
    def save():path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save();stage('test_records_build',['-m','src.campaign_test_records']);state['state']='neural_scoring';save()
    stage('neural_production',['-m','src.campaign_neural_test']);state['state']='neural_export';save();state['outputs'].append(export('neural'));save()
    if result['chosen']=='joint_stack_d3':
        state['state']='reverse_index';save();stage('reverse_test_build',['-m','src.campaign_reverse','index','--split','test'])
        state['state']='joint_scoring';save();stage('joint_production',['-m','src.campaign_joint_test','--workers',str(args.workers)]);state['state']='joint_export';save();state['outputs'].append(export('joint'))
    state['state']='ready_for_amazon_evaluation';save()


if __name__=='__main__':main()
