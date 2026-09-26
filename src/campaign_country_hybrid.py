"""Country-transfer diagnostic: restore prior France predictions, keep new US/India."""
import argparse,collections,json,os,subprocess
from pathlib import Path
from .config import Config
from .disk_store import connect
from .rescoring import filehash


def merge_rows(anchors,baseline,current,candidates,destination):
    expected='source1_entity_id\tmatched_entity_ids'
    if next(baseline).rstrip('\r\n')!=expected or next(current).rstrip('\r\n')!=expected:
        raise ValueError('Invalid matching header')
    if next(candidates).rstrip('\r\n')!='source1_entity_id\tcandidate_entity_ids':
        raise ValueError('Invalid candidate header')
    destination.write(expected+'\n');seen=set();stats=collections.defaultdict(collections.Counter)
    for source,country in anchors:
        if country not in {'US','India','France'}:raise ValueError('Unexpected country')
        old=next(baseline).rstrip('\r\n').split('\t');new=next(current).rstrip('\r\n').split('\t');pool=next(candidates).rstrip('\r\n').split('\t')
        if any(len(row)!=2 or row[0]!=source for row in (old,new,pool)):raise ValueError('Source alignment')
        chosen=old if country=='France' else new
        matches=chosen[1].split(',') if chosen[1] else [];available=pool[1].split(',') if pool[1] else []
        picked=set(matches)
        if len(available)!=len(set(available)) or len(matches)!=len(picked) or not picked<=set(available):raise ValueError('Candidate/subset integrity')
        if not seen.isdisjoint(picked):raise ValueError('Cross-country duplicate target ownership')
        seen.update(picked);previous=set(new[1].split(',')) if new[1] else set()
        destination.write(source+'\t'+chosen[1]+'\n')
        for c in (stats[country],stats['ALL']):
            c['anchors']+=1;c['matches']+=len(matches);c['pairs']+=len(available);c['empty']+=not matches;c['changed_anchors_vs_main']+=picked!=previous;c['added_vs_main']+=len(picked-previous);c['removed_vs_main']+=len(previous-picked)
    if any(next(f,None) is not None for f in (baseline,current,candidates)):raise ValueError('Extra input rows')
    return dict(stats)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',default='F:/dotinit-Amazon-ML-Challenge/outputs/improved_unique_owner');parser.add_argument('--current',default='outputs/campaign_0931_oof_unique');parser.add_argument('--output',default='outputs/campaign_0942_france_baseline');args=parser.parse_args()
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='parth':raise ValueError('Run only on parth')
    baseline=Path(args.baseline);current=Path(args.current);out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Refusing to overwrite an output')
    inputs={}
    for name,root in [('baseline',baseline),('current',current)]:
        v=json.loads((root/'validation.json').read_text(encoding='utf-8'));actual=filehash(root/'matching_results.tsv')
        if actual!=v['matching_sha256'] or v['duplicate_target_ids']!=0:raise ValueError('Input is not the verified unique-owner export')
        inputs[name]={'path':str(root),'matching_sha256':actual,'anchors':v['anchors'],'candidate_sha256':v['candidate_sha256']}
    candidate=current/'candidate_pairs.tsv';candidate_sha=filehash(candidate)
    if any(v['candidate_sha256']!=candidate_sha for v in inputs.values()):raise ValueError('Input candidate pools differ')
    con=connect(Path(Config.load('config/windows.json').working_dir)/'test/records.sqlite',True);out.mkdir(parents=True,exist_ok=True);partial=out/'matching_results.partial'
    with (baseline/'matching_results.tsv').open(encoding='utf-8') as old,(current/'matching_results.tsv').open(encoding='utf-8') as new,candidate.open(encoding='utf-8') as pool,partial.open('w',encoding='utf-8',newline='') as dest:
        stats=merge_rows(con.execute('select entity_id,country from anchors order by rid'),old,new,pool,dest)
    con.close()
    if any(v['anchors']!=stats['ALL']['anchors'] for v in inputs.values()):raise ValueError('Anchor coverage differs')
    partial.replace(out/'matching_results.tsv');os.link(candidate.resolve(),out/'candidate_pairs.tsv')
    report={'state':'streaming_validated_official_validation_pending','experimental':True,'amazon_score':None,'policy':'Previous 0.931 France predictions; current 0.942 main US/India predictions unchanged','inputs':inputs,'by_country':stats,'duplicate_target_ids':0,'candidate_sha256':candidate_sha,'matching_sha256':filehash(out/'matching_results.tsv'),'streaming_validation':'PASS: complete source alignment, all rows, valid candidate subset, per-row and global target uniqueness'}
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
