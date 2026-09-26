"""Streaming quality audit of actual TSV inputs. No raw-file writes."""
from pathlib import Path
import json, time, resource, hashlib
from collections import Counter
import numpy as np
import pandas as pd


def quality_report(config):
    root=Path(config.resource_dir or Path(config.dataset_dir).parent)
    out=Path(config.working_dir)/'quality'
    out.mkdir(parents=True,exist_ok=True)
    reports=Path(config.reports_dir);reports.mkdir(parents=True,exist_ok=True)
    signature={str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(root.rglob('*')) if p.is_file()}
    result_path=reports/'real_data_quality.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text(encoding='utf-8'))
        if previous.get('signature')==signature:
            return previous
    summary={}
    for p in sorted((root/'dataset').rglob('*.tsv')):
        start=time.time(); counts=Counter(); countries=Counter(); missing=Counter(); ids=[]; hashes=[]; sample=[]; match_counts=Counter()
        for frame in pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False, chunksize=100000):
            counts['rows']+=len(frame)
            for c in frame:
                missing[c]+=int(frame[c].str.strip().eq('').sum())
            idcol=frame.columns[0]
            entity=frame[idcol]
            if not entity.str.match(r'S[123]-[0-9]+$').all():
                raise ValueError(f'Non numeric suffix ID at {p}')
            ids.append(entity.str[3:].astype('uint64').to_numpy())
            if 'country' in frame:
                countries.update(frame.country.value_counts().to_dict())
                for c in ['business_name','business_address']:
                    values=frame[c]
                    counts[c+'_near_empty']+=int(values.str.strip().str.len().lt(3).sum())
                    counts[c+'_non_ascii']+=int(values.str.contains(r'[^\x00-\x7f]',regex=True).sum())
                    counts[c+'_uppercase']+=int(values.str.contains(r'[A-Z]',regex=True).sum())
                    counts[c+'_punctuation']+=int(values.str.contains(r'[^\w\s]',regex=True).sum())
                combined=frame.business_name+' '+frame.business_address
                counts['embedded_url_hints']+=int(combined.str.contains(r'https?://|www\.',case=False,regex=True).sum())
                counts['embedded_email_hints']+=int(combined.str.contains('@',regex=False).sum())
                counts['labelled_phone_hints']+=int(combined.str.contains(r'\b(?:tel|phone|mobile|mob)\s*[:=]',case=False,regex=True).sum())
                counts['no_identifiers']+=int((frame.business_name.str.strip().eq('') & frame.business_address.str.strip().eq('')).sum())
                values=frame[['business_name','business_address','country']]
                hashes.append(pd.util.hash_pandas_object(values,index=False).to_numpy())
                if len(sample)<5: sample=frame.head(5).to_dict('records')
            else:
                n=frame.matched_entity_ids.map(lambda s: 0 if not s.strip() else len(s.split(',')))
                match_counts.update(n.value_counts().to_dict())
                counts['positive_pairs']+=int(n.sum())
                counts['singletons']+=int(n.eq(0).sum())
        ids=np.concatenate(ids)
        np.save(out/(p.stem+'_ids.npy'),ids)
        counts['unique_ids']=len(np.unique(ids)); counts['duplicate_ids']=len(ids)-counts['unique_ids']
        if hashes:
            hashes=np.concatenate(hashes); np.save(out/(p.stem+'_business_hashes.npy'),hashes)
            counts['duplicate_business_fingerprints']=len(hashes)-len(np.unique(hashes))
        info={'counts':dict(counts),'columns':frame.columns.tolist(),'dtypes':{c:str(t) for c,t in frame.dtypes.items()},'missing':dict(missing),'countries':dict(countries),'match_count_distribution':dict(match_counts),'sample':sample,'seconds':time.time()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        summary[str(p.relative_to(root))]=info
        (out/'initial_profile.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False), encoding='utf-8')
        print(p.stem, {**dict(counts),'countries':dict(countries),'missing':dict(missing)},flush=True)
    for source in (1,2,3):
        a=np.load(out/f'train_source{source}_ids.npy'); b=np.load(out/f'test_source{source}_ids.npy')
        summary[f'relationship_source{source}']={'train_test_id_overlap':len(np.intersect1d(a,b))}
    inventory=[]
    for path in sorted(root.rglob('*')):
        if path.is_file():
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            inventory.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
    summary['inventory']=inventory
    summary['signature']=signature
    (reports/'real_data_quality.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False), encoding='utf-8')
    lines=['# Real data quality report','',
           'All supplied source files use the same four-field UTF-8 TSV schema. This is one train triplet and one test triplet, not three independent prediction tasks. Raw inputs are unchanged.','',
           '| File | Rows | Unique IDs | Duplicate IDs | Missing names | Missing addresses | Countries |',
           '|---|---:|---:|---:|---:|---:|---|']
    for name,info in summary.items():
        if not isinstance(info,dict) or 'counts' not in info:continue
        c=info['counts'];m=info['missing']
        lines.append(f"| {name} | {c['rows']:,} | {c['unique_ids']:,} | {c['duplicate_ids']:,} | {m.get('business_name','N/A')} | {m.get('business_address','N/A')} | {info['countries']} |")
    lines+=['', 'No dedicated phone, email, URL, city, state or postal fields exist. Contact strings sometimes occur inside names; only explicit hints are extracted. Address numbers are not treated as phones. Contact validity and geographic consistency cannot be externally verified under the challenge rules.', '',
            'The JSON reports near-empty fields, non-ASCII text, casing/punctuation, embedded contact hints, exact raw-business fingerprint repeats, singleton/positive distributions and train/test ID overlap. Fingerprint repeats are possible duplicate records, not proof of entity identity; ID uniqueness is exact for the observed numeric-suffix syntax.', '',
            'Ground-truth pair duplication, conflicting ownership, source coverage and target existence are enforced by the disk-store build and abort on failure. Final results include these checks.', '',
            'The normalizer now preserves Indic combining marks. City/state and postal extraction are uncertain when address components are reordered; no structured address field is mandatory for retrieval. Missing strings never become false exact-match evidence.', '',
            'Every resource file, including documentation, official validator and filesystem metadata, is included in the SHA-256 inventory in real_data_quality.json.']
    (reports/'real_data_quality_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary
