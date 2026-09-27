"""Resumable offline test scoring for the selected local context model."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd

from .context_experiments import context
from .detail_evidence import features as detail_features
from .disk_blocking import rules_text
from .disk_store import COLUMNS, connect, fetch_records
from .evidence import enrich
from .local_training_features import require_signature, save
from .peer_evidence import features as peer_features
from .preprocessing import preprocess
from .provenance import PairProvenance, frequent_keys
from .raw_evidence import features as raw_features
from .rescoring import batches, filehash, numeric_id

W = {}


def json_save(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def native_libraries(root):
    """Compile key generation only; leave the original index manifests untouched."""
    import ziglang
    compiler = Path(ziglang.__file__).parent / 'zig.exe'
    directory = root / 'native'
    directory.mkdir(exist_ok=True)
    source_dir = Path(__file__).parent / 'native'
    env = dict(os.environ, ZIG_GLOBAL_CACHE_DIR=str((root / 'zig_cache').resolve()),
               ZIG_LOCAL_CACHE_DIR=str((root / 'zig_local').resolve()))
    for name in ('index.cpp', 'supplement.cpp', 'provenance.cpp'):
        source = source_dir / name
        if name == 'provenance.cpp':
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            destination = directory / ('provenance-' + digest + '.dll')
        else:
            digest = hashlib.sha256((source.read_text(encoding='utf-8') +
                       (source_dir / 'index.cpp').read_text(encoding='utf-8')).encode()).hexdigest()[:16]
            destination = directory / ('libber-' + digest + '.dll')
        if not destination.exists():
            print('COMPILING', name, flush=True)
            subprocess.run([str(compiler), 'c++', '-target', 'x86_64-windows-gnu', '-O3',
                            '-std=c++17', '-shared', '-static', str(source.resolve()),
                            '-o', str(destination.resolve())], env=env, check=True)


def prepare_runtime(args):
    native_libraries(args.output)
    keys_path, rids_path = args.output / 'target_keys.npy', args.output / 'target_rids.npy'
    if not keys_path.exists() or not rids_path.exists():
        print('PREPARING_TARGET_LOOKUP', flush=True)
        con = connect(args.work / 'test/records.sqlite', True)
        try:
            n = con.execute('SELECT COUNT(*) FROM targets').fetchone()[0]
            keys, rids = np.empty(n, np.uint64), np.empty(n, np.uint32)
            for i, (rid, entity) in enumerate(con.execute('SELECT rid,entity_id FROM targets ORDER BY rid')):
                keys[i], rids[i] = numeric_id(entity), rid
            order = np.argsort(keys)
            keys, rids = keys[order], rids[order]
            if np.any(keys[1:] == keys[:-1]):
                raise ValueError('Ambiguous numeric target IDs')
            for path, array in ((keys_path, keys), (rids_path, rids)):
                temp = path.with_suffix('.partial')
                with temp.open('wb') as stream:
                    np.save(stream, array)
                temp.replace(path)
        finally:
            con.close()
    counts = args.output / 'posting_counts.joblib'
    if not counts.exists():
        # Posting exclusions are needed to reproduce the original blocking features.
        print('PREPARING_BLOCKING_FEATURE_LOOKUP', flush=True)
        save(counts, {name: frequent_keys(args.work / 'test' / filename, 240)
                      for name, filename in [('base', 'blocking_index.bin'), ('extra', 'supplement_index.bin')]})


def init_worker(work, output, model):
    work, output = Path(work), Path(output)
    W['con'] = connect(work / 'test/records.sqlite', True)
    W['model'] = joblib.load(model)
    if W['model']['artifact_type'] != 'local_context_reranker':
        raise ValueError('Expected local context reranker')
    W['engineer'] = joblib.load(work / 'feature_engineer.joblib')
    W['keys'] = np.load(output / 'target_keys.npy', mmap_mode='r')
    W['rids'] = np.load(output / 'target_rids.npy', mmap_mode='r')
    W['provenance'] = PairProvenance(output / 'native', joblib.load(output / 'posting_counts.joblib'), 240)
    for model_part in (W['model'], W['model']['base']):
        model_part['matcher'].estimator.set_params(thread_count=1)


def score_batch(batch):
    start, rows = batch
    con, model = W['con'], W['model']
    anchors = fetch_records(con, 'anchors', range(start+1, start+len(rows)+1))
    if anchors.entity_id.tolist() != [source for source, _ in rows]:
        raise ValueError('Candidate rows differ from test Source 1 order')
    for _, targets in rows:
        if len(set(targets)) != len(targets):
            raise ValueError('Duplicate candidate ID')
    ids = [target for _, targets in rows for target in targets]
    probabilities = np.empty(0)
    if ids:
        keys = np.fromiter((numeric_id(v) for v in ids), dtype=np.uint64, count=len(ids))
        positions = np.searchsorted(W['keys'], keys)
        if np.any(positions >= len(W['keys'])) or not np.array_equal(W['keys'][positions], keys):
            raise ValueError('Unknown candidate ID')
        rids = W['rids'][positions]
        repeat = np.repeat(np.arange(len(rows)), [len(targets) for _, targets in rows])
        targets = fetch_records(con, 'targets', rids)
        target_positions = np.searchsorted(targets.rid.to_numpy(), rids)
        if targets.entity_id.to_numpy()[target_positions].tolist() != ids:
            raise ValueError('Candidate ID representation differs from database')
        masks = W['provenance'].masks(anchors, targets, repeat, target_positions)
        if not np.all(masks):
            raise ValueError('Candidate blocking provenance mismatch')
        pairs = pd.DataFrame({'anchor_rid': anchors.rid.to_numpy()[repeat], 'target_rid': rids,
            'source1_entity_id': anchors.entity_id.to_numpy()[repeat], 'candidate_entity_id': ids,
            'blocking_rules': [rules_text(mask) for mask in masks]})
        records = preprocess(pd.concat([anchors[COLUMNS[:4]], targets[COLUMNS[:4]]], ignore_index=True))
        engineer = W['engineer']
        original = engineer.transform(pairs, engineer.prepare(records))
        base = model['base']
        stats = base['enrichment']['stats']
        features = pd.concat([original, enrich(pairs, con, stats), detail_features(pairs, con, stats),
                              raw_features(pairs, con)], axis=1)
        if features.columns.tolist() != base['feature_names']:
            raise ValueError('Base model feature schema mismatch')
        p = base['matcher'].estimator.predict_proba(features)[:, 1]
        features = pd.concat([features, context(pairs, p), peer_features(pairs, p, con)], axis=1)
        if features.columns.tolist() != model['feature_names']:
            raise ValueError('Reranker feature schema mismatch')
        probabilities = model['matcher'].predict(features)
    matching, candidates, offset, accepted_count = [], [], 0, 0
    for source, targets in rows:
        p = probabilities[offset:offset+len(targets)]
        offset += len(targets)
        accepted = sorted(((target, float(prob)) for target, prob in zip(targets, p)
                           if prob >= model['threshold']), key=lambda item: (-item[1], item[0]))
        matching.append(source + '\t' + ','.join(target for target, _ in accepted) + '\n')
        candidates.append(source + '\t' + ','.join(targets) + '\n')
        accepted_count += len(accepted)
    return {'start': start, 'anchors': len(rows), 'pairs': len(ids), 'matches': accepted_count,
            'matching': ''.join(matching), 'candidates': ''.join(candidates)}


def run(args):
    if args.workers < 1 or args.batch < 1:
        raise ValueError('Workers and batch size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'run.lock').open('a+b')
    import msvcrt
    lock.seek(0)
    lock.write(b'1'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    # Keep the lock open until process exit, including setup and validation.
    manifest = json.loads((args.work / 'test/store_manifest.json').read_text(encoding='utf-8'))
    total = manifest['counts']['source1']
    model = joblib.load(args.model)
    if model.get('artifact_type') != 'local_context_reranker':
        raise ValueError('Expected local context reranker')
    for name, digest in model['base']['enrichment']['signature']['code'].items():
        if filehash(Path(__file__).parent / name) != digest:
            raise ValueError(f'Training feature code changed: {name}')
    code = ['infer_local_context.py', 'features.py', 'preprocessing.py', 'normalization.py',
            'context_experiments.py', 'peer_evidence.py', 'provenance.py', 'native/index.cpp',
            'native/supplement.cpp', 'native/provenance.cpp']
    inputs = [args.work / 'test' / name for name in
              ('records.sqlite', 'blocking_index.bin', 'supplement_index.bin')]
    signature = {'model': filehash(args.model), 'candidates': filehash(args.candidates),
        'engineer': filehash(args.work / 'feature_engineer.joblib'), 'batch': args.batch,
        'inputs': {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in inputs},
        'code': {name: filehash(Path(__file__).parent / name) for name in code},
        'threshold': model['threshold'], 'expected_anchors': total}
    require_signature(args.output / 'signature.json', signature)
    del model
    if (args.output / 'complete.json').exists():
        print('ALREADY_COMPLETE', str(args.output), flush=True)
        return
    json_save(args.output / 'progress.json', {'state': 'setup', 'expected_anchors': total})
    prepare_runtime(args)
    shards = args.output / 'shards'
    shards.mkdir(exist_ok=True)
    started = time.time()
    progress = {'state': 'scoring', 'anchors': 0, 'pairs': 0, 'matches': 0, 'expected_anchors': total}
    pending = deque()
    matching = args.output / 'matching_results.tsv.partial'
    candidates = args.output / 'candidate_pairs.tsv.partial'
    with matching.open('w', encoding='utf-8', newline='') as mw, candidates.open('w', encoding='utf-8', newline='') as cw:
        mw.write('source1_entity_id\tmatched_entity_ids\n')
        cw.write('source1_entity_id\tcandidate_entity_ids\n')
        def consume():
            start, future = pending.popleft()
            path = shards / f'{start:08d}.joblib'
            result = joblib.load(path) if future is None else future.result()
            if result['start'] != progress['anchors']:
                raise ValueError('Inference shard order/coverage mismatch')
            if future is not None:
                save(path, result)
            mw.write(result['matching']); cw.write(result['candidates'])
            mw.flush(); cw.flush()
            for key in ('anchors', 'pairs', 'matches'):
                progress[key] += result[key]
            progress['seconds_this_run'] = time.time() - started
            json_save(args.output / 'progress.json', progress)
            print('INFERENCE', json.dumps(progress), flush=True)
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=init_worker, initargs=(str(args.work), str(args.output), str(args.model))) as pool:
            for batch in batches(args.candidates, args.batch, None):
                path = shards / f'{batch[0]:08d}.joblib'
                pending.append((batch[0], None if path.exists() else pool.submit(score_batch, batch)))
                if len(pending) >= args.workers * 2:
                    consume()
            while pending:
                consume()
    if progress['anchors'] != total:
        raise ValueError('Incomplete test coverage')
    matching.replace(args.output / 'matching_results.tsv')
    candidates.replace(args.output / 'candidate_pairs.tsv')
    progress['state'] = 'validating'
    json_save(args.output / 'progress.json', progress)
    command = [sys.executable, '-X', 'utf8', 'vendor/student_resource/utils/validate_submission.py',
        '--matching', str(args.output / 'matching_results.tsv'), '--candidate', str(args.output / 'candidate_pairs.tsv'),
        '--test-dir', str(args.test_dir), '--check-ids']
    with (args.output / 'official_validator.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    json_save(args.output / 'official_validator_command.json', {'command': command, 'exit_code': result.returncode})
    if result.returncode:
        raise RuntimeError('Official validation failed; see official_validator.log')
    progress['state'] = 'complete'
    json_save(args.output / 'progress.json', progress)
    json_save(args.output / 'complete.json', progress)
    print('INFERENCE_COMPLETE', json.dumps(progress), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('artifacts/local_training_context/model.joblib'))
    parser.add_argument('--work', type=Path, default=Path('work/real_v1'))
    parser.add_argument('--candidates', type=Path, default=Path('outputs/real_submission/candidate_pairs.tsv'))
    parser.add_argument('--output', type=Path, default=Path('outputs/local_context_submission'))
    parser.add_argument('--test-dir', type=Path, default=Path('student_resource/student_resource/dataset/test'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=250)
    run(parser.parse_args())
