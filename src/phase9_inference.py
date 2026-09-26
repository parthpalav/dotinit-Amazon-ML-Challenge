"""Checkpointed full-test inference for the locked original-pool Phase 9 model.

Reuses the hash-verified original candidate export. Global target arbitration is
performed in a separate pass before per-anchor/source margin gating and export.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .disk_store import COLUMNS, connect, fetch_records, NativeIndex
from .disk_blocking import rules_text
from .enhanced_features import compute_enhanced_features
from .veto_features import compute_veto_features
from .preprocessing import preprocess
from .provenance import PairProvenance, frequent_keys
from .rescoring import batches, numeric_id, filehash
from .phase9_true_control import fingerprint

ROOT = Path('work/phase9_inference')
REPORT = Path('reports/phase9_inference')
CANDIDATES = Path('outputs/real_submission/candidate_pairs.tsv')
MANIFEST = Path('config/phase9_production_lock.json')
ORIGINAL_LIBRARY = Path('work/real_v1/native/libber-b6e5285e72862775.dylib')
DTYPE = np.dtype([('anchor', '<u4'), ('target', '<u4'), ('p', '<f8')])
W = {}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class OriginalKeys:
    """Use the preserved pre-rebuild library; never regenerate indexes."""
    keys = NativeIndex.keys

    def __init__(self, supplemental):
        self.width = 64 if supplemental else 24
        self.lib = ctypes.CDLL(str(ORIGINAL_LIBRARY.resolve()))
        self.key_function = self.lib.ber_extra_keys if supplemental else self.lib.ber_keys
        chars = ctypes.POINTER(ctypes.c_char_p)
        self.key_function.argtypes = [chars, chars, chars, ctypes.c_size_t, ctypes.c_void_p]


def provenance(counts):
    value = PairProvenance(ROOT / 'native', counts, 240)
    value.base = OriginalKeys(False)
    value.extra = OriginalKeys(True)
    return value


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    locked = json.loads(MANIFEST.read_text())
    assert locked['candidate_pool'] == 'work/real_v1'
    assert (locked['max_candidates_per_anchor'], locked['retrieval_posting_limit']) == (32, 240)
    assert not locked['phonetic_relaxed_address_channels']
    assert locked['threshold'] == 0.5500000000000002 and locked['margin_delta'] == .2
    assert locked['target_arbitration']['enabled'] and locked['target_arbitration']['max_per_source'] is None
    for item in [locked['model'], *locked['pinned_dependencies']]:
        assert Path(item['path']).stat().st_size == item['bytes']
        assert filehash(item['path']) == item['sha256'], item['path']
    model = joblib.load(locked['model']['path'])
    assert model.estimator.feature_names_ == locked['features'] and model.calibrator is not None
    baseline = json.loads(Path('reports/improvements/baseline_manifest.json').read_text())
    candidate_hash = filehash(CANDIDATES)
    assert candidate_hash == baseline['files'][str(CANDIDATES)]['sha256']
    store = json.loads(Path('work/real_v1/test/store_manifest.json').read_text())
    for path, expected in store['inputs'].items():
        assert [Path(path).stat().st_size, Path(path).stat().st_mtime_ns] == expected
    assert store['counts']['source1'] == 1732544
    assert ORIGINAL_LIBRARY.exists()
    code = ['phase9_inference.py', 'veto_features.py', 'enhanced_features.py', 'features.py',
            'normalization.py', 'preprocessing.py', 'contacts.py', 'provenance.py', 'native/provenance.cpp']
    signature = dict(manifest=fingerprint(MANIFEST), model=locked['model'],
        candidates=fingerprint(CANDIDATES), original_library=fingerprint(ORIGINAL_LIBRARY),
        input_sources=[fingerprint(p) for p in store['inputs']],
        code={name: filehash(Path('src') / name) for name in code}, counts=store['counts'])
    sigpath = ROOT / 'signature.json'
    if sigpath.exists():
        assert json.loads(sigpath.read_text()) == signature, 'Changed inputs/code; use a new run directory'
    else:
        save_json(sigpath, signature)
    save_json(REPORT / 'artifact_verification.json', signature)
    count_path = ROOT / 'posting_counts.joblib'
    if not count_path.exists():
        counts = {key: frequent_keys(Path('work/real_v1/test') / filename, 240)
                  for key, filename in [('base', 'blocking_index.bin'), ('extra', 'supplement_index.bin')]}
        joblib.dump(counts, count_path)
    else:
        counts = joblib.load(count_path)
    for key, filename in [('base', 'blocking_index.bin'), ('extra', 'supplement_index.bin')]:
        assert filehash(Path('work/real_v1/test') / filename) == counts[key]['sha256']
    if not (ROOT / 'target_keys.npy').exists():
        con = connect('work/real_v1/test/records.sqlite', True)
        n = sum(store['counts'][s] for s in ['source2', 'source3'])
        keys = np.empty(n, np.uint64)
        rids = np.empty(n, np.uint32)
        for i, (rid, entity) in enumerate(con.execute('SELECT rid,entity_id FROM targets ORDER BY rid')):
            keys[i] = numeric_id(entity)
            rids[i] = rid
        assert i + 1 == n
        order = np.argsort(keys)
        assert np.all(np.diff(keys[order]) > 0)
        np.save(ROOT / 'target_rids.npy', rids[order])
        np.save(ROOT / 'target_keys.npy', keys[order])
        con.close()
    print('ARTIFACTS VERIFIED; original candidates and original key library pinned', flush=True)
    return locked, store


def init_worker():
    W['thread_limit'] = threadpool_limits(limits=1)
    W['locked'] = json.loads(MANIFEST.read_text())
    W['con'] = connect('work/real_v1/test/records.sqlite', True)
    W['model'] = joblib.load(W['locked']['model']['path'])
    W['engineer'] = joblib.load(W['locked']['feature_engineer'])
    W['keys'] = np.load(ROOT / 'target_keys.npy', mmap_mode='r')
    W['rids'] = np.load(ROOT / 'target_rids.npy', mmap_mode='r')
    W['provenance'] = provenance(joblib.load(ROOT / 'posting_counts.joblib'))


def feature_matrix(pairs, anchors, targets, engineer, con, names):
    records = preprocess(pd.concat([anchors[COLUMNS[:4]], targets[COLUMNS[:4]]], ignore_index=True))
    base = engineer.transform(pairs, engineer.prepare(records))
    maps = tuple(frame.set_index('entity_id').to_dict('index') for frame in [anchors, targets])
    veto = compute_veto_features(pairs, con, record_maps=maps)
    enhanced = compute_enhanced_features(pairs, con, record_maps=maps)
    return pd.concat([base.reset_index(drop=True), veto.reset_index(drop=True),
                      enhanced.reset_index(drop=True)], axis=1)[names]


def score_batch(batch):
    start, rows = batch
    anc = fetch_records(W['con'], 'anchors', range(start + 1, start + len(rows) + 1))
    assert anc.entity_id.tolist() == [row[0] for row in rows]
    ids = [target for _, ts in rows for target in ts]
    counts = np.array([len(ts) for _, ts in rows])
    assert all(len(ts) == len(set(ts)) <= 32 for _, ts in rows)
    repeat = np.repeat(np.arange(len(rows)), counts)
    keys = np.fromiter(map(numeric_id, ids), dtype=np.uint64, count=len(ids))
    positions = np.searchsorted(W['keys'], keys)
    assert np.all(positions < len(W['keys'])) and np.array_equal(W['keys'][positions], keys)
    rids = W['rids'][positions]
    targets = fetch_records(W['con'], 'targets', rids)
    masks = W['provenance'].masks(anc, targets, repeat, np.searchsorted(targets.rid.to_numpy(), rids))
    assert np.all(masks), 'Candidate not retrievable under original keys/posting limit'
    pairs = pd.DataFrame(dict(anchor_rid=anc.rid.to_numpy()[repeat], target_rid=rids,
        source1_entity_id=anc.entity_id.to_numpy()[repeat], candidate_entity_id=ids,
        blocking_rules=[rules_text(m) for m in masks]))
    X = feature_matrix(pairs, anc, targets, W['engineer'], W['con'], W['locked']['features'])
    result = np.empty(len(pairs), DTYPE)
    result['anchor'] = start + repeat + 1
    result['target'] = rids
    result['p'] = W['model'].predict(X)
    return start, len(rows), result


def smoke():
    init_worker()
    data = joblib.load('work/real_v1/features/validation.joblib')
    ix = np.unique(np.r_[np.arange(128), np.random.default_rng(42).choice(len(data['pairs']), 1024, replace=False)])
    pairs = data['pairs'].iloc[ix]
    con = connect('work/real_v1/train/records.sqlite', True)
    anchors = fetch_records(con, 'anchors', pairs.anchor_rid.unique())
    targets = fetch_records(con, 'targets', pairs.target_rid.unique())
    actual = feature_matrix(pairs, anchors, targets, W['engineer'], con, W['locked']['features'])
    expected = pd.concat([data['features'].iloc[ix].reset_index(drop=True),
        joblib.load('work/improvements/veto_cache/validation_veto.joblib').iloc[ix].reset_index(drop=True),
        joblib.load('work/improvements/enhanced_cache/validation_enhanced.joblib').iloc[ix].reset_index(drop=True)], axis=1)[W['locked']['features']]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(W['model'].predict(actual), W['model'].predict(expected))
    # Compare posting-aware replay with actual original-index lookup on test pairs.
    batch = next(batches(CANDIDATES, 100, 100))
    start = time.time()
    _, n, scores = score_batch(batch)
    anchor_frame = fetch_records(W['con'], 'anchors', range(1, n + 1))
    target_frame = fetch_records(W['con'], 'targets', scores['target'])
    ai = scores['anchor'] - 1
    ti = np.searchsorted(target_frame.rid.to_numpy(), scores['target'])
    replay = W['provenance'].masks(anchor_frame, target_frame, ai, ti)
    native_mask = np.zeros(len(scores), np.uint16)
    for extra, filename in [(False, 'blocking_index.bin'), (True, 'supplement_index.bin')]:
        # The preserved macOS library uses ber_open, predating the ber_attach ABI.
        # Its handles must never be mixed with the current library's handles.
        native = NativeIndex.__new__(NativeIndex)
        original = OriginalKeys(extra)
        native.lib = original.lib
        native.width = original.width
        native.mapping = None
        native.lib.ber_open.argtypes = [ctypes.c_char_p]
        native.lib.ber_open.restype = ctypes.c_void_p
        native.lib.ber_close.argtypes = [ctypes.c_void_p]
        native.handle = native.lib.ber_open(str(Path('work/real_v1/test') / filename).encode())
        assert native.handle
        native.key_function = original.key_function
        original_lookup = original.lib.ber_extra_lookup if extra else original.lib.ber_lookup
        original_lookup.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        original_lookup.restype = ctypes.c_size_t
        native.lookup_function = original_lookup
        rr, mm, oo = native.lookup(native.keys(anchor_frame), 240)
        for i in range(n):
            row = np.flatnonzero(ai == i)
            raw = rr[oo[i]:oo[i+1]]
            positions = np.searchsorted(raw, scores['target'][row])
            valid = positions < len(raw)
            valid[valid] &= raw[positions[valid]] == scores['target'][row][valid]
            hit = np.flatnonzero(valid)
            native_mask[row[hit]] |= mm[oo[i] + positions[hit]]
        native.close()
    np.testing.assert_array_equal(replay, native_mask)
    save_json(REPORT / 'smoke.json', dict(feature_rows=len(ix), exact_81_feature_parity=True,
        exact_probability_parity=True, exact_original_test_provenance_parity=True,
        test_anchors=n, test_pairs=len(scores), seconds=time.time()-start))
    W['con'].close()
    W.clear()
    print('SMOKE PASSED: exact feature/probability and original-index provenance parity', flush=True)


def score(workers, size):
    root = ROOT / f'scores_{size}'
    root.mkdir(exist_ok=True)
    done = 0
    total_pairs = 0
    started = time.time()
    pending = deque()
    def consume():
        nonlocal done
        start, n, result = pending.popleft().result()
        path = root / f'{start:08d}.npy'
        with path.with_suffix('.partial').open('wb') as f:
            np.save(f, result)
        path.with_suffix('.partial').replace(path)
        done += n
        status = dict(anchors_completed=start+n, total_anchors=1732544,
            anchors_scored_this_run=done, elapsed_seconds=time.time()-started)
        save_json(REPORT / 'progress.json', status)
        if done % (size*10) == 0:
            print('SCORED', json.dumps(status), flush=True)
    total = 0
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker) as pool:
        for batch in batches(CANDIDATES, size, None):
            total += len(batch[1])
            n_pairs = sum(len(ts) for _, ts in batch[1])
            total_pairs += n_pairs
            path = root / f'{batch[0]:08d}.npy'
            if path.exists():
                x = np.load(path, mmap_mode='r')
                assert x.dtype == DTYPE and len(x) == n_pairs and np.isfinite(x['p']).all()
                continue
            pending.append(pool.submit(score_batch, batch))
            if len(pending) >= workers*2:
                consume()
        while pending:
            consume()
    assert total == 1732544 and total_pairs == 55431940
    save_json(REPORT / 'scoring_complete.json', dict(anchors=total, pairs=total_pairs,
        seconds_this_run=time.time()-started, workers=workers, batch=size))
    return root


def update_owners(best, owner, x, offset):
    order = np.lexsort((np.arange(len(x)), -x['p'], x['target']))
    first = order[np.r_[True, np.diff(x['target'][order]) != 0]]
    chosen = first[x['p'][first] > best[x['target'][first]]]
    best[x['target'][chosen]] = x['p'][chosen]
    owner[x['target'][chosen]] = offset + chosen


def select_rows(x, owner, offset, start, nrows, source2_count, delta, threshold):
    winner = owner[x['target']] == offset + np.arange(len(x))
    arb = np.where(winner, x['p'], 0.)
    group = (x['anchor'].astype(np.int64) - start - 1)*2 + (x['target'] > source2_count)
    maximum = np.zeros(nrows*2)
    np.maximum.at(maximum, group, arb)
    return (maximum[group]-arb <= delta) & (arb >= threshold)


def export(root, size, output, locked, store):
    assert not output.exists(), 'Use a fresh output directory; preserve previous artifacts'
    output.mkdir(parents=True)
    n_targets = store['counts']['source2'] + store['counts']['source3']
    best = np.full(n_targets + 1, -np.inf)
    owner = np.full(n_targets + 1, np.iinfo(np.uint64).max, np.uint64)
    offset = 0
    # In each shard, first maximal-scoring row wins; earlier shards win exact ties.
    for path in sorted(root.glob('*.npy')):
        x = np.load(path)
        update_owners(best, owner, x, offset)
        offset += len(x)
    assert offset == 55431940
    np.save(ROOT / 'global_winner_rows.npy', owner)
    stats = dict(anchors=0, scored_pairs=0, matched_pairs=0, singletons=0,
                 model=locked['model'], threshold=locked['threshold'], margin_delta=.2,
                 global_target_uniqueness=True, candidate_sha256=filehash(CANDIDATES))
    seen = np.zeros(n_targets + 1, bool)
    offset = 0
    with (output / 'matching_results.tsv.partial').open('w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for start, rows in batches(CANDIDATES, size, None):
            x = np.load(root / f'{start:08d}.npy')
            counts = np.array([len(ts) for _, ts in rows])
            repeat = np.repeat(np.arange(len(rows)), counts)
            assert np.array_equal(x['anchor'], start + repeat + 1)
            selected = select_rows(x, owner, offset, start, len(rows), store['counts']['source2'],
                                   locked['margin_delta'], locked['threshold'])
            assert not seen[x['target'][selected]].any()
            seen[x['target'][selected]] = True
            starts = np.r_[0, np.cumsum(counts)]
            for i, (anchor, ids) in enumerate(rows):
                lo, hi = starts[i:i+2]
                positions = np.flatnonzero(selected[lo:hi])
                chosen = sorted(((ids[j], x['p'][lo+j]) for j in positions), key=lambda v: (-v[1], v[0]))
                f.write(anchor + '\t' + ','.join(t for t, _ in chosen) + '\n')
                stats['singletons'] += int(not chosen)
            stats['anchors'] += len(rows)
            stats['scored_pairs'] += len(x)
            stats['matched_pairs'] += int(selected.sum())
            offset += len(x)
    assert stats['anchors'] == 1732544 and stats['scored_pairs'] == 55431940
    (output / 'matching_results.tsv.partial').rename(output / 'matching_results.tsv')
    shutil.copy2(CANDIDATES, output / 'candidate_pairs.tsv')
    assert filehash(output / 'candidate_pairs.tsv') == stats['candidate_sha256']
    stats['matching_sha256'] = filehash(output / 'matching_results.tsv')
    save_json(REPORT / 'export.json', stats)
    print('EXPORTED', json.dumps(stats), flush=True)


def validate(output):
    command = [sys.executable, 'vendor/student_resource/utils/validate_submission.py',
        '--matching', str(output / 'matching_results.tsv'), '--candidate', str(output / 'candidate_pairs.tsv'),
        '--test-dir', '../student_resource/dataset/test', '--check-ids']
    with (REPORT / 'official_validator.log').open('w') as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    save_json(REPORT / 'official_validator_command.json', dict(command=command, exit_code=result.returncode))
    print((REPORT / 'official_validator.log').read_text(), flush=True)
    if result.returncode:
        raise RuntimeError('Official validator failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch', type=int, default=500)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    locked, store = prepare()
    smoke()
    if args.smoke_only:
        return
    scores = score(args.workers, args.batch)
    output = Path('outputs/phase9_submission')
    export(scores, args.batch, output, locked, store)
    validate(output)
    save_json(REPORT / 'completion.json', dict(status='complete', output=str(output),
        full_test_anchors=1732544, official_validator_exit_code=0))


if __name__ == '__main__':
    main()
