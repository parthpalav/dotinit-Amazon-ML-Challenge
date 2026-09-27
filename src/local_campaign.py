"""Local CPU improvement campaign with frozen confirmation and resumable inference.

Uses existing stores and fitting caches. It does not call prepare, quality, audit,
or the old all-in-one run command, and never overwrites a submitted TSV/model.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gc
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import Config
from .context_experiments import context
from .disk_blocking import DiskBlocker, rules_text
from .disk_store import COLUMNS, connect, fetch_records
from .evidence import enrich, make_stats
from .experiments import choose, evaluate
from .local_evidence import LocalTransliterator, features as detail_features
from .model import CalibratedMatcher
from .peer_evidence import features as peer_features
from .preprocessing import preprocess
from .rescoring import DTYPE, batches, filehash, lookup_cache, numeric_id, sourcehash

ROOT = Path('work/local_098_v1')
OUT = Path('reports/local_098_v1')
ART = Path('artifacts/local_098_v1')
FEATURES = ROOT / 'features'
W = {}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    joblib.dump(value, temp)
    temp.replace(path)


def save_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('wb') as stream:
        np.save(stream, value)
    temp.replace(path)


def status(stage, **values):
    row = {'stage': stage, 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'), **values}
    json_write(OUT / 'status.json', row)
    print(json.dumps(row), flush=True)


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Inputs or recipe changed: preserve {path} and use a new campaign directory')
    else:
        json_write(path, value)


def groups(pairs, size=250):
    """Never split an anchor's candidates across feature batches."""
    a = pairs.anchor_rid.to_numpy()
    if np.any(a[1:] < a[:-1]):
        raise ValueError('Pairs must be sorted by anchor')
    starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1, len(a)]
    for i in range(0, len(starts) - 1, size):
        yield int(starts[i]), int(starts[min(i + size, len(starts) - 1)])


def pair_signature(pairs):
    import hashlib
    return hashlib.sha256(pairs[['anchor_rid', 'target_rid']].to_numpy(dtype='<u4').tobytes()).hexdigest()


def read_part(cfg, split):
    return joblib.load(Path(cfg.working_dir) / 'features' / (split + '.joblib'))


def training_signature(cfg, args):
    files = [Path(cfg.working_dir) / 'features' / (s + '.joblib') for s in ('fit', 'calibration', 'validation')]
    files += [Path(cfg.working_dir) / 'entity_selection.npz', Path(cfg.working_dir) / 'feature_engineer.joblib',
              Path(cfg.working_dir) / 'train/store_manifest.json', Path(cfg.working_dir) / 'candidate_ranker.joblib']
    code = ['local_campaign.py', 'local_evidence.py', 'context_experiments.py', 'peer_evidence.py',
            'evidence.py', 'model.py', 'features.py', 'normalization.py', 'preprocessing.py',
            'contacts.py', 'disk_blocking.py', 'coarse_ranker.py', 'provenance.py', 'native/provenance.cpp']
    return {'inputs': {str(p): filehash(p) for p in files},
            'code': {s: sourcehash(Path('src') / s) for s in code},
            'config': cfg.to_dict(), 'iterations': args.iterations,
            'base_depths': args.base_depths, 'meta_depths': args.meta_depths,
            'threads': args.threads, 'floor': args.floor, 'target': .98,
            'confirmation_anchors': args.confirmation_anchors,
            'xgboost': __import__('xgboost').__version__}


def prepare_split(cfg, split, part):
    path = FEATURES / (split + '.joblib')
    pairs = part['pairs']
    signature = pair_signature(pairs)
    if path.exists():
        saved = joblib.load(path)
        if saved['pairs'] != signature:
            raise ValueError('Feature alignment changed')
        return saved['features']
    con = connect(Path(cfg.working_dir) / 'train/records.sqlite', True)
    stats = make_stats(cfg, 'train')
    transliterator = LocalTransliterator()
    blocks = []
    for lo, hi in groups(pairs):
        block = FEATURES / (split + '_parts') / f'{lo:08d}.joblib'
        if block.exists():
            extra = joblib.load(block)
        else:
            p = pairs.iloc[lo:hi]
            extra = pd.concat([enrich(p, con, stats), detail_features(p, con, transliterator)], axis=1)
            dump(block, extra)
        if not extra.index.equals(pairs.iloc[lo:hi].index):
            raise ValueError('Partial feature alignment changed')
        blocks.append(extra)
        status('preparing_features', split=split, pairs_done=hi, pairs_total=len(pairs))
    X = pd.concat([part['features'], pd.concat(blocks)], axis=1).astype(np.float32)
    if X.columns.duplicated().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError('Invalid feature schema or values')
    dump(path, {'pairs': signature, 'features': X})
    con.close()
    transliterator.close()
    return X


def calibration_masks(part):
    anchors = np.asarray(sorted(part['truth_counts']), dtype=object)
    rng = np.random.default_rng(20260930)
    rng.shuffle(anchors)
    held = set(anchors[:2000])
    calibrate = part['pairs'].source1_entity_id.isin(held).to_numpy()
    return ~calibrate, calibrate


def ownership_probabilities(pairs, probabilities):
    """Keep only a target's unique highest-scoring owner; abstain on ties."""
    frame = pd.DataFrame({'target': pairs.candidate_entity_id.to_numpy(), 'p': probabilities})
    maximum = frame.groupby('target', sort=False).p.transform('max')
    winner = frame.p == maximum
    ties = winner.groupby(frame.target, sort=False).transform('sum')
    return np.where(winner & (ties == 1), probabilities, 0.)


def estimator(depth, trees, threads, seed=42, stopping=False):
    return XGBClassifier(n_estimators=trees, max_depth=depth, learning_rate=.045,
                         min_child_weight=8, subsample=.9, colsample_bytree=.9,
                         reg_lambda=10., reg_alpha=.05, objective='binary:logistic',
                         eval_metric='logloss', tree_method='hist', max_bin=256,
                         n_jobs=threads, random_state=seed,
                         early_stopping_rounds=100 if stopping else None)


def train_base(cfg, args, data, X):
    early, calibration = calibration_masks(data['calibration'])
    rows = []
    for depth in args.base_depths:
        name = f'xgb_base_d{depth}'
        path, report = ART / (name + '.joblib'), OUT / (name + '.json')
        if path.exists() and report.exists():
            rows.append(json.loads(report.read_text()))
            continue
        status('training_base', model=name, device='CPU', maximum_trees=args.iterations)
        model = estimator(depth, args.iterations, args.threads, stopping=True)
        model.fit(X['fit'], data['fit']['pairs'].label,
                  eval_set=[(X['calibration'][early], data['calibration']['pairs'].label[early])], verbose=100)
        matcher = CalibratedMatcher(model)
        matcher.calibrate(X['calibration'][calibration], data['calibration']['pairs'].label[calibration])
        prob = matcher.predict(X['validation'])
        deployed = ownership_probabilities(data['validation']['pairs'], prob)
        threshold, table = choose(data['validation'], deployed)
        row = {'name': name, 'kind': 'base', 'depth': depth, 'trees': model.best_iteration + 1,
               'threshold': threshold, 'path': str(path), **evaluate(data['validation'], deployed, threshold)}
        dump(path, {'matcher': matcher, 'feature_names': X['fit'].columns.tolist(), 'threshold': threshold})
        save_array(ROOT / (name + '_validation.npy'), prob)
        json_write(report, row)
        table.to_csv(OUT / (name + '_thresholds.tsv'), sep='\t', index=False)
        rows.append(row)
        status('base_selection_result', **row)
    best = max(rows, key=lambda r: r['f0.5'])
    frozen_json(OUT / 'base_selection.json', best)
    return best


def cross_fit(args, data, X, best):
    pairs = data['fit']['pairs']
    ids = np.asarray(sorted(data['fit']['truth_counts']), dtype=object)
    np.random.default_rng(20260930).shuffle(ids)
    probability = np.empty(len(pairs), dtype=np.float64)
    coverage = np.zeros(len(pairs), dtype=np.uint8)
    for k, excluded in enumerate(np.array_split(ids, 3)):
        mask = pairs.source1_entity_id.isin(excluded).to_numpy()
        path = ROOT / f'oof_fold{k}.joblib'
        if path.exists():
            saved = joblib.load(path)
            if not np.array_equal(saved['rows'], np.flatnonzero(mask)):
                raise ValueError('Cross-fitting groups changed')
            probability[mask] = saved['p']
        else:
            status('cross_fitting', fold=k + 1, folds=3, trees=best['trees'])
            model = estimator(best['depth'], best['trees'], args.threads)
            model.fit(X['fit'][~mask], pairs.label[~mask], verbose=False)
            probability[mask] = model.predict_proba(X['fit'][mask])[:, 1]
            dump(path, {'rows': np.flatnonzero(mask), 'p': probability[mask], 'excluded': excluded})
            # Native JSON also makes each new model portable across platforms.
            model.save_model(ART / f'oof_base{k}.json')
            del model
            gc.collect()
        coverage[mask] += 1
    if not np.all(coverage == 1):
        raise ValueError('Every fitting pair must have exactly one out-of-fold prediction')
    save_array(ROOT / 'fit_oof.npy', probability)
    return probability


def meta_features(cfg, split, pairs, X, probabilities):
    import hashlib
    path = FEATURES / (split + '_meta.joblib')
    sig = {'pairs': pair_signature(pairs), 'probabilities': hashlib.sha256(probabilities.tobytes()).hexdigest()}
    if path.exists():
        saved = joblib.load(path)
        if saved['signature'] != sig:
            raise ValueError('Context cache changed')
        return saved['features']
    con = connect(Path(cfg.working_dir) / 'train/records.sqlite', True)
    blocks = []
    for lo, hi in groups(pairs):
        p, q = pairs.iloc[lo:hi], probabilities[lo:hi]
        block = FEATURES / (split + '_meta_parts') / f'{lo:08d}.joblib'
        if block.exists():
            saved = joblib.load(block)
            if saved['signature'] != sig:
                raise ValueError('Context checkpoint changed')
            extra = saved['features']
        else:
            extra = pd.concat([context(p, q), peer_features(p, q, con)], axis=1)
            dump(block, {'signature': sig, 'features': extra})
        blocks.append(extra)
        status('preparing_context', split=split, pairs_done=hi, pairs_total=len(pairs))
    result = pd.concat([X, pd.concat(blocks)], axis=1).astype(np.float32)
    dump(path, {'signature': sig, 'features': result})
    con.close()
    return result


def train_meta(cfg, args, data, X, best, oof):
    base = joblib.load(best['path'])
    meta = {}
    for split in data:
        p = oof if split == 'fit' else base['matcher'].estimator.predict_proba(X[split])[:, 1]
        meta[split] = meta_features(cfg, split, data[split]['pairs'], X[split], p)
    early, calibration = calibration_masks(data['calibration'])
    rows = [best]
    for depth in args.meta_depths:
        name = f'xgb_context_d{depth}'
        path, report = ART / (name + '.joblib'), OUT / (name + '.json')
        if path.exists() and report.exists():
            rows.append(json.loads(report.read_text()))
            continue
        status('training_context', model=name, maximum_trees=args.iterations)
        model = estimator(depth, args.iterations, args.threads, seed=43, stopping=True)
        model.fit(meta['fit'], data['fit']['pairs'].label,
                  eval_set=[(meta['calibration'][early], data['calibration']['pairs'].label[early])], verbose=100)
        matcher = CalibratedMatcher(model)
        matcher.calibrate(meta['calibration'][calibration], data['calibration']['pairs'].label[calibration])
        p = matcher.predict(meta['validation'])
        deployed = ownership_probabilities(data['validation']['pairs'], p)
        threshold, table = choose(data['validation'], deployed)
        row = {'name': name, 'kind': 'context', 'base_path': best['path'], 'depth': depth,
               'trees': model.best_iteration + 1, 'threshold': threshold, 'path': str(path),
               **evaluate(data['validation'], deployed, threshold)}
        dump(path, {'matcher': matcher, 'feature_names': meta['fit'].columns.tolist(), 'threshold': threshold})
        save_array(ROOT / (name + '_validation.npy'), p)
        json_write(report, row)
        table.to_csv(OUT / (name + '_thresholds.tsv'), sep='\t', index=False)
        rows.append(row)
        status('context_selection_result', **row)
    json_write(OUT / 'model_comparison.json', rows)
    return max(rows, key=lambda r: r['f0.5'])


def reserve_confirmation(cfg, n):
    selection = np.load(Path(cfg.working_dir) / 'entity_selection.npz')
    used = np.concatenate([selection[k] for k in selection.files])
    all_ids = np.arange(1, 2206822, dtype=np.uint32)
    # Reconstruct and exclude all three earlier campaign holdouts, including
    # the unfinished v4 holdout. None can become this campaign's fresh test.
    for seed in (20260926, 20260927, 20260928):
        eligible = np.setdiff1d(all_ids, used)
        prior = np.random.default_rng(seed).choice(eligible, 5000, replace=False)
        used = np.union1d(used, prior)
    original = Path('../../work/real_v1/entity_selection.npz')
    if original.exists():
        old = np.load(original)
        used = np.union1d(used, np.concatenate([old[k] for k in old.files]))
    ids = np.sort(np.random.default_rng(20261001).choice(np.setdiff1d(all_ids, used), n, replace=False))
    if np.intersect1d(ids, used).size:
        raise ValueError('Confirmation overlaps prior samples')
    frozen_json(OUT / 'confirmation_split.json', {'seed': 20261001, 'anchors': n,
                                                'excluded': len(used), 'ids': ids.tolist()})
    return ids


def confirmation_part(cfg, ids):
    path = ROOT / 'confirmation.joblib'
    if path.exists():
        part = joblib.load(path)
        if part['ids'] != ids.tolist():
            raise ValueError('Confirmation sample changed')
        return part
    blocker = DiskBlocker(cfg, 'train')
    engineer = joblib.load(Path(cfg.working_dir) / 'feature_engineer.joblib')
    result = {'pairs': [], 'features': [], 'truth_counts': {}, 'ids': ids.tolist()}
    for start in range(0, len(ids), 100):
        file = ROOT / 'confirmation_parts' / f'{start:06d}.joblib'
        if file.exists():
            saved = joblib.load(file)
        else:
            anchors = fetch_records(blocker.con, 'anchors', ids[start:start + 100])
            pairs, targets, _, truth = blocker.retrieve(anchors, with_truth=True)
            records = preprocess(pd.concat([anchors[COLUMNS[:4]], targets[COLUMNS[:4]]], ignore_index=True))
            X = engineer.transform(pairs, engineer.prepare(records))
            saved = {'pairs': pairs, 'features': X, 'truth_counts': {k: len(v) for k, v in truth.items()}}
            dump(file, saved)
        result['pairs'].append(saved['pairs'])
        result['features'].append(saved['features'])
        result['truth_counts'].update(saved['truth_counts'])
        status('preparing_fresh_confirmation', anchors_done=min(start + 100, len(ids)), anchors_total=len(ids))
    result['pairs'] = pd.concat(result['pairs'], ignore_index=True)
    result['features'] = pd.concat(result['features'], ignore_index=True)
    dump(path, result)
    blocker.close()
    return result


def per_entity_scores(part, p, threshold):
    truth = part['truth_counts']
    pos = {v: i for i, v in enumerate(truth)}
    ix = np.array([pos[v] for v in part['pairs'].source1_entity_id])
    keep = p >= threshold
    tp = np.bincount(ix, weights=keep * part['pairs'].label.to_numpy(), minlength=len(truth))
    predicted = np.bincount(ix, weights=keep, minlength=len(truth))
    den = .25 * np.array(list(truth.values())) + predicted
    return np.divide(1.25 * tp, den, out=np.ones(len(truth)), where=den > 0)


def promote(confirmation, floor):
    # This is a local validation requirement, never a leaderboard guarantee.
    return bool(confirmation['f0.5'] >= floor and confirmation['bootstrap_95ci'][0] >= floor)


def confirm(cfg, args, best):
    path = ART / 'selected_model.joblib'
    if not path.exists():
        model = joblib.load(best['path'])
        bundle = {'base': joblib.load(best['base_path']) if best['kind'] == 'context' else model,
                  'context': model if best['kind'] == 'context' else None,
                  'threshold': best['threshold'], 'selection': best,
                  'feature_schema': 'local_icu_v1', 'no_external_entity_data': True}
        dump(path, bundle)
    frozen = {'sha256': filehash(path), 'threshold': best['threshold'], 'selection': best,
              'local_floor': args.floor, 'target': .98,
              'scope': 'Frozen before fresh confirmation; no hidden test labels available'}
    frozen_json(OUT / 'frozen_selection.json', frozen)
    report = OUT / 'confirmation.json'
    if report.exists():
        result = json.loads(report.read_text())
        if result['model_sha256'] != frozen['sha256']:
            raise ValueError('Confirmation belongs to a different model')
        return result
    ids = reserve_confirmation(cfg, args.confirmation_anchors)
    part = confirmation_part(cfg, ids)
    X = prepare_split(cfg, 'confirmation', part)
    model = joblib.load(path)
    if model['context'] is not None:
        raw = model['base']['matcher'].estimator.predict_proba(X)[:, 1]
        X = meta_features(cfg, 'confirmation', part['pairs'], X, raw)
        p = model['context']['matcher'].predict(X)
    else:
        p = model['base']['matcher'].predict(X)
    save_array(ROOT / 'confirmation_probabilities.npy', p)
    pair_metrics = evaluate(part, p, best['threshold'])
    p = ownership_probabilities(part['pairs'], p)
    result = evaluate(part, p, best['threshold'])
    result['pair_threshold_metrics'] = pair_metrics
    scores = per_entity_scores(part, p, best['threshold'])
    rng = np.random.default_rng(20261002)
    bootstrap = [float(scores[rng.integers(0, len(scores), len(scores))].mean()) for _ in range(2000)]
    result.update({'bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist(),
                   'model_sha256': frozen['sha256'], 'threshold': best['threshold'],
                   'anchors': len(ids), 'leaderboard_score': None,
                   'limitation': 'Training holdout covers India and US; France has no supplied labels. Ownership competition is measured only among sampled anchors.'})
    con = connect(Path(cfg.working_dir) / 'train/records.sqlite', True)
    frame = fetch_records(con, 'anchors', ids).set_index('entity_id')
    countries = frame.loc[list(part['truth_counts']), 'country_norm'].to_numpy()
    result['country_f05'] = {c: float(scores[countries == c].mean()) for c in np.unique(countries)}
    con.close()
    result['eligible_for_local_inference'] = promote(result, args.floor)
    result['reached_local_target'] = result['f0.5'] >= .98
    json_write(report, result)
    status('fresh_confirmation_result', **result)
    return result


def init_scoring(cfg_dict, model_path):
    from .provenance import PairProvenance
    cfg = Config(**cfg_dict)
    W['cfg'], W['con'] = cfg, connect(Path(cfg.working_dir) / 'test/records.sqlite', True)
    W['model'] = joblib.load(model_path)
    for member in (W['model']['base'], W['model']['context']):
        if member is not None:
            member['matcher'].estimator.set_params(n_jobs=1)
    W['engineer'] = joblib.load(Path(cfg.working_dir) / 'feature_engineer.joblib')
    W['keys'] = np.load('work/improvements/test_target_keys.npy', mmap_mode='r')
    W['rids'] = np.load('work/improvements/test_target_rids.npy', mmap_mode='r')
    W['provenance'] = PairProvenance(Path(cfg.working_dir) / 'native',
                                    joblib.load('work/improvements/test_posting_counts.joblib'), cfg.retrieval_posting_limit)
    W['stats'] = joblib.load('work/improvements/evidence_stats_test.joblib')
    W['transliterator'] = LocalTransliterator()


def score_batch(task):
    start, rows = task
    con = W['con']
    anchors = fetch_records(con, 'anchors', range(start + 1, start + len(rows) + 1))
    if anchors.entity_id.tolist() != [r[0] for r in rows]:
        raise ValueError('Candidate row order does not match test records')
    ids = [v for _, values in rows for v in values]
    output = np.empty(len(ids), dtype=DTYPE)
    if not ids:
        return start, len(rows), output
    encoded = np.fromiter(map(numeric_id, ids), dtype=np.uint64)
    index = np.searchsorted(W['keys'], encoded)
    if np.any(index >= len(W['keys'])) or not np.array_equal(W['keys'][index], encoded):
        raise ValueError('Unknown test candidate')
    rids = W['rids'][index]
    targets = fetch_records(con, 'targets', rids)
    repeated = np.repeat(np.arange(len(rows)), [len(v) for _, v in rows])
    masks = W['provenance'].masks(anchors, targets, repeated, np.searchsorted(targets.rid.to_numpy(), rids))
    if not np.all(masks):
        raise ValueError('Candidate provenance changed')
    pairs = pd.DataFrame({'anchor_rid': anchors.rid.to_numpy()[repeated], 'target_rid': rids,
                           'source1_entity_id': anchors.entity_id.to_numpy()[repeated],
                           'candidate_entity_id': ids, 'blocking_rules': [rules_text(v) for v in masks]})
    records = preprocess(pd.concat([anchors[COLUMNS[:4]], targets[COLUMNS[:4]]], ignore_index=True))
    X = W['engineer'].transform(pairs, W['engineer'].prepare(records))
    X = pd.concat([X, enrich(pairs, con, W['stats']), detail_features(pairs, con, W['transliterator'])], axis=1).astype(np.float32)
    bundle = W['model']
    if X.columns.tolist() != bundle['base']['feature_names']:
        raise ValueError('Deployment base feature schema mismatch')
    if bundle['context'] is not None:
        p = bundle['base']['matcher'].estimator.predict_proba(X)[:, 1]
        X = pd.concat([X, context(pairs, p), peer_features(pairs, p, con)], axis=1).astype(np.float32)
        if X.columns.tolist() != bundle['context']['feature_names']:
            raise ValueError('Deployment context feature schema mismatch')
        p = bundle['context']['matcher'].predict(X)
    else:
        p = bundle['base']['matcher'].predict(X)
    output['anchor'], output['target'], output['p'] = pairs.anchor_rid, rids, p
    return start, len(rows), output


def score_test(cfg, args, confirmation):
    if not promote(confirmation, args.floor):
        raise ValueError('Fresh confirmation did not meet the local score floor')
    root = ROOT / 'test_scores'
    root.mkdir(parents=True, exist_ok=True)
    from .provenance import prepare_counts
    lookup_cache(cfg)
    prepare_counts(cfg)
    make_stats(cfg, 'test')
    signature = {'model_sha256': filehash(ART / 'selected_model.joblib'),
                  'candidate_sha256': filehash(args.candidates), 'batch': args.batch,
                  'campaign_signature': filehash(OUT / 'signature.json')}
    frozen_json(root / 'signature.json', signature)
    lock = {'sha256': signature['model_sha256'], 'threshold': confirmation['threshold']}
    frozen_json(root / 'selection.json', lock)
    pending, completed, pairs_scored = deque(), 0, 0
    start_time = time.time()

    def consume():
        nonlocal completed, pairs_scored
        start, n, values = pending.popleft().result()
        save_array(root / f'{start:08d}.npy', values)
        completed += n
        pairs_scored += len(values)
        status('test_inference', anchors_done=start + n, anchors_total=1732544,
               newly_scored_anchors=completed, pairs_scored_this_run=pairs_scored,
               elapsed_seconds=time.time() - start_time)

    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_scoring, initargs=(cfg.to_dict(), str(ART / 'selected_model.joblib'))) as pool:
        total = 0
        for task in batches(args.candidates, args.batch, None):
            start, rows = task
            total = start + len(rows)
            path = root / f'{start:08d}.npy'
            if path.exists():
                saved = np.load(path, mmap_mode='r')
                expected = np.repeat(np.arange(start + 1, total + 1), [len(v) for _, v in rows])
                if saved.dtype != DTYPE or not np.array_equal(saved['anchor'], expected) or not np.isfinite(saved['p']).all() or np.any((saved['p'] < 0) | (saved['p'] > 1)):
                    raise ValueError('Invalid score checkpoint')
                continue
            pending.append(pool.submit(score_batch, task))
            if len(pending) >= args.workers * 2:
                consume()
        while pending:
            consume()
    if total != 1732544:
        raise ValueError('Unexpected test anchor coverage')
    json_write(root / 'SCORING_COMPLETE.json', {'anchors': total, 'signature': signature})
    return root


def main(args):
    if not .95 <= args.floor <= 1 or args.confirmation_anchors < 5000:
        raise ValueError('Keep the requested local floor >= .95 and at least 5000 fresh confirmation anchors')
    for directory in (ROOT, OUT, ART, FEATURES):
        directory.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(args.config)
    # One process owns this campaign. An OS lock releases automatically on exit.
    import fcntl
    lock = (ROOT / 'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen_json(OUT / 'signature.json', training_signature(cfg, args))
    json_write(OUT / 'baseline_preserved.json', {
        'user_reported_leaderboard': .94,
        'baseline_directories': ['outputs/improved_unique_owner', 'outputs/improved_pair_threshold'],
        'new_artifacts': str(ART), 'new_output': args.output,
        'policy': 'Separate outputs; no existing trained model or submission is overwritten.'})
    status('loading_existing_training_caches', device='CPU', audits=False)
    data = {s: read_part(cfg, s) for s in ('fit', 'calibration', 'validation')}
    sets = [set(d['truth_counts']) for d in data.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Fitting, calibration and selection anchors overlap')
    X = {s: prepare_split(cfg, s, part) for s, part in data.items()}
    base = train_base(cfg, args, data, X)
    oof = cross_fit(args, data, X, base)
    best = train_meta(cfg, args, data, X, base, oof)
    del data, X, oof
    gc.collect()
    confirmation = confirm(cfg, args, best)
    if not promote(confirmation, args.floor):
        status('not_promoted', reason='Fresh holdout lower 95% confidence bound is below the requested local floor.',
               confirmation=confirmation, existing_submission_preserved=True)
        return
    if not args.infer:
        status('validated_locally', confirmation=confirmation, inference_started=False)
        return
    root = score_test(cfg, args, confirmation)
    from .finalize_improved import finalize
    validation = Path(args.output) / 'validation.json'
    if not validation.exists():
        if Path(args.output).exists() and any(Path(args.output).iterdir()):
            archive = ROOT / ('interrupted_export_' + str(time.time_ns()))
            Path(args.output).rename(archive)
        status('exporting_submission', output=args.output)
        finalize(argparse.Namespace(config=args.config, work=str(root), selection=str(root / 'selection.json'),
                                    output=args.output, candidates=args.candidates, unique_owner=True))
    else:
        report = json.loads(validation.read_text())
        if report['model_sha256'] != filehash(ART / 'selected_model.joblib') or report['matching_sha256'] != filehash(Path(args.output) / 'matching_results.tsv'):
            raise ValueError('Existing export does not match this run')
    status('official_validation', output=args.output)
    from .real_pipeline import official_validate
    official_validate(replace(cfg, output_dir=args.output, reports_dir=str(OUT)))
    status('complete', output=args.output, official_validator_exit_code=0,
           local_confirmation=confirmation['f0.5'], leaderboard_score=None,
           reached_local_target=confirmation['reached_local_target'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/local_098.json')
    parser.add_argument('--base-depths', nargs='+', type=int, default=[7, 10])
    parser.add_argument('--meta-depths', nargs='+', type=int, default=[5, 8])
    parser.add_argument('--iterations', type=int, default=1800)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=250)
    parser.add_argument('--confirmation-anchors', type=int, default=10000)
    parser.add_argument('--floor', type=float, default=.95)
    parser.add_argument('--candidates', default='outputs/improved_unique_owner/candidate_pairs.tsv')
    parser.add_argument('--output', default='outputs/local_098_v1_unique')
    parser.add_argument('--infer', action='store_true')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception:
        status('failed', error=traceback.format_exc())
        raise
