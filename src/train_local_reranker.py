"""Learn candidate context on separate anchors using a saved enriched base model."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .context_experiments import context
from .disk_store import connect
from .experiments import choose, evaluate
from .local_training_features import frame_signature, require_signature, save
from .model import CalibratedMatcher
from .peer_evidence import features as peer_features


def split_entities(truth):
    ids = np.asarray(sorted(truth), dtype=object)
    if len(ids) < 10:
        raise ValueError('At least ten calibration anchors are required')
    np.random.default_rng(20260930).shuffle(ids)
    n = len(ids) // 5
    return {'fit': ids[2*n:], 'early': ids[n:2*n], 'calibrate': ids[:n]}


def peers_in_batches(pairs, probabilities, con):
    ordered = pairs.sort_values('anchor_rid', kind='stable')
    probability = pd.Series(probabilities, index=pairs.index).loc[ordered.index].to_numpy()
    anchors = ordered.anchor_rid.to_numpy()
    starts = np.r_[0, np.flatnonzero(anchors[1:] != anchors[:-1]) + 1, len(anchors)]
    blocks = []
    for g in range(0, len(starts)-1, 250):
        lo, hi = starts[g], starts[min(g+250, len(starts)-1)]
        blocks.append(peer_features(ordered.iloc[lo:hi], probability[lo:hi], con))
        print('RERANK_PEERS', hi, len(pairs), flush=True)
    return pd.concat(blocks).reindex(pairs.index)


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Use a new output directory: {args.output}')
    base = joblib.load(args.base)
    if not base.get('enrichment'):
        raise ValueError('An enriched local base model is required')
    expected = base['enrichment']['signature']
    if json.loads((args.cache / 'signature.json').read_text(encoding='utf-8')) != expected:
        raise ValueError('Enrichment cache differs from trained base model')
    base_hash = hashlib.sha256(args.base.read_bytes()).hexdigest()
    data, X = {}, {}
    con = connect(args.work / 'train/records.sqlite', readonly=True)
    try:
        for split in ('calibration', 'validation'):
            part = joblib.load(args.work / 'features' / (split + '.joblib'))
            pairs = part['pairs']
            fingerprint = frame_signature(pairs[['anchor_rid', 'target_rid', 'source1_entity_id', 'candidate_entity_id']])
            if fingerprint != expected['pairs'][split]:
                raise ValueError('Candidate rows differ from base training')
            extra = pd.concat([joblib.load(p) for p in sorted((args.cache / split).glob('*.joblib'))])
            if not extra.index.equals(pairs.index):
                raise ValueError('Enrichment rows are not aligned')
            features = pd.concat([part.pop('features'), extra], axis=1)
            if list(features.columns) != base['feature_names']:
                raise ValueError('Base feature schema differs')
            # Raw scores avoid the base sigmoid fitted on these same calibration labels.
            p = base['matcher'].estimator.predict_proba(features)[:, 1]
            destination = args.cache / ('rerank_' + split)
            signature = {'base_sha256': base_hash, 'pairs': fingerprint,
                         'peer_code': hashlib.sha256((Path(__file__).parent / 'peer_evidence.py').read_bytes()).hexdigest(),
                         'probabilities': hashlib.sha256(p.tobytes()).hexdigest()}
            require_signature(destination / 'signature.json', signature)
            peer_path = destination / 'peers.joblib'
            if peer_path.exists():
                peers = joblib.load(peer_path)
            else:
                peers = peers_in_batches(pairs, p, con)
                save(peer_path, peers)
            if not peers.index.equals(pairs.index):
                raise ValueError('Peer features are not aligned')
            X[split] = pd.concat([features, context(pairs, p), peers], axis=1)
            data[split] = part
    finally:
        con.close()
    if set(data['calibration']['truth_counts']) & set(data['validation']['truth_counts']):
        raise ValueError('Calibration and validation entities overlap')
    ids = split_entities(data['calibration']['truth_counts'])
    masks = {name: data['calibration']['pairs'].source1_entity_id.isin(values).to_numpy()
             for name, values in ids.items()}
    labels = data['calibration']['pairs'].label
    for name, mask in masks.items():
        if set(labels[mask].unique()) != {0, 1}:
            raise ValueError(f'{name}: both classes required')
    estimator = CatBoostClassifier(iterations=1800, depth=7, learning_rate=.035,
        l2_leaf_reg=10, loss_function='Logloss', task_type='GPU', random_seed=43,
        thread_count=3, allow_writing_files=False, verbose=100)
    estimator.fit(X['calibration'][masks['fit']], labels[masks['fit']],
        eval_set=(X['calibration'][masks['early']], labels[masks['early']]), early_stopping_rounds=150)
    matcher = CalibratedMatcher(estimator)
    matcher.calibrate(X['calibration'][masks['calibrate']], labels[masks['calibrate']])
    probabilities = matcher.predict(X['validation'])
    threshold, table = choose(data['validation'], probabilities)
    metrics = evaluate(data['validation'], probabilities, threshold)
    report = {'metrics': metrics, 'threshold': threshold, 'base_sha256': base_hash,
        'base_model': str(args.base.resolve()), 'trees': estimator.tree_count_,
        'entity_counts': {name: len(values) for name, values in ids.items()},
        'scope': 'Validation is model/threshold selection, not an official or independent test score. '
                 'Base early stopping previously used these calibration anchors. '
                 'Reranker fitting, early stopping and sigmoid calibration use separate anchors.',
        'artifact_type': 'local_context_reranker'}
    args.output.mkdir(parents=True)
    save(args.output / 'model.joblib', {'matcher': matcher, 'threshold': threshold,
        'feature_names': X['validation'].columns.tolist(), 'base': base,
        'report': report, 'artifact_type': report['artifact_type']})
    (args.output / 'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    table.to_csv(args.output / 'thresholds.tsv', sep='\t', index=False)
    np.save(args.output / 'validation_probabilities.npy', probabilities)
    print('RERANK_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('artifacts/local_training_enriched_d10/model.joblib'))
    parser.add_argument('--work', type=Path, default=Path('work/real_v1'))
    parser.add_argument('--cache', type=Path, default=Path('work/local_enriched_v1'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/local_training_context'))
    run(parser.parse_args())
