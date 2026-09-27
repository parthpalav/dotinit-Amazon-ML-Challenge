"""Train from the transferred real_v1 caches without campaign assets or audits."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import sys
import time


SPLITS = ('fit', 'calibration', 'validation')


def required_inputs(work):
    paths = [work / 'features' / (split + '.joblib') for split in SPLITS]
    paths += [work / 'train/store_manifest.json']
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing local training inputs:\n' + '\n'.join(missing))
    return paths


def check_split_ids(groups):
    """Reject leakage before fitting; include anchors with no candidate rows."""
    seen = set()
    for split in SPLITS:
        ids = set(groups[split])
        if not ids:
            raise ValueError(f'{split}: empty entity split')
        if seen.intersection(ids):
            raise ValueError(f'{split}: Source 1 entities overlap another split')
        seen.update(ids)


def run(args):
    required_inputs(args.work)
    # Lazy imports allow --help and missing-input errors before dependency setup.
    import joblib
    import numpy as np
    from catboost import CatBoostClassifier
    from .model import CalibratedMatcher
    from .real_metrics import tune, entity_scores

    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}; choose a new --output')
    data = {s: joblib.load(args.work / 'features' / (s + '.joblib')) for s in SPLITS}
    check_split_ids({s: data[s]['truth_counts'] for s in SPLITS})
    enrichment = None
    if args.enriched:
        from .local_training_features import enrich_splits
        enrichment = enrich_splits(data, args.work, args.feature_cache)
    columns = list(data['fit']['features'].columns)
    for split, part in data.items():
        pairs, features = part['pairs'], part['features']
        if len(pairs) != len(features) or not pairs.index.equals(features.index):
            raise ValueError(f'{split}: features and labels are not aligned')
        if list(features.columns) != columns:
            raise ValueError(f'{split}: feature schema differs from fit')
        if not set(pairs.source1_entity_id).issubset(part['truth_counts']):
            raise ValueError(f'{split}: candidate entity missing from truth counts')
        if set(pairs.label.unique()) != {0, 1}:
            raise ValueError(f'{split}: binary labels with both classes required')

    # These are the original cached features, not a replacement for the missing
    # campaign's contextual probabilities or pretrained neural representation.
    counts = json.loads((args.work / 'train/store_manifest.json').read_text(encoding='utf-8'))['counts']
    target_count = counts['source2'] + counts['source3']
    args.output.mkdir(parents=True)
    started = time.time()
    model = CatBoostClassifier(
        iterations=args.iterations, depth=args.depth, learning_rate=.035,
        l2_leaf_reg=10, loss_function='Logloss', task_type=args.device,
        random_seed=42, thread_count=args.threads, allow_writing_files=False,
        verbose=100,
    )
    print('LOCAL_TRAIN_START', args.device, flush=True)
    model.fit(data['fit']['features'], data['fit']['pairs'].label,
              eval_set=(data['calibration']['features'], data['calibration']['pairs'].label),
              early_stopping_rounds=150)
    matcher = CalibratedMatcher(model)
    matcher.calibrate(data['calibration']['features'], data['calibration']['pairs'].label)
    validation = data['validation']
    probabilities = matcher.predict(validation['features'])
    truth = validation['truth_counts']
    positions = {entity: i for i, entity in enumerate(truth)}
    indices = np.array([positions[x] for x in validation['pairs'].source1_entity_id], dtype=np.int32)
    labels = validation['pairs'].label.to_numpy()
    truth_counts = np.array(list(truth.values()))
    threshold, table = tune(indices, labels, probabilities, truth_counts, target_count)
    metrics = entity_scores(indices, labels, probabilities, truth_counts, threshold, target_count)
    report = {
        'training_source': str(args.work.resolve()), 'device': args.device,
        'threshold': threshold, 'metrics': metrics, 'trees': model.tree_count_,
        'seconds': time.time() - started, 'feature_names': columns,
        'split_entities': {s: len(data[s]['truth_counts']) for s in SPLITS},
        'scope': 'Validation is model/threshold-selection data, not a competition score. '
                 'Calibration data is also used for early stopping. This is not the campaign ensemble.',
        'enriched': args.enriched,
        'requested_parameters': {'depth': args.depth, 'iterations': args.iterations},
        'artifact_type': 'local_cached_feature_matcher',
        'runtime': {'python': sys.version, 'packages': {
            package: version(package) for package in
            ('numpy', 'pandas', 'scikit-learn', 'joblib', 'catboost')
        }},
    }
    temporary = args.output / 'model.joblib.partial'
    joblib.dump({'matcher': matcher, 'threshold': threshold, 'feature_names': columns,
                 'report': report, 'artifact_type': report['artifact_type'],
                 'enrichment': enrichment}, temporary)
    temporary.replace(args.output / 'model.joblib')
    table.to_csv(args.output / 'thresholds.tsv', sep='\t', index=False)
    np.save(args.output / 'validation_probabilities.npy', probabilities)
    (args.output / 'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('LOCAL_TRAIN_COMPLETE', str(args.output), json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('work/real_v1'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/local_training'))
    parser.add_argument('--device', choices=['GPU', 'CPU'], default='GPU')
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=1800)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--enriched', action='store_true', help='Add offline raw-text, number and transliteration features')
    parser.add_argument('--feature-cache', type=Path, default=Path('work/local_enriched_v1'))
    args = parser.parse_args()
    if min(args.depth, args.iterations, args.threads) < 1:
        parser.error('depth, iterations and threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
