"""Offline enrichment of existing candidate pairs, with resumable feature shards."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .disk_store import connect, fetch_records
from .evidence import enrich, fold
from .detail_evidence import features as detail_features
from .raw_evidence import features as raw_features


def frame_signature(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def require_signature(path, signature):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != signature:
            raise ValueError(f'Feature inputs changed; use a new cache directory: {path.parent}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(signature, indent=2), encoding='utf-8')


def save(path, value):
    temporary = path.with_suffix('.partial')
    joblib.dump(value, temporary)
    temporary.replace(path)


def fit_stats(pairs, con):
    """Fit frequency features on fitting anchors only, never holdout/test rows."""
    anchors = fetch_records(con, 'anchors', pairs.anchor_rid.unique())
    counts, document_frequency, sizes = Counter(), Counter(), Counter()
    for row in anchors.itertuples(index=False):
        name = fold(row.name_norm)
        counts[(row.country_norm, name)] += 1
        sizes[row.country_norm] += 1
        for token in set(name.split()):
            document_frequency[(row.country_norm, token)] += 1
    return {'counts': dict(counts), 'df': dict(document_frequency), 'sizes': dict(sizes)}


def enrich_splits(data, work, cache, batch_size=5000):
    if batch_size < 1:
        raise ValueError('Feature batch size must be positive')
    database = work / 'train/records.sqlite'
    if not database.is_file():
        raise FileNotFoundError(database)
    sources = ['local_training_features.py', 'raw_evidence.py', 'detail_evidence.py', 'evidence.py']
    signature = {
        'recipe': 'fit-only-frequency+normalized+transliteration+raw-alignment-v1',
        'database': [str(database.resolve()), database.stat().st_size, database.stat().st_mtime_ns],
        'pairs': {split: frame_signature(part['pairs'][['anchor_rid', 'target_rid', 'source1_entity_id', 'candidate_entity_id']])
                  for split, part in data.items()},
        'code': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in sources},
        'batch_size': batch_size,
    }
    require_signature(cache / 'signature.json', signature)
    con = connect(database, readonly=True)
    try:
        stats_path = cache / 'fit_stats.joblib'
        if stats_path.exists():
            stats = joblib.load(stats_path)
        else:
            stats = fit_stats(data['fit']['pairs'], con)
            save(stats_path, stats)
        for split, part in data.items():
            directory = cache / split
            directory.mkdir(exist_ok=True)
            blocks = []
            pairs = part['pairs']
            for lo in range(0, len(pairs), batch_size):
                batch = pairs.iloc[lo:lo + batch_size]
                path = directory / f'{lo:08d}.joblib'
                if path.exists():
                    extra = joblib.load(path)
                else:
                    extra = pd.concat([enrich(batch, con, stats), detail_features(batch, con, stats),
                                       raw_features(batch, con)], axis=1).astype(np.float32)
                    save(path, extra)
                if not extra.index.equals(batch.index) or not np.isfinite(extra.to_numpy()).all():
                    raise ValueError(f'Invalid feature shard: {path}')
                blocks.append(extra)
                print('ENRICH_FEATURES', split, min(lo + batch_size, len(pairs)), len(pairs), flush=True)
            part['features'] = pd.concat([part['features'], pd.concat(blocks)], axis=1)
            if part['features'].columns.duplicated().any():
                raise ValueError('Duplicate enriched feature columns')
    finally:
        con.close()
    return {'signature': signature, 'stats_path': str(stats_path.resolve()),
            'stats': stats, 'code_files': sources}
