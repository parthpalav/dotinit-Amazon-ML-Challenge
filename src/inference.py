"""Batch scoring, atomic TSV generation, and automatic local validation."""
import csv
import logging
from pathlib import Path
import tempfile
import joblib
import pandas as pd
from .blocking import Blocker
from .config import Config
from .data_loader import load_sources
from .embeddings import LocalEmbeddings
from .preprocessing import preprocess
from .submission import validate_submission

LOG = logging.getLogger(__name__)


def infer(config: Config):
    # joblib is pickle-based: only load artifacts you trust.
    artifact = joblib.load(config.model_path)
    if artifact.get("artifact_version") != 1:
        raise ValueError("Unsupported model artifact version")
    trained_config = Config(**artifact["config"])
    anchors, targets = (preprocess(x) for x in load_sources(config.dataset_dir, "test"))
    # Retrieval settings must match training; only runtime paths are supplied by the CLI.
    blocker = Blocker(targets, trained_config)
    records = pd.concat([anchors, targets], ignore_index=True)
    engineer = artifact["feature_engineer"]
    cache = engineer.prepare(records)
    embedding_cache = None
    if artifact["uses_embeddings"]:
        directory = config.embedding_model_dir or trained_config.embedding_model_dir
        embedder = LocalEmbeddings(directory, artifact["embedding_fingerprint"])
        embedding_cache = embedder.prepare(records)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Each S1 is processed independently, so output need not retain every candidate in RAM.
    with tempfile.TemporaryDirectory(prefix=".submission-", dir=output) as temporary:
        matching_path = Path(temporary) / "matching_results.tsv"
        candidate_path = Path(temporary) / "candidate_pairs.tsv"
        scored_pairs = 0
        with matching_path.open("w", newline="", encoding="utf-8") as matching_file, candidate_path.open("w", newline="", encoding="utf-8") as candidate_file:
            mw, cw = csv.writer(matching_file, delimiter="\t"), csv.writer(candidate_file, delimiter="\t")
            mw.writerow(["source1_entity_id", "matched_entity_ids"])
            cw.writerow(["source1_entity_id", "candidate_entity_ids"])
            for i in range(len(anchors)):
                source = anchors.iloc[i].entity_id
                scored, selected = [], []
                for pairs in blocker.iter_pairs(anchors.iloc[i:i+1]):
                    features = engineer.transform(pairs, cache)
                    if embedding_cache is not None:
                        features = LocalEmbeddings.add_features(features, pairs, embedding_cache)
                    if features.columns.tolist() != artifact["feature_names"]:
                        raise ValueError("Inference feature schema does not match the fitted model")
                    probabilities = artifact["matcher"].predict(features)
                    scored.extend(pairs.candidate_entity_id.tolist())
                    scored_pairs += len(pairs)
                    selected.extend((float(p), target) for target, p in zip(pairs.candidate_entity_id, probabilities)
                                    if p >= artifact["threshold"])
                mw.writerow([source, ",".join(target for _, target in sorted(selected, key=lambda x: (-x[0], x[1])))])
                cw.writerow([source, ",".join(scored)])
                if (i + 1) % 1000 == 0:
                    LOG.info("Scored %d/%d Source 1 records", i + 1, len(anchors))
        report = validate_submission(matching_path, candidate_path, anchors, targets)
        if report["candidate_pairs"] != scored_pairs:
            raise ValueError("Exported candidate count differs from the scored count")
        matching_path.replace(output / matching_path.name)
        candidate_path.replace(output / candidate_path.name)
    LOG.info("Submission validated: %s", report)
    return report
