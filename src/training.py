"""Group-disjoint fitting, calibration, model selection, and audit artifacts."""
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import tempfile
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .blocking import Blocker
from .data_loader import load_sources, load_truth
from .embeddings import LocalEmbeddings
from .evaluation import blocking_comparison, candidate_report, evaluate, predictions_from_pairs
from .features import FeatureEngineer
from .model import CalibratedMatcher, model_candidates
from .preprocessing import preprocess
from .threshold import optimize_threshold

LOG = logging.getLogger(__name__)


def split_source1(anchors, truth, config):
    if len(anchors) < 10:
        raise ValueError("At least 10 Source 1 entities are required for fit/calibration/validation splitting")
    ids = sorted(anchors.entity_id)

    def split(values, fraction, seed):
        strata = [int(bool(truth[x])) for x in values]
        try:
            return train_test_split(values, test_size=fraction, random_state=seed, stratify=strata)
        except ValueError:
            LOG.warning("Singleton stratification is infeasible for this split; using a seeded S1 split")
            return train_test_split(values, test_size=fraction, random_state=seed)

    remaining, validation = split(ids, config.validation_fraction, config.seed)
    fitting, calibration = split(remaining, config.calibration_fraction / (1 - config.validation_fraction), config.seed + 1)
    sets = {"fit": set(fitting), "calibration": set(calibration), "validation": set(validation)}
    assert not (sets["fit"] & sets["validation"] or sets["fit"] & sets["calibration"] or sets["validation"] & sets["calibration"])
    return {name: anchors.loc[anchors.entity_id.isin(group)].copy() for name, group in sets.items()}


def labels_for(pairs, truth):
    return np.asarray([int(row.candidate_entity_id in truth[row.source1_entity_id])
                       for row in pairs.itertuples(index=False)], dtype=np.int8)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(config):
    started = perf_counter()
    raw_anchors, raw_targets = load_sources(config.dataset_dir, "train")
    truth = load_truth(config.dataset_dir, raw_anchors, raw_targets)
    anchors, targets = preprocess(raw_anchors), preprocess(raw_targets)
    splits = split_source1(anchors, truth, config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_truth = {name: {entity: truth[entity] for entity in frame.entity_id} for name, frame in splits.items()}
    pairs, reports, comparisons = {}, {}, []
    for name, frame in splits.items():
        # Independent candidate generation; no positives injected from ground truth.
        pairs[name] = Blocker(targets, config).generate(frame)
        reports[name] = candidate_report(pairs[name], split_truth[name], len(targets))
        LOG.info("%s candidate report: %s", name, reports[name])
        comparison = blocking_comparison(pairs[name], split_truth[name], frame, targets)
        comparison.insert(0, "split", name)
        comparisons.append(comparison)
    pd.concat(comparisons, ignore_index=True).to_csv(output / "blocking_comparison.tsv", sep="\t", index=False)
    (output / "candidate_metrics.json").write_text(json.dumps(reports, indent=2))
    for name, report in reports.items():
        recall = report["candidate_recall"]
        if recall is not None and recall < config.minimum_candidate_recall:
            message = f"{name} candidate recall {recall:.2%} is below target {config.minimum_candidate_recall:.2%}; tune blocking before trusting scores"
            if config.fail_on_low_candidate_recall:
                raise ValueError(message)
            LOG.warning(message)
    labels = {name: labels_for(frame, truth) for name, frame in pairs.items()}
    if len(np.unique(labels["fit"])) < 2:
        raise ValueError("Fitting candidates must contain both matches and nonmatches. Check data, split, and blocking.")
    engineer = FeatureEngineer(config.tfidf_max_features).fit(pd.concat([splits["fit"], targets], ignore_index=True))
    records = pd.concat([anchors, targets], ignore_index=True)
    cache = engineer.prepare(records)
    features = {name: engineer.transform(frame, cache) for name, frame in pairs.items()}
    models = model_candidates(config, labels["fit"])
    experiments, fitted = [], {}
    embedding_cache = None
    embedding_fingerprint = None
    if config.embedding_model_dir:
        embedder = LocalEmbeddings(config.embedding_model_dir)
        embedding_fingerprint = embedder.fingerprint
        embedding_cache = embedder.prepare(records)
        models["xgboost_embeddings"] = model_candidates(config, labels["fit"])["xgboost"]
    for name, estimator in models.items():
        run_started = perf_counter()
        uses_embeddings = name == "xgboost_embeddings"
        active_features = ({split: LocalEmbeddings.add_features(features[split], pairs[split], embedding_cache)
                            for split in splits} if uses_embeddings else features)
        estimator.fit(active_features["fit"], labels["fit"])
        matcher = CalibratedMatcher(estimator)
        if name != "exact_rule":
            matcher.calibrate(active_features["calibration"], labels["calibration"])
        probabilities = matcher.predict(active_features["validation"])
        threshold, threshold_table = optimize_threshold(pairs["validation"], probabilities,
            split_truth["validation"], len(targets), config.metric)
        predictions = predictions_from_pairs(pairs["validation"], probabilities, split_truth["validation"], threshold)
        scores = evaluate(split_truth["validation"], predictions, len(targets), config.metric)
        experiment = {"model": name, "status": "evaluated", "candidate_recall": reports["validation"]["candidate_recall"],
                      "average_candidates": reports["validation"]["average_candidates"], **scores,
                      "threshold": threshold, "calibration": matcher.calibration_status,
                      "runtime_seconds": perf_counter() - run_started, "eligible": True}
        experiments.append(experiment)
        fitted[name] = {"matcher": matcher, "threshold": threshold, "uses_embeddings": uses_embeddings,
                        "feature_names": active_features["fit"].columns.tolist()}
        threshold_table.to_csv(output / f"thresholds_{name}.tsv", sep="\t", index=False)
        diagnostic = pairs["validation"].copy()
        diagnostic["label"] = labels["validation"]
        diagnostic["probability"] = probabilities
        diagnostic["selected"] = probabilities >= threshold
        diagnostic.to_csv(output / f"validation_pairs_{name}.tsv", sep="\t", index=False)
        LOG.info("%s F0.5=%.5f precision=%.5f recall=%.5f threshold=%.5f singleton_accuracy=%s",
                 name, scores["f0.5"], scores["precision"], scores["recall"], threshold, scores["singleton_accuracy"])
    if config.embedding_model_dir:
        traditional = next(row for row in experiments if row["model"] == "xgboost")
        semantic = next(row for row in experiments if row["model"] == "xgboost_embeddings")
        semantic["eligible"] = semantic["f0.5"] > traditional["f0.5"]
    else:
        experiments.append({"model": "xgboost_embeddings", "status": "skipped_no_permitted_local_model", "eligible": False})
    table = pd.DataFrame(experiments)
    eligible = table[table.eligible].sort_values(["f0.5", "precision", "model"], ascending=[False, False, True], kind="stable")
    winner = str(eligible.iloc[0].model)
    table["selected"] = table.model.eq(winner)
    table.to_csv(output / "model_comparison.tsv", sep="\t", index=False)
    split_manifest = pd.DataFrame([(entity, split) for split, frame in splits.items() for entity in frame.entity_id],
                                 columns=["source1_entity_id", "split"])
    split_manifest.to_csv(output / "split_manifest.tsv", sep="\t", index=False)
    versions = {package: importlib.metadata.version(package) for package in
                ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "rapidfuzz", "joblib")}
    manifest = {"selected_model": winner, "config": config.to_dict(), "versions": versions,
                "python": platform.python_version(), "runtime_seconds": perf_counter() - started,
                "metric_definition": "Mean per-S1 F0.5; empty/empty=1, one empty=0" if config.metric == "entity_macro" else "Micro pair F0.5; singleton diagnostics separate",
                "validation_use": "Model and threshold selection; this is not an untouched generalization estimate",
                "tfidf_fit_corpus": "Fitting S1 plus all training S2/S3 reference records; no held-out S1 or test records",
                "negative_sampling": "All generated candidates, no subsampling or positive injection",
                "input_sha256": {p.name: file_sha256(p) for p in sorted((Path(config.dataset_dir) / "train").glob("*.tsv"))},
                "candidate_metrics": reports, "embedding_fingerprint": embedding_fingerprint}
    artifact = {"artifact_version": 1, "model_name": winner, "config": config.to_dict(),
                "feature_engineer": engineer, "embedding_fingerprint": embedding_fingerprint,
                "manifest": manifest, **fitted[winner]}
    model_path = Path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-", dir=model_path.parent) as temporary:
        pending = Path(temporary) / "model.joblib"
        joblib.dump(artifact, pending)
        pending.replace(model_path)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("Selected %s; saved %s. Preserving fitted model/calibrator/threshold together (no uncalibrated full-data refit).", winner, model_path)
    return manifest
