from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from src.blocking import Blocker
from src.config import Config
from src.data_loader import load_sources, load_truth, read_source
from src.evaluation import candidate_report, evaluate, fbeta
from src.features import FeatureEngineer
from src.inference import infer
from src.model import CalibratedMatcher
from src.normalization import normalize_address, normalize_business_name, extract_address_components
from src.preprocessing import preprocess
from src.submission import validate_submission
from src.threshold import optimize_threshold
from src.training import split_source1, train
from utils.make_demo_data import make_demo


def records(rows):
    return preprocess(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))


@pytest.mark.parametrize("raw,expected", [
    ("ABC Corporation Pvt. Ltd.", "abc corporation"),
    ("Tata Consultancy Services Limited", "tata consultancy services"),
    ("McDonald's, Inc.", "mcdonalds"), (None, ""), (np.nan, ""),
    ("A & B Intl.", "a and b international"), ("ＡＣＭＥ LLC", "acme"),
    ("株式会社 東京", "株式会社 東京"), ("Limited Edition", "limited edition"),
])
def test_name_normalization(raw, expected):
    assert normalize_business_name(raw) == expected


def test_address_hints_and_missing():
    assert normalize_address("12 Main Road, Apt. 2") == "12 main rd apt 2"
    hints = extract_address_components("12 Sample Road, Example City, Region 12345")
    assert hints["postal_code"] == "12345"
    assert hints["city"] == "example city"
    assert hints["state"] == "region"
    assert hints["street_number"] == "12"
    assert extract_address_components("")["postal_code"] == ""
    assert extract_address_components("12 High Street, Town SW1A 1AA")["postal_code"] == "sw1a1aa"


def test_blocking_union_country_recovery_and_no_cartesian():
    anchors = records([["S1-a", "McDonalds", "1 Oak Road, Town, State 12345", "Unseen"],
                       ["S1-b", "", "", "Unseen"]])
    targets = records([["S2-a", "Mc Donald", "9 Pine St", "Different"],
                       ["S3-a", "McDonalds", "1 Oak Rd, Town, State 12345", "Unseen"],
                       ["S3-b", "Unrelated", "", "Unseen"]])
    blocker = Blocker(targets, Config())
    pairs = blocker.generate(anchors)
    assert set(pairs.candidate_entity_id) == {"S2-a", "S3-a"}
    assert pairs.source1_entity_id.eq("S1-a").all()
    exact_rules = set(pairs.loc[pairs.candidate_entity_id.eq("S3-a"), "blocking_rules"].iloc[0].split("|"))
    assert {"country_name", "city_name", "postal", "name_ngram", "address", "strong_name"} <= exact_rules
    assert not pairs.duplicated(["source1_entity_id", "candidate_entity_id"]).any()
    report = candidate_report(pairs, {"S1-a": {"S3-a", "S3-b"}, "S1-b": set()}, len(targets))
    assert report["candidate_recall"] == 0.5
    assert report["zero_candidate_fraction"] == 0.5


def test_missing_features_are_not_false_exact_matches():
    frame = records([["S1-a", "", "", ""], ["S2-a", "", "", ""]])
    pairs = pd.DataFrame([["S1-a", "S2-a", "postal"]], columns=["source1_entity_id", "candidate_entity_id", "blocking_rules"])
    engineer = FeatureEngineer().fit(frame)
    features = engineer.transform(pairs, engineer.prepare(frame))
    assert np.isfinite(features.to_numpy()).all()
    assert features.name_normalized_exact.iloc[0] == 0
    assert features.address_tfidf_cosine.iloc[0] == 0
    assert features.country_match.iloc[0] == 0
    assert features.postal_code_match.iloc[0] == 0


def test_entity_metric_penalizes_misses_and_false_singleton_merges():
    truth = {"S1-a": {"S2-a", "S3-a"}, "S1-b": set(), "S1-c": {"S2-c"}}
    pred = {"S1-a": {"S2-a"}, "S1-b": set(), "S1-c": set()}
    scores = evaluate(truth, pred, 5)
    assert scores["f0.5"] == pytest.approx((fbeta(1, 0.5) + 1) / 3)
    assert scores["singleton_accuracy"] == 1
    assert scores["false_negative"] == 2
    pred["S1-b"] = {"S3-x"}
    assert evaluate(truth, pred, 5)["singleton_accuracy"] == 0
    assert evaluate(truth, pred, 5)["false_positive_rate"] == pytest.approx(1 / 12)


def test_threshold_can_reject_all_even_probability_one():
    pairs = pd.DataFrame([["S1-a", "S2-a", "strong_name"]], columns=["source1_entity_id", "candidate_entity_id", "blocking_rules"])
    threshold, table = optimize_threshold(pairs, [1.0], {"S1-a": set()}, 1)
    assert threshold > 1.0
    assert table["f0.5"].max() == 1


def test_open_set_country_literal_na_and_id_validation(tmp_path):
    path = tmp_path / "source.tsv"
    pd.DataFrame([["S1-01", "Northwind", "", "NA"]], columns=["entity_id", "business_name", "business_address", "country"]).to_csv(path, sep="\t", index=False)
    assert read_source(path, 1).country.iloc[0] == "NA"
    with pytest.raises(ValueError, match="S2-"):
        read_source(path, 2)
    with pytest.raises(FileNotFoundError, match="Missing required"):
        load_sources(tmp_path, "train")


def test_truth_requires_explicit_singletons(tmp_path):
    make_demo(tmp_path, 20)
    anchors, targets = load_sources(tmp_path, "train")
    path = tmp_path / "train/train_ground_truth.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    frame.iloc[1:].to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="explicitly cover"):
        load_truth(tmp_path, anchors, targets)


def test_disjoint_reproducible_source1_splits(tmp_path):
    make_demo(tmp_path, 40)
    anchors, targets = load_sources(tmp_path, "train")
    truth = load_truth(tmp_path, anchors, targets)
    first = split_source1(anchors, truth, Config())
    second = split_source1(anchors, truth, Config())
    sets = [set(frame.entity_id) for frame in first.values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    assert set.union(*sets) == set(anchors.entity_id)
    assert all(first[name].equals(second[name]) for name in first)


@pytest.mark.parametrize("bad", ["S1-a", "S2-missing", "S2-a,S2-a"])
def test_validator_rejects_illegal_matches(tmp_path, bad):
    anchors = records([["S1-a", "Example", "", ""]])
    targets = records([["S2-a", "Example", "", ""]])
    matching, candidate = tmp_path / "matches.tsv", tmp_path / "candidates.tsv"
    matching.write_text(f"source1_entity_id\tmatched_entity_ids\nS1-a\t{bad}\n")
    candidate.write_text("source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-a\n")
    with pytest.raises(ValueError):
        validate_submission(matching, candidate, anchors, targets)


def test_validator_rejects_match_outside_candidates(tmp_path):
    matching, candidate = tmp_path / "matches.tsv", tmp_path / "candidates.tsv"
    matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a\n")
    candidate.write_text("source1_entity_id\tcandidate_entity_ids\nS1-a\t\n")
    with pytest.raises(ValueError, match="absent"):
        validate_submission(matching, candidate, records([["S1-a", "", "", ""]]), records([["S2-a", "", "", ""]]))


def test_embedding_permission_and_unknown_config(tmp_path):
    with pytest.raises(ValueError, match="permission"):
        Config(embedding_model_dir="somewhere", embedding_license="MIT")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"misspelled_option": True}))
    with pytest.raises(ValueError, match="Unknown"):
        Config.load(str(path))


def test_full_training_inference_and_exact_scored_candidate_export(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    make_demo(root, 40)
    config = Config(dataset_dir=str(root), output_dir=str(tmp_path / "output"),
                    model_path=str(tmp_path / "model.joblib"), trees=8, batch_size=7)
    manifest = train(config)
    table = pd.read_csv(tmp_path / "output/model_comparison.tsv", sep="\t")
    assert set(table.loc[table.status.eq("evaluated"), "model"]) == {"exact_rule", "logistic_regression", "random_forest", "xgboost"}
    assert manifest["candidate_metrics"]["validation"]["candidate_recall"] == 1
    scored_count = []
    original = CalibratedMatcher.predict

    def observe(self, features):
        scored_count.append(len(features))
        return original(self, features)

    monkeypatch.setattr(CalibratedMatcher, "predict", observe)
    report = infer(config)
    assert report["source1_rows"] == 40
    assert report["candidate_pairs"] == sum(scored_count)
    test_anchors, test_targets = (preprocess(x) for x in load_sources(root, "test"))
    expected = Blocker(test_targets, config).generate(test_anchors)
    exported = pd.read_csv(tmp_path / "output/candidate_pairs.tsv", sep="\t", keep_default_na=False)
    actual = {(row.source1_entity_id, target) for row in exported.itertuples(index=False)
              for target in row.candidate_entity_ids.split(",") if target}
    assert actual == set(zip(expected.source1_entity_id, expected.candidate_entity_id))
    second_output = tmp_path / "second"
    infer(replace(config, output_dir=str(second_output)))
    assert (tmp_path / "output/matching_results.tsv").read_bytes() == (second_output / "matching_results.tsv").read_bytes()
    # Empty target tables are valid at inference: every S1 must still be exported.
    for source in (2, 3):
        pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
            root / "test" / f"test_source{source}.tsv", sep="\t", index=False)
    empty_report = infer(replace(config, output_dir=str(tmp_path / "empty_output")))
    assert empty_report == {"source1_rows": 40, "candidate_pairs": 0, "matches": 0, "singletons": 40}
