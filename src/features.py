"""Pairwise numeric features with training-fitted sparse TF-IDF."""
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from rapidfuzz.fuzz import WRatio, token_set_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from .blocking import RULES
from .normalization import ngrams


def equal(a: str, b: str) -> float:
    return float(bool(a and b) and a == b)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def similarity(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(a, b) if a and b else 0.0


def length_ratio(a: str, b: str) -> float:
    return min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0


class FeatureEngineer:
    def __init__(self, max_features: int = 100000):
        self.max_features = max_features
        self.vectorizers = {}
        self.feature_names = None

    def fit(self, records: pd.DataFrame):
        for field in ("name_norm", "address_norm"):
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                         max_features=self.max_features, dtype=np.float32)
            try:
                vectorizer.fit(records[field])
            except ValueError as exc:
                if "empty vocabulary" not in str(exc):
                    raise
                vectorizer = None
            self.vectorizers[field] = vectorizer
        return self

    def prepare(self, records: pd.DataFrame):
        """Build split-local caches. Never persisted with the model artifact."""
        records = records.reset_index(drop=True)
        return {
            "rows": records.to_dict("records"),
            "positions": {entity: i for i, entity in enumerate(records.entity_id)},
            "matrices": {field: vectorizer.transform(records[field]) if vectorizer is not None else None
                         for field, vectorizer in self.vectorizers.items()},
        }

    def transform(self, pairs: pd.DataFrame, cache: dict) -> pd.DataFrame:
        if not self.vectorizers:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        features, left, right = [], [], []
        for pair in pairs.itertuples(index=False):
            li = cache["positions"][pair.source1_entity_id]
            ri = cache["positions"][pair.candidate_entity_id]
            left.append(li)
            right.append(ri)
            a, b = cache["rows"][li], cache["rows"][ri]
            row = {}
            for prefix, raw, normalized in (("name", "business_name", "name_norm"),
                                              ("address", "business_address", "address_norm")):
                x, y = a[normalized], b[normalized]
                xs, ys = set(x.split()), set(y.split())
                row.update({f"{prefix}_exact": equal(a[raw], b[raw]),
                            f"{prefix}_normalized_exact": equal(x, y),
                            f"{prefix}_levenshtein_similarity": similarity(x, y),
                            f"{prefix}_jaccard_similarity": jaccard(xs, ys),
                            f"{prefix}_token_overlap": len(xs & ys),
                            f"{prefix}_length_difference": abs(len(x) - len(y)),
                            f"{prefix}_length_ratio": length_ratio(x, y),
                            f"{prefix}_left_missing": float(not x),
                            f"{prefix}_right_missing": float(not y)})
            for n in (3, 4, 5):
                row[f"name_char_{n}gram_similarity"] = jaccard(ngrams(a["name_norm"], n), ngrams(b["name_norm"], n))
            row["address_char_ngram_similarity"] = jaccard(ngrams(a["address_norm"], 3), ngrams(b["address_norm"], 3))
            for component in ("postal_code", "city", "state", "street_number"):
                row[f"{component}_match"] = equal(a[component], b[component])
                row[f"{component}_both_present"] = float(bool(a[component] and b[component]))
            row["country_match"] = equal(a["country_norm"], b["country_norm"])
            row["country_both_present"] = float(bool(a["country_norm"] and b["country_norm"]))
            row["name_jaro_winkler"] = JaroWinkler.normalized_similarity(a["name_norm"], b["name_norm"]) if a["name_norm"] and b["name_norm"] else 0.0
            row["name_wratio"] = WRatio(a["name_norm"], b["name_norm"])/100 if a["name_norm"] and b["name_norm"] else 0.0
            row["address_token_set_similarity"] = token_set_ratio(a["address_norm"], b["address_norm"])/100 if a["address_norm"] and b["address_norm"] else 0.0
            for field in ("email", "email_domain", "website_domain", "phone"):
                row[f"{field}_match"] = equal(a.get(field, ""), b.get(field, ""))
                row[f"{field}_both_present"] = float(bool(a.get(field) and b.get(field)))
            row["phone_suffix_match"] = equal(a.get("phone", "")[-7:], b.get("phone", "")[-7:])
            row["name_address_combined_similarity"] = similarity(
                (a["name_norm"] + " " + a["address_norm"]).strip(),
                (b["name_norm"] + " " + b["address_norm"]).strip())
            rules = set(pair.blocking_rules.split("|"))
            for rule in RULES:
                row[f"blocked_by_{rule}"] = float(rule in rules)
            row["number_of_blocking_rules"] = len(rules)
            row["candidate_source3"] = float(pair.candidate_entity_id.startswith("S3-"))
            features.append(row)
        result = pd.DataFrame(features)
        for prefix, field in (("name", "name_norm"), ("address", "address_norm")):
            matrix = cache["matrices"][field]
            result[f"{prefix}_tfidf_cosine"] = (np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel()
                                                 if matrix is not None and left else np.zeros(len(pairs)))
        if self.feature_names is None and len(pairs):
            self.feature_names = result.columns.tolist()
        if self.feature_names is not None:
            result = result.reindex(columns=self.feature_names, fill_value=0)
        return result.astype(np.float32)
