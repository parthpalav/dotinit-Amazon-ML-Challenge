"""Inverted-index retrieval: union every emitted pair, retaining provenance."""
from collections import Counter, defaultdict
import logging
import pandas as pd
from rapidfuzz.fuzz import ratio
from .config import Config
from .normalization import ngrams

LOG = logging.getLogger(__name__)
RULES = ("country_name", "city_name", "postal", "name_ngram", "address", "strong_name", "name_number", "address_number", "name_pair")
PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "blocking_rules"]
GENERIC_NAME = {"and", "the", "of", "company", "corporation", "group", "services", "international"}
GENERIC_ADDRESS = {"rd", "st", "ave", "ln", "blvd", "dr", "near", "main", "building", "apt", "ste", "and", "the"}


def name_tokens(row):
    return {t for t in row["name_norm"].split() if t not in GENERIC_NAME and len(t) > 1}


def address_tokens(row):
    return {t for t in row["address_norm"].split() if t not in GENERIC_ADDRESS and len(t) > 1}


def grams(row):
    return {(n, g) for n in (3, 4, 5) for g in ngrams(row["name_norm"], n)}


class Blocker:
    def __init__(self, targets: pd.DataFrame, config: Config):
        self.config = config
        self.targets = targets.to_dict("records")
        self.indices = {name: defaultdict(list) for name in
                        ("country_name", "city_name", "postal", "gram", "address", "exact_name")}
        self.target_grams = []
        for i, row in enumerate(self.targets):
            country = row["country_norm"]
            if country:
                for token in name_tokens(row):
                    self.indices["country_name"][(country, token)].append(i)
                    if row["city"]:
                        self.indices["city_name"][(country, row["city"], token)].append(i)
                if row["postal_code"]:
                    self.indices["postal"][(country, row["postal_code"])].append(i)
                for token in address_tokens(row):
                    self.indices["address"][(country, token)].append(i)
            row_grams = grams(row)
            self.target_grams.append(row_grams)
            for gram in row_grams:
                self.indices["gram"][gram].append(i)
            if row["name_norm"]:
                self.indices["exact_name"][row["name_norm"]].append(i)
        self.dropped_keys = {}
        generic_limit = min(config.max_posting, max(config.common_token_floor,
                           int(len(self.targets) * config.max_document_frequency)))
        for name, index in self.indices.items():
            limit = config.max_posting if name in {"postal", "exact_name", "city_name"} else generic_limit
            dropped = [key for key, values in index.items() if len(values) > limit]
            for key in dropped:
                del index[key]
            self.dropped_keys[name] = len(dropped)
        LOG.info("Blocking target_records=%d suppressed_keys=%s; posting suppression trades recall for cost",
                 len(self.targets), self.dropped_keys)

    def retrieve(self, row: dict) -> dict[int, set[str]]:
        found = defaultdict(set)
        country = row["country_norm"]
        if country:
            for token in sorted(name_tokens(row)):
                for i in self.indices["country_name"].get((country, token), ()):
                    found[i].add("country_name")
                if row["city"]:
                    for i in self.indices["city_name"].get((country, row["city"], token), ()):
                        found[i].add("city_name")
            if row["postal_code"]:
                for i in self.indices["postal"].get((country, row["postal_code"]), ()):
                    found[i].add("postal")
            counts = Counter(i for token in sorted(address_tokens(row))
                             for i in self.indices["address"].get((country, token), ()))
            for i, count in counts.items():
                if count >= self.config.address_min_overlap:
                    found[i].add("address")
        query_grams = grams(row)
        counts = Counter(i for gram in sorted(query_grams) for i in self.indices["gram"].get(gram, ()))
        # Name retrieval is country-independent to recover missing/inconsistent countries.
        for i, count in counts.items():
            target_grams = self.target_grams[i]
            intersection = len(query_grams & target_grams)
            union = len(query_grams | target_grams)
            if count >= self.config.min_ngram_overlap and intersection / max(1, union) >= self.config.min_ngram_jaccard:
                found[i].add("name_ngram")
            if count >= self.config.min_ngram_overlap and ratio(row["name_norm"], self.targets[i]["name_norm"]) / 100 >= self.config.strong_name_ratio:
                found[i].add("strong_name")
        for i in self.indices["exact_name"].get(row["name_norm"], ()):
            found[i].add("strong_name")
        return found

    def iter_pairs(self, anchors: pd.DataFrame):
        """Bounded pair batches for inference; no truncation of the retrieved union."""
        batch = []
        for row in anchors.to_dict("records"):
            found = self.retrieve(row)
            for i in sorted(found, key=lambda j: self.targets[j]["entity_id"]):
                batch.append((row["entity_id"], self.targets[i]["entity_id"], "|".join(sorted(found[i]))))
                if len(batch) >= self.config.batch_size:
                    yield pd.DataFrame(batch, columns=PAIR_COLUMNS)
                    batch = []
        if batch:
            yield pd.DataFrame(batch, columns=PAIR_COLUMNS)

    def generate(self, anchors: pd.DataFrame) -> pd.DataFrame:
        batches = list(self.iter_pairs(anchors))
        return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=PAIR_COLUMNS)
