"""Serializable configuration; relative paths resolve from the working directory."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    dataset_dir: str = "dataset"
    output_dir: str = "output"
    model_path: str = "artifacts/model.joblib"
    seed: int = 42
    validation_fraction: float = 0.20
    calibration_fraction: float = 0.20
    metric: str = "entity_macro"
    max_posting: int = 5000
    max_document_frequency: float = 0.20
    common_token_floor: int = 20
    min_ngram_overlap: int = 2
    min_ngram_jaccard: float = 0.12
    strong_name_ratio: float = 0.88
    address_min_overlap: int = 2
    batch_size: int = 20000
    tfidf_max_features: int = 100000
    trees: int = 300
    threads: int = 1
    minimum_candidate_recall: float = 0.98
    fail_on_low_candidate_recall: bool = False
    embedding_model_dir: str | None = None
    embedding_license: str | None = None
    embedding_constraints_confirmed: bool = False
    backend: str = "memory"
    working_dir: str = "work/real"
    reports_dir: str = "reports"
    resource_dir: str | None = None
    io_chunk_size: int = 50000
    retrieval_posting_limit: int = 120
    max_candidates_per_anchor: int = 32
    fit_anchor_limit: int = 30000
    calibration_anchor_limit: int = 5000
    validation_anchor_limit: int = 10000
    workers: int = 4
    anchor_batch_size: int = 500

    def __post_init__(self):
        if self.backend not in {"memory", "disk"}:
            raise ValueError("backend must be memory or disk")
        for name in ("io_chunk_size", "retrieval_posting_limit", "max_candidates_per_anchor",
                     "fit_anchor_limit", "calibration_anchor_limit", "validation_anchor_limit", "workers", "anchor_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not (0 < self.validation_fraction < 1 and 0 < self.calibration_fraction < 1
                and self.validation_fraction + self.calibration_fraction < 1):
            raise ValueError("Calibration + validation fractions must be between zero and one.")
        if self.metric not in {"entity_macro", "micro"}:
            raise ValueError("metric must be entity_macro or micro")
        for name in ("max_posting", "common_token_floor", "min_ngram_overlap",
                     "address_min_overlap", "batch_size", "tfidf_max_features", "trees", "threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_document_frequency", "min_ngram_jaccard", "strong_name_ratio",
                     "minimum_candidate_recall"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.embedding_model_dir and (self.embedding_license not in {"MIT", "Apache-2.0"}
                                        or not self.embedding_constraints_confirmed):
            raise ValueError("Local embeddings require an MIT/Apache-2.0 license declaration and challenge permission confirmation.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | None):
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text())
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)
