"""Optional local-only semantic features. Never download a model."""
from pathlib import Path
import hashlib
import os
import numpy as np


def model_fingerprint(directory: str) -> str:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Local embedding model not found: {root}")
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Embedding model directory is empty")
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class LocalEmbeddings:
    def __init__(self, directory: str, expected_fingerprint: str | None = None):
        self.fingerprint = model_fingerprint(directory)
        if expected_fingerprint and self.fingerprint != expected_fingerprint:
            raise ValueError("Local embedding model content differs from the training artifact")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install requirements-embeddings.txt to use optional local embeddings") from exc
        self.model = SentenceTransformer(str(Path(directory).resolve()), device="cpu",
                                         local_files_only=True, trust_remote_code=False)

    def prepare(self, records):
        positions = {entity: i for i, entity in enumerate(records.entity_id)}
        vectors = {}
        for field in ("name_norm", "address_norm"):
            values = records[field].tolist()
            vectors[field] = self.model.encode(values, batch_size=64, convert_to_numpy=True,
                                               normalize_embeddings=True, show_progress_bar=False)
            vectors[field][np.array([not x for x in values])] = 0
        return positions, vectors

    @staticmethod
    def add_features(features, pairs, cache):
        positions, vectors = cache
        left = [positions[x] for x in pairs.source1_entity_id]
        right = [positions[x] for x in pairs.candidate_entity_id]
        result = features.copy()
        for prefix, field in (("name", "name_norm"), ("address", "address_norm")):
            result[f"{prefix}_embedding_cosine"] = np.sum(vectors[field][left] * vectors[field][right], axis=1)
        return result
