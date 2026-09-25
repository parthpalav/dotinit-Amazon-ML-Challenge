"""Strict TSV ingestion and ground-truth validation."""
import json
import logging
from pathlib import Path
import pandas as pd

LOG = logging.getLogger(__name__)
REQUIRED = ("entity_id", "business_name", "business_address", "country")


def read_source(path: Path, source: int) -> pd.DataFrame:
    # Preserve literal IDs/country values such as "NA" and numeric-looking text.
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for col in REQUIRED:
        frame[col] = frame[col].fillna("").str.strip()
    ids = frame.entity_id
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"{path}: blank or duplicate entity IDs")
    if not ids.str.startswith(f"S{source}-").all() or ids.str.len().le(3).any():
        raise ValueError(f"{path}: expected nonempty S{source}- IDs")
    if ids.str.contains(r"[,\s]", regex=True).any():
        raise ValueError(f"{path}: IDs cannot contain commas or whitespace")
    LOG.info("%s shape=%s columns=%s missing=%s unique_ids=%d countries=%d",
             path, frame.shape, frame.columns.tolist(),
             frame.replace("", pd.NA).isna().sum().to_dict(), ids.nunique(),
             frame.country.replace("", pd.NA).nunique())
    return frame


def check_required_files(root: str | Path, split: str) -> list[Path]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    paths = [Path(root) / split / f"{split}_source{i}.tsv" for i in (1, 2, 3)]
    required = paths + ([Path(root) / split / "train_ground_truth.tsv"] if split == "train" else [])
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    return paths


def load_sources(root: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = check_required_files(root, split)
    frames = [read_source(p, i) for i, p in enumerate(paths, 1)]
    if frames[0].empty:
        raise ValueError("Source 1 is empty")
    return frames[0], pd.concat(frames[1:], ignore_index=True)


def parse_ids(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        result = json.loads(value)
        if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
            raise ValueError("Expected a JSON string list")
        return [x.strip() for x in result]
    return [x.strip() for x in value.split(",")]


def load_truth(root: str | Path, anchors: pd.DataFrame, targets: pd.DataFrame) -> dict[str, set[str]]:
    path = Path(root) / "train/train_ground_truth.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "matched_entity_ids"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: expected {sorted(required)}")
    frame.source1_entity_id = frame.source1_entity_id.str.strip()
    if frame.source1_entity_id.duplicated().any():
        raise ValueError("Ground truth contains repeated Source 1 IDs")
    if set(frame.source1_entity_id) != set(anchors.entity_id):
        raise ValueError("Ground truth must explicitly cover every training S1, including empty singleton rows")
    known = set(targets.entity_id)
    truth, owners = {}, {}
    for row in frame.itertuples(index=False):
        ids = parse_ids(row.matched_entity_ids)
        if len(ids) != len(set(ids)) or not set(ids).issubset(known):
            raise ValueError(f"Invalid or duplicate matches for {row.source1_entity_id}")
        for target in ids:
            if target in owners:
                raise ValueError(f"{target} belongs to multiple deduplicated Source 1 entities")
            owners[target] = row.source1_entity_id
        truth[row.source1_entity_id] = set(ids)
    LOG.info("Ground truth shape=%s columns=%s missing=%s unique_ids=%d singletons=%d matches=%d",
             frame.shape, frame.columns.tolist(), frame.replace("", pd.NA).isna().sum().to_dict(),
             frame.source1_entity_id.nunique(), sum(not x for x in truth.values()),
             sum(map(len, truth.values())))
    return truth
