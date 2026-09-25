"""Submission schema checks shared by inference and the standalone validator."""
import pandas as pd
from .data_loader import parse_ids


def validate_submission(matching_path, candidate_path, anchors, targets):
    matching = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    candidates = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    expected = set(anchors.entity_id)
    known = set(targets.entity_id)
    parsed = []
    for frame, column in ((matching, "matched_entity_ids"), (candidates, "candidate_entity_ids")):
        if frame.columns.tolist() != ["source1_entity_id", column]:
            raise ValueError(f"Expected exactly source1_entity_id and {column}")
        if frame.source1_entity_id.duplicated().any() or set(frame.source1_entity_id) != expected:
            raise ValueError(f"{column}: expected exactly one row for every test S1")
        rows = {}
        for source, cell in zip(frame.source1_entity_id, frame[column]):
            ids = parse_ids(cell)
            if cell and cell != ",".join(ids):
                raise ValueError(f"{column}: expected canonical comma-separated IDs")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{source}: duplicate {column}")
            if any(not x.startswith(("S2-", "S3-")) for x in ids) or not set(ids).issubset(known):
                raise ValueError(f"{source}: invalid or nonexistent {column}")
            rows[source] = set(ids)
        parsed.append(rows)
    matches, candidate_sets = parsed
    for source in expected:
        if not matches[source].issubset(candidate_sets[source]):
            raise ValueError(f"{source}: final match absent from candidate list")
    return {"source1_rows": len(expected), "candidate_pairs": sum(map(len, candidate_sets.values())),
            "matches": sum(map(len, matches.values())), "singletons": sum(not x for x in matches.values())}
