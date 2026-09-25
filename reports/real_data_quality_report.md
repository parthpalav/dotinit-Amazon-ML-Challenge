# Real data quality report

All supplied source files use the same four-field UTF-8 TSV schema. This is one train triplet and one test triplet, not three independent prediction tasks. Raw inputs are unchanged.

| File | Rows | Unique IDs | Duplicate IDs | Missing names | Missing addresses | Countries |
|---|---:|---:|---:|---:|---:|---|
| dataset/test/test_source1.tsv | 1,732,544 | 1,732,544 | 0 | 0 | 0 | {'India': 809986, 'US': 663106, 'France': 259452} |
| dataset/test/test_source2.tsv | 4,887,273 | 4,887,273 | 0 | 0 | 129408 | {'India': 2312565, 'US': 1871330, 'France': 703378} |
| dataset/test/test_source3.tsv | 5,082,316 | 5,082,316 | 0 | 0 | 136098 | {'India': 2405000, 'US': 1945701, 'France': 731615} |
| dataset/train/train_ground_truth.tsv | 2,206,821 | 2,206,821 | 0 | N/A | N/A | {} |
| dataset/train/train_source1.tsv | 2,206,821 | 2,206,821 | 0 | 0 | 0 | {'US': 1323633, 'India': 883188} |
| dataset/train/train_source2.tsv | 5,034,616 | 5,034,616 | 0 | 0 | 168967 | {'US': 3016817, 'India': 2017799} |
| dataset/train/train_source3.tsv | 5,285,603 | 5,285,603 | 0 | 0 | 175916 | {'US': 3170056, 'India': 2115547} |

No dedicated phone, email, URL, city, state or postal fields exist. Contact strings sometimes occur inside names; only explicit hints are extracted. Address numbers are not treated as phones. Contact validity and geographic consistency cannot be externally verified under the challenge rules.

The JSON reports near-empty fields, non-ASCII text, casing/punctuation, embedded contact hints, exact raw-business fingerprint repeats, singleton/positive distributions and train/test ID overlap. Fingerprint repeats are possible duplicate records, not proof of entity identity; ID uniqueness is exact for the observed numeric-suffix syntax.

Ground-truth pair duplication, conflicting ownership, source coverage and target existence are enforced by the disk-store build and abort on failure. Final results include these checks.

The normalizer now preserves Indic combining marks. City/state and postal extraction are uncertain when address components are reordered; no structured address field is mandatory for retrieval. Missing strings never become false exact-match evidence.

Every resource file, including documentation, official validator and filesystem metadata, is included in the SHA-256 inventory in real_data_quality.json.

Cross-source exact raw-business fingerprint intersections are recorded in `cross_source_business_fingerprints.json`. These count distinct shared raw name/address/country fingerprints; they are descriptive, not ground-truth entity labels.
