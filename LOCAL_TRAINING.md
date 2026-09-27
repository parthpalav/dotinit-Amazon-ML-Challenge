# Local training with the available assets

From PowerShell in the project root:

```powershell
.\utils\train_local.ps1
```

The launcher creates `.venv`, installs the dependencies, then trains CatBoost on
the GPU. Package installation uses the package registry; training makes no API
calls and downloads no data or pretrained weights. PyTorch is not required for
this CatBoost path. A compatible NVIDIA driver is required for GPU execution.
If Python is elsewhere, pass `-Python 'C:/path/to/python.exe'`. CPU execution is
available explicitly with `-Device CPU`.

The trainer reads the existing `work/real_v1/features/{fit,calibration,validation}.joblib`
caches and `work/real_v1/train/store_manifest.json`. It preserves their Source 1
splits, fits on fit, uses calibration for early stopping and sigmoid calibration,
and chooses the threshold by entity-macro F0.5 on validation. These selection
metrics are not an unbiased test score. It does not run audits or rebuild indexes.

Results are saved separately in `artifacts/local_training/`: `model.joblib`,
`training.json`, `thresholds.tsv`, and `validation_probabilities.npy`. Existing
output directories are refused; pass `-Output artifacts/local_training_v2` for
another run. A failed run may leave its output directory; use a new output path.

This is a new model using the original cached features. It does not reproduce
the missing campaign ensemble, Hungarian raw-text features, or MiniLM training.
Its artifact is a feature matcher, not a complete `real_pipeline infer` bundle.
Test inference integration and final submission validation remain separate work.

The original campaign commands still require their original campaign assets;
code cannot recreate their learned models from filenames. Neural preparation
now fails explicitly when local weights are absent, instead of downloading them.

## Enriched model and contextual reranker

The stronger local path reconstructs additional text features from the existing
training SQLite store. Frequency features fit only on fitting anchors. Feature
shards are resumable and reject changed candidate rows, database identity, or
feature code. These commands do not run audits or download model weights:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local-training.txt
.\.venv\Scripts\python.exe -u -m src.train_local --enriched --depth 10 --iterations 4000 --output artifacts/local_training_enriched_d10
# Run only after the preceding command succeeds:
.\.venv\Scripts\python.exe -u -m src.train_local_reranker
```

The base model adds normalized token coverage, number conflicts, transliterated
and phonetic comparisons, legal-suffix distinctions, and one-to-one raw-token
alignment. The reranker adds candidate-group context and candidate-to-candidate
text agreement. It uses raw base probabilities (not the base sigmoid already
fitted on calibration labels). Original calibration anchors are split 60/20/20
for reranker fitting, early stopping and sigmoid calibration. The base's early
stopping did use the original calibration set, so these are not fully independent
from base selection. Validation anchors are never used for gradient fitting.

The reranker saves its base model and feature recipe inside
`artifacts/local_training_context/model.joblib`. Both artifacts still require
their corresponding feature computation for test inference; neither can be
passed directly to the old `real_pipeline infer` command. Local selection gains
do not establish improvements on the official leaderboard or on unlabeled France
records. Older submitted models and TSVs remain separate.

Measured results on the same existing 10,000-anchor validation selection set:

| Local model | Macro F0.5 | False matches |
| --- | ---: | ---: |
| Original cached features, depth 10 | 0.900982 | 1,723 |
| Enriched text features, depth 10 | 0.962279 | 439 |
| Enriched base + candidate-context reranker | 0.964502 | 321 |

The selected local artifact is `artifacts/local_training_context/model.joblib`,
with threshold `0.7250000000000003`. The comparison is recorded in
`reports/local_enriched_comparison.json`; installed versions are recorded in
`reports/local_enriched_environment.txt`. No official test score or new submission
is claimed. The local result exceeds 0.95 but does not reach 0.98.
