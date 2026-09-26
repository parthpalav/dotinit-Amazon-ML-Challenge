# Resume on the A5000 workstation

The laptop inference was deliberately stopped at **133,000 / 1,732,544 anchors**. All **1,330** completed score shards are contiguous, with **4,255,387** finite pair probabilities. Do not retrain or discard these checkpoints to resume.

## Transfer two things

1. Git code: pull `main` from `origin` after the transfer commit is pushed. An offline `transfer/amazon-ml-code.bundle` is also provided if GitHub authentication is unavailable.
2. Non-Git assets: copy `transfer/amazon-ml-assets.zip` **and** `transfer/amazon-ml-assets.sha256` to the workstation. The archive contains all datasets, all saved models, original outputs, feature/index caches, diagnostics and completed inference shards. Existing hard-linked copies are stored once and restored as hard links where supported. The virtual environment and platform-specific compiled libraries are excluded and rebuilt. Incomplete temporary files and empty SQLite journals are excluded.

Extract with the provided restore command, not Explorer/unzip: the manifest restores hard links and exact index timestamps required by posting-count cache checks. Restoration verifies archive and per-file SHA256 hashes. Only restore this project's trusted archive: model files contain pickled Python objects.

Allow approximately 25 GB for the restored files if hard links work, plus the archive and a new environment. Allow 45 GB if hard links are unavailable. Keep the checkout and assets on the local NVMe, not a network share.

## Environment

Use **Python 3.12** and a C++17 compiler (`g++` or `clang++`) available on PATH. This works on Windows or Linux; `config/windows.json` is a historical filename, not an OS restriction. The compiler rebuilds the tiny native libraries for the destination OS. CUDA toolkit installation is not required for this CPU inference path.

Windows PowerShell, from the checkout root:

```powershell
git pull --ff-only origin main
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-transfer.txt
.venv/Scripts/python.exe -m src.transfer_assets restore "D:/transfer/amazon-ml-assets.zip"
g++ --version
.venv/Scripts/python.exe -u -m src.complete_improvements --workers 6 --full-validator
```

Linux, from the checkout root:

```bash
git pull --ff-only origin main
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-transfer.txt
.venv/bin/python -m src.transfer_assets restore /path/to/amazon-ml-assets.zip
g++ --version
.venv/bin/python -u -m src.complete_improvements --workers 6 --full-validator
```

If using the offline Git bundle instead of GitHub, start with `git clone /path/to/amazon-ml-code.bundle amazon-ml` and `cd amazon-ml`, then skip the `git pull` command above.

Do not run two continuations at the same time. The command resumes the saved shards, scores the remainder, writes `outputs/improved_pair_threshold/` and `outputs/improved_unique_owner/`, and validates them. The independent pair-threshold policy is the one confirmed on holdout. Unique ownership is an additional, separately named variant, not a separately proven score improvement.

The supplied validator is included at `vendor/student_resource/utils/validate_submission.py`. Both candidates and matches receive exhaustive streaming validation. With `--full-validator`, the supplied official checker also validates both complete TSVs and all target IDs. This uses more RAM and is appropriate for the 128 GB workstation. Without the flag, the laptop-safe matching-only official check remains available. Completion requires `reports/improvements/completion_status.json` to say `complete`.

## Workstation estimate

Reported hardware: Xeon W-2265, 128 GB DDR4, SK hynix PC801 2 TB NVMe, RTX A5000 24 GB.

Budget **roughly 1–2.5 hours for remaining inference and validation after setup**, initially with six workers. This is an estimate, not a benchmark on that computer. Extra RAM should greatly reduce the laptop's page-read pressure, but the A5000 will remain mostly idle because feature extraction and prediction are CPU-based. The GPU was used for training, which is already complete.

For a measured ETA before the full run, process 5,000 new anchors (the first 133,000 are already cached):

```bash
python -u -m src.rescoring --model artifacts/improvements/catboost_d10_evidence.joblib --work work/improvements/test_scores_d10_direct --workers 6 --direct-provenance --limit 138000
```

Use the Python executable from the destination `.venv`. This benchmark's new checkpoints are reusable by the full command. If its printed processing time is T seconds, approximate remaining scoring time as **T × (1,732,544 − 138,000) / 5,000**, then allow time for two exports and official validation. The initial benchmark includes startup and cold-cache costs, so later speed may improve. Inspect `reports/improvements/full_rescoring.log` and `work/improvements/test_scores_d10_direct/progress.json` for live speed.

## Results and limits

Fresh untouched confirmation: **0.943215 macro F0.5**, original model **0.878801** on the same 5,000 anchors. Threshold frozen at **0.675**. Selection score was **0.943147**. Amazon's score remains the team's reported **0.842** until the improved submission is evaluated. France generalization is not established by the US/India holdout.

Baseline model and both original TSVs remain byte-for-byte unchanged. No paid service, Colab session, or remote job has been started. No inference will restart on the laptop as part of packaging.


## Created asset bundle
Archive size: **9.97 GB** (9,972,950,552 bytes). Unique restored data: **22.18 GB**; logical size with duplicated hard-link paths: **38.84 GB**. Archive SHA256: `a220f6ab74d6d9e89f8e89054f4a2e958b9a719476c25ab3de010c51bbea6029`. Includes 1,465 files and 25 hard-link aliases. The archive is deliberately ignored by Git.
