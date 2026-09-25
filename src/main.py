"""CLI: python -m src.main {train,infer,run} --config config/default.json."""
import argparse
from dataclasses import replace
import logging
import random
import numpy as np
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Offline multi-source business entity resolution")
    parser.add_argument("command", choices=["train", "infer", "run"])
    parser.add_argument("--config")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = Config.load(args.config)
        overrides = {key: getattr(args, key) for key in ("dataset_dir", "output_dir", "model_path") if getattr(args, key) is not None}
        config = replace(config, **overrides)
        random.seed(config.seed)
        np.random.seed(config.seed)
        if config.backend == "disk":
            from .real_pipeline import prepare, train_real, audit_blocking, infer_real, official_validate
            from .quality import quality_report
            if args.command in {"train", "run"}:
                quality_report(config)
                prepare(config)
                train_real(config)
            if args.command == "run":
                audit_blocking(config)
            if args.command in {"infer", "run"}:
                infer_real(config)
                official_validate(config)
            if args.command == "run":
                from .real_report import generate
                generate(config)
            return
        if args.command == "run":
            from .data_loader import check_required_files
            check_required_files(config.dataset_dir, "train")
            check_required_files(config.dataset_dir, "test")
        if args.command in {"train", "run"}:
            from .training import train
            train(config)
        if args.command in {"infer", "run"}:
            from .inference import infer
            infer(config)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
        logging.getLogger(__name__).error("Pipeline failed: %s", exc, exc_info=args.verbose)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
