"""Generate benchmark dataset and train all 4 cases.

Produces one checkpoint per case in outputs/runs/{case}/best.pt,
ready for deployment benchmarking with scripts/export_trt.py.

Examples
--------
    # Full pipeline: data generation + train all cases
    uv run scripts/train_all_cases.py

    # Skip data generation (already done)
    uv run scripts/train_all_cases.py --skip-datagen

    # Train only specific cases
    uv run scripts/train_all_cases.py --cases logical_head mwpm_teacher

    # Use compiled backend for training
    uv run scripts/train_all_cases.py --backend compiled
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

from cli import setup_logging


logger = logging.getLogger(__name__)

CASES = ("logical_head", "mwpm_teacher", "hybrid", "tn_teacher")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_generation_bench.yaml"),
        help="Data generation config",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train_bench.yaml"),
        help="Training config (case is overridden per run)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASES),
        choices=CASES,
        help="Cases to train (default: all 4)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["pytorch", "compiled", "cuda"],
        help="Override training backend",
    )
    parser.add_argument(
        "--skip-datagen",
        action="store_true",
        help="Skip dataset generation (assumes data already exists)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data and checkpoints",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def generate_data(args: argparse.Namespace) -> None:
    """Run dataset generation for all requested cases."""
    from qec_generator import Config, generate_datasets, generate_raw_data

    cfg = Config.from_yaml(args.data_config)

    logger.info("Generating raw data (%d settings)", len(list(cfg.iter_settings())))
    generate_raw_data(cfg, overwrite=args.overwrite)

    # Generate datasets for all cases (tn_teacher needs cuquantum for soft labels)
    cases_to_generate = tuple(args.cases)
    logger.info("Building datasets for cases: %s", cases_to_generate)
    generate_datasets(cfg, cases=cases_to_generate, overwrite=args.overwrite)


def train_case(case: str, args: argparse.Namespace) -> Path:
    """Train a single case and return the best checkpoint path."""
    from gnn.trainer import TrainConfig, Trainer

    cfg = TrainConfig.from_yaml(args.train_config)

    # Override case
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["case"] = case
    if args.backend is not None:
        cfg_dict["backend"] = args.backend

    cfg = TrainConfig(**cfg_dict)

    logger.info(
        "Training case=%s (hidden_dim=%d, layers=%d, epochs=%d, backend=%s)",
        case,
        cfg.hidden_dim,
        cfg.num_layers,
        cfg.epochs,
        cfg.backend,
    )

    t0 = time.perf_counter()
    trainer = Trainer(cfg)
    best_path = trainer.fit()
    elapsed = time.perf_counter() - t0

    logger.info("  → %s done in %.1fs, checkpoint: %s", case, elapsed, best_path)
    return best_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    # Step 1: Generate data
    if not args.skip_datagen:
        logger.info("=" * 60)
        logger.info("Step 1: Dataset generation")
        logger.info("=" * 60)
        try:
            generate_data(args)
        except Exception:
            logger.exception("Data generation failed")
            sys.exit(1)
    else:
        logger.info("Skipping data generation (--skip-datagen)")

    # Step 2: Train each case
    logger.info("=" * 60)
    logger.info("Step 2: Training %d cases", len(args.cases))
    logger.info("=" * 60)

    checkpoints = {}
    for case in args.cases:
        try:
            checkpoints[case] = train_case(case, args)
        except Exception:
            logger.exception("Training failed for case=%s", case)

    # Summary
    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info("=" * 60)
    for case, path in checkpoints.items():
        logger.info("  %s: %s", case, path)

    if len(checkpoints) < len(args.cases):
        failed = set(args.cases) - set(checkpoints)
        logger.warning("Failed cases: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
