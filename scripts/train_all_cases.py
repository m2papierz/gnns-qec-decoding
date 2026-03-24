"""Generate dataset and train all 4 cases.

Runs data generation once, then invokes train_gnn.py for each case
as a subprocess so that logging, torch.compile caches, and GPU state
are clean between runs.

Examples
--------
    # Full pipeline: data generation + train all cases
    uv run scripts/train_all_cases.py

    # Skip data generation (already done)
    uv run scripts/train_all_cases.py --skip-datagen

    # Train only specific cases
    uv run scripts/train_all_cases.py --cases logical_head hybrid

    # Use compiled backend for training
    uv run scripts/train_all_cases.py --backend compiled
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from cli import setup_logging
from constants import CASES


logger = logging.getLogger(__name__)

_TRAIN_SCRIPT = Path(__file__).parent / "train_gnn.py"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_generation.yaml"),
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train.yaml"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASES),
        choices=CASES,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["pytorch", "compiled", "cuda"],
    )
    parser.add_argument("--skip-datagen", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def generate_data(args: argparse.Namespace) -> None:
    from qec_generator import Config, generate_datasets, generate_raw_data

    cfg = Config.from_yaml(args.data_config)
    generate_raw_data(cfg, overwrite=args.overwrite)
    generate_datasets(
        cfg,
        cases=tuple(args.cases),
        overwrite=args.overwrite,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    if not args.skip_datagen:
        logger.info("Generating datasets from %s", args.data_config)
        try:
            generate_data(args)
        except Exception:
            logger.exception("Data generation failed")
            sys.exit(1)

    for case in args.cases:
        cmd = [
            sys.executable,
            str(_TRAIN_SCRIPT),
            "-c",
            str(args.train_config),
            "--case",
            case,
        ]
        if args.backend is not None:
            cmd += ["--backend", args.backend]
        if args.verbose:
            cmd.append("-v")

        logger.info("=" * 60)
        logger.info("Training: %s", case)
        logger.info("=" * 60)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(
                "Training failed for case=%s (exit %d)", case, result.returncode
            )
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
