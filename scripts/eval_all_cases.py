"""Evaluate all decoders: baselines + GNN models.

Runs the full 5-decoder evaluation matrix from the implementation plan:

  1. MWPM baseline     — static DEM weights, PyMatching
  2. BP+OSD baseline   — static DEM weights, CUDA-Q
  3. GNN direct        — direct/best.pt => threshold logits => observable
  4. GNN => MWPM       — edge/best.pt => learned weights => PyMatching
  5. GNN => BP+OSD     — edge/best.pt => learned weights => CUDA-Q

Each step is a subprocess so that GPU state is clean between runs.
Steps are skipped if their prerequisites are missing (e.g. no checkpoint).

Examples
--------
    # Full evaluation (after training)
    uv run scripts/eval_all_cases.py

    # Skip baselines (already computed)
    uv run scripts/eval_all_cases.py --skip-baselines

    # Only baselines (no GNN checkpoints needed)
    uv run scripts/eval_all_cases.py --only-baselines

    # Custom paths
    uv run scripts/eval_all_cases.py \
        --data-config configs/data_generation.yaml \
        --runs-dir outputs/runs \
        --results-dir outputs/results
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from cli import setup_logging


logger = logging.getLogger(__name__)


def _run(
    label: str,
    cmd: list[str],
    *,
    required: bool = True,
) -> bool:
    """Run a subprocess, log outcome, return True on success."""
    logger.info("-" * 60)
    logger.info("%s", label)
    logger.info("  %s", " ".join(cmd))
    logger.info("-" * 60)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        if required:
            logger.error("%s FAILED (exit %d)", label, result.returncode)
        else:
            logger.warning("%s FAILED (exit %d) — skipping", label, result.returncode)
        return False

    logger.info("%s — done", label)
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_generation.yaml"),
        help="Data generation config (for baseline evaluation).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Packaged datasets directory (for GNN evaluation).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/runs"),
        help="Directory containing trained model checkpoints.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory for evaluation result JSONs.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--only-baselines", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    py = sys.executable
    scripts = Path(__file__).parent
    results = args.results_dir
    results.mkdir(parents=True, exist_ok=True)
    verbose = ["-v"] if args.verbose else []

    mwpm_baseline = results / "mwpm_baseline.json"
    bp_osd_baseline = results / "bp_osd_baseline.json"

    if not args.skip_baselines:
        # 1. MWPM baseline
        _run(
            "1/5  MWPM baseline",
            [
                py,
                str(scripts / "eval_mwpm.py"),
                "-c",
                str(args.data_config),
                "--splits",
                args.split,
                "-o",
                str(mwpm_baseline),
                *verbose,
            ],
        )

        # 2. BP+OSD baseline
        _run(
            "2/5  BP+OSD baseline",
            [
                py,
                str(scripts / "eval_bp_osd.py"),
                "-c",
                str(args.data_config),
                "--splits",
                args.split,
                "-o",
                str(bp_osd_baseline),
                *verbose,
            ],
            required=False,
        )

    if args.only_baselines:
        logger.info("--only-baselines set, stopping here.")
        return

    direct_ckpt = args.runs_dir / "direct" / "best.pt"
    edge_ckpt = args.runs_dir / "edge" / "best.pt"

    # 3. GNN direct
    if direct_ckpt.is_file():
        _run(
            "3/5  GNN direct",
            [
                py,
                str(scripts / "eval_gnn.py"),
                "--checkpoint",
                str(direct_ckpt),
                "--datasets-dir",
                str(args.datasets_dir),
                "--split",
                args.split,
                "--baseline",
                str(mwpm_baseline),
                "-o",
                str(results / "gnn_direct.json"),
                *verbose,
            ],
        )
    else:
        logger.warning("3/5  GNN direct — SKIPPED (no checkpoint: %s)", direct_ckpt)

    # 4. GNN edge => MWPM
    if edge_ckpt.is_file():
        _run(
            "4/5  GNN edge => MWPM",
            [
                py,
                str(scripts / "eval_gnn.py"),
                "--checkpoint",
                str(edge_ckpt),
                "--datasets-dir",
                str(args.datasets_dir),
                "--split",
                args.split,
                "--decoder",
                "mwpm",
                "--baseline",
                str(mwpm_baseline),
                "-o",
                str(results / "gnn_edge_mwpm.json"),
                *verbose,
            ],
        )

        # 5. GNN edge => BP+OSD
        _run(
            "5/5  GNN edge => BP+OSD",
            [
                py,
                str(scripts / "eval_gnn.py"),
                "--checkpoint",
                str(edge_ckpt),
                "--datasets-dir",
                str(args.datasets_dir),
                "--split",
                args.split,
                "--decoder",
                "bp_osd",
                "--baseline",
                str(bp_osd_baseline),
                "-o",
                str(results / "gnn_edge_bp_osd.json"),
                *verbose,
            ],
            required=False,
        )
    else:
        logger.warning(
            "4/5  GNN edge => MWPM — SKIPPED (no checkpoint: %s)",
            edge_ckpt,
        )
        logger.warning(
            "5/5  GNN edge => BP+OSD — SKIPPED (no checkpoint: %s)",
            edge_ckpt,
        )

    produced = sorted(results.glob("*.json"))
    logger.info("=" * 60)
    logger.info("Evaluation complete. Result files:")
    for p in produced:
        logger.info("  %s", p)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
