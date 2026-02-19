"""
Evaluate a trained GNN decoder and compare with MWPM baseline.

Examples
--------
    # Evaluate on test split
    uv run scripts/eval_gnn.py --checkpoint outputs/runs/logical_head/best.pt

    # Compare with MWPM baseline
    uv run scripts/eval_gnn.py --checkpoint outputs/runs/logical_head/best.pt \\
        --baseline outputs/results/mwpm_baseline.json

    # Save JSON report to custom path
    uv run scripts/eval_gnn.py --checkpoint outputs/runs/logical_head/best.pt \\
        -o outputs/results/gnn_logical.json
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from gnn.eval import evaluate_all, print_report, save_report


def setup_logging(verbose: bool = False) -> None:
    """
    Configure root logging.

    Parameters
    ----------
    verbose : bool
        Enable ``DEBUG`` level if True, otherwise ``INFO``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (e.g. outputs/runs/logical_head/best.pt)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Root directory of packaged datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="MWPM baseline JSON report for comparison",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Shots per inference batch (default: 256)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save JSON report to this path (default: next to checkpoint)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for GNN evaluation."""
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    report = evaluate_all(
        checkpoint=args.checkpoint,
        datasets_dir=args.datasets_dir,
        split=args.split,
        baseline_path=args.baseline,
        batch_size=args.batch_size,
    )

    print_report(report)

    # Always save report: explicit path or default next to checkpoint
    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.parent / f"eval_{args.split}.json"
    save_report(report, output_path)


if __name__ == "__main__":
    main()
