"""Benchmark GNN decoder inference across backends and batch sizes.

Autodiscovers trained checkpoints and available backends (including
CUDA kernels if built), measures latency, throughput, and peak GPU
memory for each combination.

Examples
--------
    # Auto-detect all available backends
    uv run scripts/benchmark_all.py

    # Explicit backends
    uv run scripts/benchmark_all.py --backends pytorch compiled cuda

    # Custom batch sizes and more iterations
    uv run scripts/benchmark_all.py --batch-sizes 16 64 128 256 --n-iters 500
"""

import argparse
import logging
from pathlib import Path
from typing import Sequence

from benchmarks.runner import run_all


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/runs"),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/benchmark_report.json"),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=["pytorch", "compiled", "cuda", "tensorrt"],
        help="Backends to benchmark (default: auto-detect all available).",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 16, 64, 128],
    )
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    report = run_all(
        runs_dir=args.runs_dir,
        output_path=args.output,
        backends=args.backends,
        batch_sizes=args.batch_sizes,
        n_iters=args.n_iters,
        warmup_iters=args.warmup_iters,
    )
    logger.info(
        "Benchmark complete: %d measurements across %d checkpoints",
        len(report.results),
        len({r["checkpoint"] for r in report.results}),
    )


if __name__ == "__main__":
    main()
