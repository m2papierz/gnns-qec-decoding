"""Export a trained GNN decoder with TensorRT and benchmark.

Loads a training checkpoint, compiles it with each requested
backend, runs timed benchmarks, and saves a JSON report.

Examples
--------
    # Benchmark pytorch vs compiled vs tensorrt
    uv run scripts/export_trt.py \\
        --checkpoint outputs/runs/direct/best.pt

    # Only pytorch and compiled (no torch-tensorrt needed)
    uv run scripts/export_trt.py \\
        --checkpoint outputs/runs/edge/best.pt \\
        --backends pytorch compiled

    # Custom batch size and iterations
    uv run scripts/export_trt.py \\
        --checkpoint outputs/runs/direct/best.pt \\
        --n-graphs 8 --n-iters 200
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (best.pt)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["pytorch", "compiled", "tensorrt"],
        choices=["pytorch", "compiled", "tensorrt"],
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16"],
        help="TensorRT precision",
    )
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=4,
        help="Number of graphs in synthetic benchmark batch",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=50,
        help="Nodes per graph in synthetic batch",
    )
    parser.add_argument(
        "--n-edges",
        type=int,
        default=120,
        help="Directed edges per graph in synthetic batch",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Warmup iterations before benchmark",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=100,
        help="Timed benchmark iterations",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path for JSON report (default: next to checkpoint)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    import torch

    from inference import (
        InferenceEngine,
        load_model_from_checkpoint,
        make_synthetic_batch,
    )

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model_from_checkpoint(
        args.checkpoint,
        device=device,
    )

    batch = make_synthetic_batch(
        n_graphs=args.n_graphs,
        n_nodes_per_graph=args.n_nodes,
        n_edges_per_graph=args.n_edges,
        device=device,
    )

    results = []
    for backend_name in args.backends:
        logger.info("Benchmarking backend: %s", backend_name)
        try:
            engine = InferenceEngine(
                model,
                backend=backend_name,
                device=device,
                precision=args.precision,
                warmup_iters=args.warmup_iters,
            )
            metrics = engine.benchmark(batch, n_iters=args.n_iters)
            results.append(metrics)

            logger.info(
                "  %s: %.2f ms/batch (%.0f graphs/s)",
                backend_name,
                metrics["mean_ms"],
                metrics["throughput_graphs_per_sec"],
            )
        except Exception:
            logger.exception("  %s: FAILED", backend_name)

    report = {
        "checkpoint": str(args.checkpoint),
        "config": cfg,
        "batch": {
            "n_graphs": args.n_graphs,
            "n_nodes_per_graph": args.n_nodes,
            "n_edges_per_graph": args.n_edges,
        },
        "device": device,
        "results": results,
    }

    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.parent / "benchmark_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
