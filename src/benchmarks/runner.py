"""Inference benchmark harness for GNN QEC decoders.

Autodiscovers trained checkpoints, benchmarks each across backends and
batch sizes, and writes a consolidated JSON report with hardware metadata.

The heavy lifting is delegated to :class:`deploy.engine.InferenceEngine`;
this module handles discovery, iteration, memory tracking, and reporting.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from constants import CASES
from deploy.engine import (
    InferenceEngine,
    load_model_from_checkpoint,
    make_synthetic_batch,
)


logger = logging.getLogger(__name__)

# Approximate detector-graph geometry for a d=5 rotated surface code
# memory experiment (~66 nodes, ~192 directed edges).  Used to build
# synthetic batches for benchmarking.
_DEFAULT_NODES: int = 66
_DEFAULT_EDGES: int = 192


@dataclass
class BenchmarkReport:
    """Full benchmark report with hardware metadata."""

    hardware: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Write report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"hardware": self.hardware, "results": self.results},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Benchmark report saved to %s", path)


def _collect_hardware_info() -> Dict[str, Any]:
    """Gather hardware and software metadata."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
        info["cuda_version"] = torch.version.cuda or "unknown"
    return info


def discover_checkpoints(runs_dir: Path) -> Dict[str, Path]:
    """Find best.pt checkpoints under runs_dir/{case}/.

    Returns
    -------
    dict
        Mapping from case name to checkpoint path.
    """
    found: Dict[str, Path] = {}
    for case in CASES:
        ckpt = runs_dir / case / "best.pt"
        if ckpt.is_file():
            found[case] = ckpt
            logger.info("Found checkpoint: %s", ckpt)
        else:
            logger.warning("No checkpoint for case '%s' at %s", case, ckpt)
    return found


def _peak_memory_mb() -> float:
    """Return peak CUDA memory allocated in MB (0.0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_checkpoint(
    checkpoint: Path,
    *,
    backends: Sequence[str] = ("pytorch", "compiled"),
    batch_sizes: Sequence[int] = (1, 16, 64, 128),
    n_iters: int = 100,
    warmup_iters: int = 10,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Benchmark a single checkpoint across backends and batch sizes.

    Parameters
    ----------
    checkpoint : Path
        Path to ``best.pt``.
    backends : sequence of str
        Backends to benchmark (``"pytorch"``, ``"compiled"``, ``"tensorrt"``).
    batch_sizes : sequence of int
        Number of graphs per batch.
    n_iters : int
        Timed iterations per measurement.
    warmup_iters : int
        Warmup passes before timing.
    device : str
        Target device.

    Returns
    -------
    list of dict
        One result dict per (backend, batch_size) combination.
    """
    model, cfg = load_model_from_checkpoint(checkpoint, device=device)
    case = cfg["case"]
    results: List[Dict[str, Any]] = []

    for backend in backends:
        for bs in batch_sizes:
            label = f"{case}/{backend}/bs={bs}"
            logger.info("Benchmarking %s ...", label)

            batch = make_synthetic_batch(
                n_graphs=bs,
                n_nodes_per_graph=_DEFAULT_NODES,
                n_edges_per_graph=_DEFAULT_EDGES,
                device=device,
            )

            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                engine = InferenceEngine(
                    model,
                    backend=backend,
                    device=device,
                    warmup_iters=warmup_iters,
                )
                metrics = engine.benchmark(batch, n_iters=n_iters)
                peak_mem = _peak_memory_mb()

                result = {
                    "checkpoint": str(checkpoint),
                    "case": case,
                    "backend": backend,
                    "batch_size": bs,
                    "n_iters": n_iters,
                    "mean_ms": round(metrics["mean_ms"], 4),
                    "std_ms": round(metrics["std_ms"], 4),
                    "median_ms": round(metrics["median_ms"], 4),
                    "min_ms": round(metrics["min_ms"], 4),
                    "max_ms": round(metrics["max_ms"], 4),
                    "throughput_graphs_per_sec": round(
                        metrics["throughput_graphs_per_sec"], 1
                    ),
                    "peak_memory_mb": round(peak_mem, 1),
                }
                results.append(result)

                logger.info(
                    "  %s: %.2f ms  (%.0f graphs/s, %.0f MB peak)",
                    label,
                    metrics["mean_ms"],
                    metrics["throughput_graphs_per_sec"],
                    peak_mem,
                )

            except Exception:
                logger.exception("  %s: FAILED", label)

    return results


def run_all(
    runs_dir: Path,
    output_path: Path,
    *,
    backends: Sequence[str] = ("pytorch", "compiled"),
    batch_sizes: Sequence[int] = (1, 16, 64, 128),
    n_iters: int = 100,
    warmup_iters: int = 10,
) -> BenchmarkReport:
    """Benchmark all discovered checkpoints and save report.

    Parameters
    ----------
    runs_dir : Path
        Directory containing ``{case}/best.pt`` checkpoints.
    output_path : Path
        Where to write ``benchmark_report.json``.
    backends, batch_sizes, n_iters, warmup_iters
        Forwarded to :func:`benchmark_checkpoint`.

    Returns
    -------
    BenchmarkReport
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = discover_checkpoints(runs_dir)

    if not checkpoints:
        logger.error("No checkpoints found in %s", runs_dir)
        return BenchmarkReport()

    report = BenchmarkReport(hardware=_collect_hardware_info())

    for _, ckpt in checkpoints.items():
        results = benchmark_checkpoint(
            ckpt,
            backends=backends,
            batch_sizes=batch_sizes,
            n_iters=n_iters,
            warmup_iters=warmup_iters,
            device=device,
        )
        report.results.extend(results)

    report.save(output_path)
    return report
