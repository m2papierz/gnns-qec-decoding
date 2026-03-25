"""Generate benchmark and evaluation plots.

Loads evaluation result JSONs (up to 5 decoders) and an optional
inference benchmark JSON, then produces publication-ready figures:

1. **LER vs p** - all decoders, per distance.  Capped Wilson CI bands.
2. **LER scaling with d** - at a fixed reference error probability.
3. **LER vs latency (Pareto)** - per backend, GNN decoders.
4. **Throughput vs batch size** - by backend.
5. **Speedup bar chart** - compiled/cuda normalised to pytorch.

All plots are saved as PNG to an output directory.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, return empty dict on failure."""
    if not path.is_file():
        logger.warning("File not found: %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_results(
    results_dir: Path,
    filenames: Sequence[str] = (
        "mwpm_baseline.json",
        "bp_osd_baseline.json",
        "gnn_direct.json",
        "gnn_edge_mwpm.json",
        "gnn_edge_bp_osd.json",
    ),
) -> Dict[str, List[Dict[str, Any]]]:
    """Load all evaluation JSONs keyed by decoder label."""
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for fname in filenames:
        path = results_dir / fname
        data = _load_json(path)
        if not data:
            continue

        results = data.get("results", [])
        if not results:
            logger.warning("No results in %s", path)
            continue

        label = _label_from_filename(fname)
        all_results[label] = results
        logger.info("Loaded %d results from %s as '%s'", len(results), fname, label)

    return all_results


def _label_from_filename(fname: str) -> str:
    mapping = {
        "mwpm_baseline.json": "MWPM",
        "bp_osd_baseline.json": "BP+OSD",
        "gnn_direct.json": "GNN direct",
        "gnn_edge_mwpm.json": "GNN→MWPM",
        "gnn_edge_bp_osd.json": "GNN→BP+OSD",
    }
    return mapping.get(fname, fname.removesuffix(".json"))


def load_benchmark_results(path: Path) -> Dict[str, Any]:
    return _load_json(path)


def _wilson_ci(
    n_errors: int, n_shots: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Wilson score interval.  Returns (p_hat, lower, upper)."""
    if n_shots == 0:
        return 0.0, 0.0, 0.0

    p_hat = n_errors / n_shots
    denom = 1 + z * z / n_shots
    centre = p_hat + z * z / (2 * n_shots)
    spread = z * math.sqrt(
        p_hat * (1 - p_hat) / n_shots + z * z / (4 * n_shots * n_shots)
    )

    lo = max(0.0, (centre - spread) / denom)
    hi = min(1.0, (centre + spread) / denom)

    # Cap CI band width to 3x the point estimate (avoids absurd bands
    # when n_errors is very small).
    if p_hat > 0:
        lo = max(lo, p_hat / 4)
        hi = min(hi, p_hat * 4)

    return p_hat, lo, hi


_DECODER_STYLES: Dict[str, Dict[str, Any]] = {
    "MWPM": {"color": "#1f77b4", "marker": "o", "ls": "--"},
    "BP+OSD": {"color": "#ff7f0e", "marker": "s", "ls": "--"},
    "GNN direct": {"color": "#2ca02c", "marker": "^", "ls": "-"},
    "GNN→MWPM": {"color": "#d62728", "marker": "D", "ls": "-"},
    "GNN→BP+OSD": {"color": "#9467bd", "marker": "v", "ls": "-"},
}

_BACKEND_COLORS: Dict[str, str] = {
    "pytorch": "#1f77b4",
    "compiled": "#ff7f0e",
    "cuda": "#2ca02c",
    "tensorrt": "#d62728",
}


def _style(label: str) -> Dict[str, Any]:
    return _DECODER_STYLES.get(label, {"color": "gray", "marker": "x", "ls": "-"})


def plot_ler_vs_p(
    eval_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Plot 1: LER vs physical error probability, one subplot per distance."""
    import matplotlib.pyplot as plt

    distances = sorted({r["distance"] for rs in eval_results.values() for r in rs})
    if not distances:
        logger.warning("No data for LER vs p plot")
        return

    fig, axes = plt.subplots(
        1, len(distances), figsize=(5 * len(distances), 4.5), squeeze=False
    )

    for col, d in enumerate(distances):
        ax = axes[0, col]

        for label, results in eval_results.items():
            subset = sorted(
                [r for r in results if r["distance"] == d],
                key=lambda r: r["error_prob"],
            )
            if not subset:
                continue

            probs = [r["error_prob"] for r in subset]
            lers = [r["logical_error_rate"] for r in subset]
            lo = [_wilson_ci(r["num_errors"], r["num_shots"])[1] for r in subset]
            hi = [_wilson_ci(r["num_errors"], r["num_shots"])[2] for r in subset]

            sty = _style(label)
            ax.plot(
                probs,
                lers,
                label=label,
                marker=sty["marker"],
                color=sty["color"],
                ls=sty["ls"],
                markersize=5,
                linewidth=1.5,
            )
            ax.fill_between(probs, lo, hi, alpha=0.10, color=sty["color"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical error prob (p)")
        ax.set_ylabel("Logical error rate (LER)")
        ax.set_title(f"d = {d}")
        ax.grid(True, which="both", alpha=0.3)

    axes[0, -1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    _save_fig(fig, output_dir / "ler_vs_p")


def plot_ler_scaling_with_d(
    eval_results: Dict[str, List[Dict]],
    output_dir: Path,
    *,
    reference_p: float | None = None,
) -> None:
    """Plot 2: LER vs distance at a fixed error probability."""
    import matplotlib.pyplot as plt

    all_probs = sorted({r["error_prob"] for rs in eval_results.values() for r in rs})
    if not all_probs:
        logger.warning("No data for LER scaling plot")
        return

    if reference_p is None:
        reference_p = all_probs[len(all_probs) // 2]
    elif reference_p not in all_probs:
        reference_p = min(all_probs, key=lambda x: abs(x - reference_p))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for label, results in eval_results.items():
        subset = sorted(
            [r for r in results if r["error_prob"] == reference_p],
            key=lambda r: r["distance"],
        )
        if not subset:
            continue

        ds = [r["distance"] for r in subset]
        lers = [r["logical_error_rate"] for r in subset]
        sty = _style(label)
        ax.plot(
            ds,
            lers,
            label=label,
            marker=sty["marker"],
            color=sty["color"],
            ls=sty["ls"],
            markersize=6,
            linewidth=1.5,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Code distance (d)")
    ax.set_ylabel("Logical error rate (LER)")
    ax.set_title(f"LER scaling - p = {reference_p}")
    ax.set_xticks(sorted({r["distance"] for rs in eval_results.values() for r in rs}))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, output_dir / "ler_scaling_d")


def plot_ler_vs_latency(
    eval_results: Dict[str, List[Dict]],
    benchmark_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot 3: LER vs inference latency - one point per (decoder, backend)."""
    import matplotlib.pyplot as plt

    bench_results = benchmark_data.get("results", [])
    if not bench_results:
        logger.warning("No benchmark data for Pareto plot")
        return

    # Build latency lookup: (case, backend) => mean_ms at the largest
    # batch_size available (representative throughput regime).
    latency_map: Dict[tuple[str, str], float] = {}
    for br in bench_results:
        key = (br["case"], br["backend"])
        if key not in latency_map or br["batch_size"] > latency_map.get(
            key + ("_bs",), 0  # type: ignore[arg-type]
        ):
            latency_map[key] = br["mean_ms"]

    _DECODER_TO_CASE = {
        "GNN direct": "direct",
        "GNN => MWPM": "edge",
        "GNN => BP+OSD": "edge",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    # Collect all available backends from benchmark data.
    all_backends = sorted({br["backend"] for br in bench_results})

    for label, results in eval_results.items():
        case = _DECODER_TO_CASE.get(label)
        if case is None:
            continue  # baselines have no GNN inference

        mean_ler = np.mean([r["logical_error_rate"] for r in results])
        sty = _style(label)

        for backend in all_backends:
            latency = latency_map.get((case, backend))
            if latency is None:
                continue

            marker = sty["marker"]
            color = _BACKEND_COLORS.get(backend, sty["color"])
            ax.scatter(
                latency,
                mean_ler,
                marker=marker,
                color=color,
                s=80,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.annotate(
                f"{label}\n({backend})",
                (latency, mean_ler),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=6.5,
                alpha=0.8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Mean LER")
    ax.set_title("LER vs Latency")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_dir / "ler_vs_latency")


def plot_throughput_vs_batch(
    benchmark_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot 4: Throughput (graphs/s) vs batch size, grouped by backend."""
    import matplotlib.pyplot as plt

    bench_results = benchmark_data.get("results", [])
    if not bench_results:
        logger.warning("No benchmark data for throughput plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    groups: Dict[tuple[str, str], tuple[list, list]] = {}
    for br in bench_results:
        key = (br["case"], br["backend"])
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].append(br["batch_size"])
        groups[key][1].append(br["throughput_graphs_per_sec"])

    for (case, backend), (bs_list, tp_list) in sorted(groups.items()):
        order = np.argsort(bs_list)
        bs_sorted = np.array(bs_list)[order]
        tp_sorted = np.array(tp_list)[order]
        color = _BACKEND_COLORS.get(backend, "gray")
        marker = "o" if case == "direct" else "s"
        ax.plot(
            bs_sorted,
            tp_sorted,
            label=f"{case}/{backend}",
            marker=marker,
            color=color,
            linewidth=1.5,
            markersize=5,
        )

    ax.set_xlabel("Batch size (graphs)")
    ax.set_ylabel("Throughput (graphs/s)")
    ax.set_title("Inference Throughput vs Batch Size")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    _save_fig(fig, output_dir / "throughput_vs_batch")


def plot_speedup_bar(
    benchmark_data: Dict[str, Any],
    output_dir: Path,
    *,
    reference_backend: str = "pytorch",
    reference_batch_size: int = 64,
) -> None:
    """Plot 5: Speedup bar chart - other backends normalised to pytorch."""
    import matplotlib.pyplot as plt

    bench_results = benchmark_data.get("results", [])
    if not bench_results:
        logger.warning("No benchmark data for speedup plot")
        return

    ref_latency: Dict[str, float] = {}
    for br in bench_results:
        if (
            br["backend"] == reference_backend
            and br["batch_size"] == reference_batch_size
        ):
            ref_latency[br["case"]] = br["mean_ms"]

    if not ref_latency:
        logger.warning(
            "No reference data for backend=%s bs=%d",
            reference_backend,
            reference_batch_size,
        )
        return

    cases = sorted(ref_latency.keys())
    backends = sorted({br["backend"] for br in bench_results} - {reference_backend})

    if not backends:
        logger.warning("Only one backend found, cannot compute speedup")
        return

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(cases) * len(backends)), 4))
    x = np.arange(len(cases))
    width = 0.8 / max(len(backends), 1)

    for i, backend in enumerate(backends):
        speedups = []
        for case in cases:
            match = [
                br
                for br in bench_results
                if br["case"] == case
                and br["backend"] == backend
                and br["batch_size"] == reference_batch_size
            ]
            if match and case in ref_latency and match[0]["mean_ms"] > 0:
                speedups.append(ref_latency[case] / match[0]["mean_ms"])
            else:
                speedups.append(0.0)

        offset = (i - (len(backends) - 1) / 2) * width
        color = _BACKEND_COLORS.get(backend, "gray")
        bars = ax.bar(
            x + offset,
            speedups,
            width,
            label=backend,
            color=color,
            alpha=0.85,
        )

        for bar, spd in zip(bars, speedups):
            if spd > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{spd:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.axhline(
        y=1.0,
        color="gray",
        ls="--",
        alpha=0.5,
        label=f"{reference_backend} (1x)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel(f"Speedup vs {reference_backend}")
    ax.set_title(f"Backend Speedup (batch_size={reference_batch_size})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_dir / "speedup_bar")


def _save_fig(fig: Any, stem: Path) -> None:
    """Save a matplotlib figure as PNG."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved %s", path)
    import matplotlib.pyplot as plt

    plt.close(fig)


def generate_all_plots(
    results_dir: Path,
    benchmark_path: Path | None,
    output_dir: Path,
    *,
    reference_p: float | None = None,
) -> None:
    """Generate all plots from evaluation and benchmark data."""

    import matplotlib

    matplotlib.use("Agg")

    eval_data = load_eval_results(results_dir)
    if not eval_data:
        logger.error("No evaluation results found in %s", results_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ler_vs_p(eval_data, output_dir)
    plot_ler_scaling_with_d(eval_data, output_dir, reference_p=reference_p)

    bench_data: Dict[str, Any] = {}
    if benchmark_path is not None:
        bench_data = load_benchmark_results(benchmark_path)

    if bench_data.get("results"):
        plot_ler_vs_latency(eval_data, bench_data, output_dir)
        plot_throughput_vs_batch(bench_data, output_dir)
        plot_speedup_bar(bench_data, output_dir)
    else:
        logger.info("No benchmark data - skipping latency/throughput/speedup plots")

    logger.info("All plots saved to %s", output_dir)
