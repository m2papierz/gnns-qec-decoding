"""Generate benchmark and evaluation plots.

Loads evaluation result JSONs (up to 5 decoders) and an optional
inference benchmark JSON, then produces publication-ready figures:

1. **LER vs p** -- all decoders, per distance (averaged over rounds).
2. **LER scaling with d** -- at a fixed reference error probability.
3. **LER vs latency** -- all GNN decoders x backends.
4. **Throughput vs batch size** -- subplots per case.
5. **Speedup bar chart** -- compiled/cuda normalised to pytorch.

All plots saved as PNG.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
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
) -> dict[str, list[dict[str, Any]]]:
    all_results: dict[str, list[dict[str, Any]]] = {}
    for fname in filenames:
        data = _load_json(results_dir / fname)
        if not data:
            continue
        results = data.get("results", [])
        if results:
            label = _label_from_filename(fname)
            all_results[label] = results
            logger.info("Loaded %d results from %s as '%s'", len(results), fname, label)
    return all_results


def _label_from_filename(fname: str) -> str:
    return {
        "mwpm_baseline.json": "MWPM",
        "bp_osd_baseline.json": "BP+OSD",
        "gnn_direct.json": "GNN direct",
        "gnn_edge_mwpm.json": "GNN>MWPM",
        "gnn_edge_bp_osd.json": "GNN>BP+OSD",
    }.get(fname, fname.removesuffix(".json"))


def load_benchmark_results(path: Path) -> dict[str, Any]:
    return _load_json(path)


def _aggregate_over_rounds(
    results: list[dict],
) -> dict[tuple[int, float], tuple[float, int, int]]:
    """Average LER over rounds for each (distance, error_prob).

    Returns {(d, p): (mean_ler, total_errors, total_shots)}.
    """
    bucket: dict[tuple[int, float], tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in results:
        key = (r["distance"], r["error_prob"])
        errs, shots = bucket[key]
        bucket[key] = (errs + r["num_errors"], shots + r["num_shots"])

    return {k: (e / s if s > 0 else 0.0, e, s) for k, (e, s) in bucket.items()}


_STYLES: dict[str, dict[str, Any]] = {
    "MWPM": {"color": "#1f77b4", "marker": "o", "ls": "-", "zorder": 5},
    "BP+OSD": {"color": "#ff7f0e", "marker": "s", "ls": "--", "zorder": 4},
    "GNN direct": {"color": "#2ca02c", "marker": "^", "ls": "-", "zorder": 3},
    "GNN>MWPM": {"color": "#d62728", "marker": "D", "ls": "-", "zorder": 3},
    "GNN>BP+OSD": {"color": "#9467bd", "marker": "v", "ls": "-", "zorder": 3},
}

_BACKEND_STYLES: dict[str, dict[str, Any]] = {
    "pytorch": {"color": "#1f77b4", "marker": "o", "ls": "-"},
    "compiled": {"color": "#ff7f0e", "marker": "s", "ls": "--"},
    "cuda": {"color": "#2ca02c", "marker": "^", "ls": "-."},
    "tensorrt": {"color": "#d62728", "marker": "D", "ls": ":"},
}


def _sty(label: str) -> dict[str, Any]:
    return _STYLES.get(label, {"color": "gray", "marker": "x", "ls": "-", "zorder": 1})


def plot_ler_vs_p(
    eval_results: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """LER vs physical error probability, one subplot per distance.

    Points are aggregated over rounds (pooled errors / pooled shots)
    to give a single clean curve per (decoder, distance).
    """
    import matplotlib.pyplot as plt

    distances = sorted({r["distance"] for rs in eval_results.values() for r in rs})
    if not distances:
        return

    fig, axes = plt.subplots(
        1,
        len(distances),
        figsize=(5 * len(distances), 4.5),
        squeeze=False,
    )

    for col, d in enumerate(distances):
        ax = axes[0, col]

        for label, results in eval_results.items():
            agg = _aggregate_over_rounds([r for r in results if r["distance"] == d])
            if not agg:
                continue

            pts = sorted(agg.items(), key=lambda x: x[0][1])
            probs = [k[1] for k, _ in pts]
            lers = [v[0] for _, v in pts]

            s = _sty(label)
            ax.plot(
                probs,
                lers,
                label=label,
                marker=s["marker"],
                color=s["color"],
                ls=s["ls"],
                markersize=6,
                linewidth=1.8,
                zorder=s["zorder"],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate (p)")
        if col == 0:
            ax.set_ylabel("Logical error rate")
        ax.set_title(f"d = {d}", fontweight="bold")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 5),
        fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, output_dir / "ler_vs_p")


def plot_ler_scaling_with_d(
    eval_results: dict[str, list[dict]],
    output_dir: Path,
    *,
    reference_p: float | None = None,
) -> None:
    """LER vs code distance at a fixed p (averaged over rounds)."""
    import matplotlib.pyplot as plt

    all_probs = sorted({r["error_prob"] for rs in eval_results.values() for r in rs})
    if not all_probs:
        return

    if reference_p is None:
        reference_p = all_probs[len(all_probs) // 2]
    elif reference_p not in all_probs:
        reference_p = min(all_probs, key=lambda x: abs(x - reference_p))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    all_d = sorted({r["distance"] for rs in eval_results.values() for r in rs})

    for label, results in eval_results.items():
        agg = _aggregate_over_rounds(results)
        ds, lers = [], []
        for d in all_d:
            v = agg.get((d, reference_p))
            if v is not None:
                ds.append(d)
                lers.append(v[0])
        if not ds:
            continue

        s = _sty(label)
        ax.plot(
            ds,
            lers,
            label=label,
            marker=s["marker"],
            color=s["color"],
            ls=s["ls"],
            markersize=7,
            linewidth=1.8,
            zorder=s["zorder"],
        )

    ax.set_yscale("log")
    ax.set_xlabel("Code distance (d)")
    ax.set_ylabel("Logical error rate")
    ax.set_title(f"LER scaling -- p = {reference_p}", fontweight="bold")
    ax.set_xticks(all_d)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save_fig(fig, output_dir / "ler_scaling_d")


def plot_ler_vs_latency(
    eval_results: dict[str, list[dict]],
    benchmark_data: dict[str, Any],
    output_dir: Path,
    *,
    batch_size: int = 64,
) -> None:
    """LER vs inference latency -- one point per (decoder, backend)
    at a fixed batch size.
    """
    import matplotlib.pyplot as plt

    bench_results = benchmark_data.get("results", [])
    if not bench_results:
        return

    # (case, backend) > mean_ms at target batch_size
    latency_map: dict[tuple[str, str], float] = {}
    for br in bench_results:
        if br["batch_size"] == batch_size:
            latency_map[(br["case"], br["backend"])] = br["mean_ms"]

    _DECODER_TO_CASE = {
        "GNN direct": "direct",
        "GNN>MWPM": "edge",
        "GNN>BP+OSD": "edge",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, results in eval_results.items():
        case = _DECODER_TO_CASE.get(label)
        if case is None:
            continue

        mean_ler = np.mean([r["logical_error_rate"] for r in results])

        for backend, lat in sorted(latency_map.items(), key=lambda x: x[1]):
            if lat is None or backend[0] != case:
                continue

            bname = backend[1]
            bs = _BACKEND_STYLES.get(bname, {"color": "gray", "marker": "x"})
            ds = _sty(label)

            ax.scatter(
                lat,
                mean_ler,
                marker=ds["marker"],
                color=bs["color"],
                s=100,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.annotate(
                f"{label}\n{bname}",
                (lat, mean_ler),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=7,
                color=bs["color"],
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Inference latency at bs={batch_size} (ms)")
    ax.set_ylabel("Mean LER (across all settings)")
    ax.set_title("LER vs Latency", fontweight="bold")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    _save_fig(fig, output_dir / "ler_vs_latency")


def plot_throughput_vs_batch(
    benchmark_data: dict[str, Any],
    output_dir: Path,
) -> None:
    """Throughput vs batch size -- one subplot per case."""
    import matplotlib.pyplot as plt

    bench_results = benchmark_data.get("results", [])
    if not bench_results:
        return

    cases = sorted({br["case"] for br in bench_results})
    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=(5.5 * len(cases), 4.5),
        squeeze=False,
    )

    for col, case in enumerate(cases):
        ax = axes[0, col]
        subset = [br for br in bench_results if br["case"] == case]

        backends = sorted({br["backend"] for br in subset})
        for backend in backends:
            pts = sorted(
                [
                    (br["batch_size"], br["throughput_graphs_per_sec"])
                    for br in subset
                    if br["backend"] == backend
                ],
            )
            if not pts:
                continue
            bs_arr, tp_arr = zip(*pts)
            s = _BACKEND_STYLES.get(
                backend, {"color": "gray", "marker": "x", "ls": "-"}
            )
            ax.plot(
                bs_arr,
                tp_arr,
                label=backend,
                marker=s["marker"],
                color=s["color"],
                ls=s["ls"],
                linewidth=1.8,
                markersize=6,
            )

        ax.set_xlabel("Batch size")
        if col == 0:
            ax.set_ylabel("Throughput (graphs/s)")
        ax.set_title(f"{case}", fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Inference Throughput", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, output_dir / "throughput_vs_batch")


def plot_speedup_bar(
    benchmark_data: dict[str, Any],
    output_dir: Path,
    *,
    reference_backend: str = "pytorch",
) -> None:
    """Speedup bar chart -- one subplot per case, x-axis = batch sizes,
    grouped bars per backend, all normalised to pytorch at the same
    batch size.
    """
    import matplotlib.pyplot as plt

    bench_results = benchmark_data.get("results", [])
    if not bench_results:
        return

    cases = sorted({br["case"] for br in bench_results})
    other_backends = sorted(
        {br["backend"] for br in bench_results} - {reference_backend}
    )
    if not other_backends:
        return

    batch_sizes = sorted({br["batch_size"] for br in bench_results})

    # Reference latencies: {(case, batch_size): mean_ms}
    ref_map: dict[tuple[str, int], float] = {}
    for br in bench_results:
        if br["backend"] == reference_backend:
            ref_map[(br["case"], br["batch_size"])] = br["mean_ms"]
    if not ref_map:
        return

    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=(5.5 * len(cases), 4.5),
        squeeze=False,
    )

    for col, case in enumerate(cases):
        ax = axes[0, col]
        x = np.arange(len(batch_sizes))
        width = 0.7 / len(other_backends)

        for i, backend in enumerate(other_backends):
            speedups = []
            for bs in batch_sizes:
                ref = ref_map.get((case, bs))
                match = [
                    br
                    for br in bench_results
                    if br["case"] == case
                    and br["backend"] == backend
                    and br["batch_size"] == bs
                ]
                if match and ref and match[0]["mean_ms"] > 0:
                    speedups.append(ref / match[0]["mean_ms"])
                else:
                    speedups.append(0.0)

            offset = (i - (len(other_backends) - 1) / 2) * width
            color = _BACKEND_STYLES.get(backend, {}).get("color", "gray")
            bars = ax.bar(
                x + offset,
                speedups,
                width,
                label=backend,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, spd in zip(bars, speedups):
                if spd > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{spd:.1f}x",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )

        ax.axhline(y=1.0, color="gray", ls="--", alpha=0.4, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(bs) for bs in batch_sizes])
        ax.set_xlabel("Batch size")
        if col == 0:
            ax.set_ylabel(f"Speedup vs {reference_backend}")
        ax.set_title(f"{case}", fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Backend speedup", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, output_dir / "speedup_bar")


def _save_fig(fig: Any, stem: Path) -> None:
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
    import matplotlib

    matplotlib.use("Agg")

    eval_data = load_eval_results(results_dir)
    if not eval_data:
        logger.error("No evaluation results found in %s", results_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ler_vs_p(eval_data, output_dir)
    plot_ler_scaling_with_d(eval_data, output_dir, reference_p=reference_p)

    bench_data: dict[str, Any] = {}
    if benchmark_path is not None:
        bench_data = load_benchmark_results(benchmark_path)

    if bench_data.get("results"):
        plot_ler_vs_latency(eval_data, bench_data, output_dir)
        plot_throughput_vs_batch(bench_data, output_dir)
        plot_speedup_bar(bench_data, output_dir)
    else:
        logger.info("No benchmark data -- skipping latency/throughput/speedup plots")

    logger.info("All plots saved to %s", output_dir)
