"""Smoke tests for benchmarks.plots — call each function with synthetic data."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def eval_results() -> dict[str, list[dict]]:
    """Minimal synthetic eval results for two decoders."""
    return {
        "MWPM": [
            {
                "distance": 3,
                "rounds": 3,
                "error_prob": 0.01,
                "num_errors": 50,
                "num_shots": 1000,
                "logical_error_rate": 0.05,
            },
            {
                "distance": 5,
                "rounds": 5,
                "error_prob": 0.01,
                "num_errors": 20,
                "num_shots": 1000,
                "logical_error_rate": 0.02,
            },
        ],
        "GNN direct": [
            {
                "distance": 3,
                "rounds": 3,
                "error_prob": 0.01,
                "num_errors": 40,
                "num_shots": 1000,
                "logical_error_rate": 0.04,
            },
            {
                "distance": 5,
                "rounds": 5,
                "error_prob": 0.01,
                "num_errors": 15,
                "num_shots": 1000,
                "logical_error_rate": 0.015,
            },
        ],
    }


@pytest.fixture()
def benchmark_data() -> dict:
    """Minimal synthetic benchmark data."""
    return {
        "results": [
            {
                "case": "direct",
                "backend": "pytorch",
                "batch_size": 64,
                "mean_ms": 10.0,
                "throughput_graphs_per_sec": 6400.0,
            },
            {
                "case": "direct",
                "backend": "compiled",
                "batch_size": 64,
                "mean_ms": 5.0,
                "throughput_graphs_per_sec": 12800.0,
            },
        ],
    }


@pytest.fixture(autouse=True)
def _matplotlib_agg() -> None:
    """Force non-interactive backend."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_plot_ler_vs_p(
    eval_results: dict[str, list[dict]],
    tmp_path: Path,
) -> None:
    from benchmarks.plots import plot_ler_vs_p

    plot_ler_vs_p(eval_results, tmp_path)
    assert (tmp_path / "ler_vs_p.png").exists()


def test_plot_ler_scaling_with_d(
    eval_results: dict[str, list[dict]],
    tmp_path: Path,
) -> None:
    from benchmarks.plots import plot_ler_scaling_with_d

    plot_ler_scaling_with_d(eval_results, tmp_path, reference_p=0.01)
    assert (tmp_path / "ler_scaling_d.png").exists()


def test_plot_throughput_vs_batch(
    benchmark_data: dict,
    tmp_path: Path,
) -> None:
    from benchmarks.plots import plot_throughput_vs_batch

    plot_throughput_vs_batch(benchmark_data, tmp_path)
    assert (tmp_path / "throughput_vs_batch.png").exists()


def test_plot_speedup_bar(
    benchmark_data: dict,
    tmp_path: Path,
) -> None:
    from benchmarks.plots import plot_speedup_bar

    plot_speedup_bar(benchmark_data, tmp_path)
    assert (tmp_path / "speedup_bar.png").exists()


def test_plot_ler_vs_latency(
    eval_results: dict[str, list[dict]],
    benchmark_data: dict,
    tmp_path: Path,
) -> None:
    from benchmarks.plots import plot_ler_vs_latency

    plot_ler_vs_latency(eval_results, benchmark_data, tmp_path, batch_size=64)


def test_generate_all_plots_no_crash(
    tmp_path: Path,
) -> None:
    from benchmarks.plots import generate_all_plots

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    generate_all_plots(results_dir, None, tmp_path / "plots")
