# benchmarks — Inference Benchmarking & Result Plots

Benchmark GNN decoder inference latency/throughput across backends, and generate plots from evaluation and benchmark data.

## Quick start

### Inference benchmarks

```bash
# Default: pytorch + compiled backends, batch sizes 1/16/64/128
uv run scripts/benchmark_all.py

# All backends (requires torch-tensorrt for TRT)
uv run scripts/benchmark_all.py --backends pytorch compiled tensorrt

# Custom batch sizes and more iterations
uv run scripts/benchmark_all.py --batch-sizes 16 64 128 256 --n-iters 500 -v
```

Output: `outputs/benchmark_report.json`

### Plots

```bash
# All plots (eval results + benchmark data)
uv run scripts/plot_results.py

# Eval-only plots (no benchmark data needed)
uv run scripts/plot_results.py --no-benchmark

# Custom reference error probability for scaling plot
uv run scripts/plot_results.py --reference-p 0.005

# Custom output directory
uv run scripts/plot_results.py -o outputs/figures
```

Output: PDF + PNG figures in `outputs/figures/`.

Requires `matplotlib` (not a core project dependency):

```bash
pip install matplotlib
```

## Benchmark report format

```json
{
  "hardware": {
    "platform": "Linux-6.1...",
    "gpu": "NVIDIA GeForce RTX 4080 Laptop GPU",
    "gpu_memory_gb": 12.0,
    "torch": "2.9.1+cu128",
    "cuda_version": "12.8"
  },
  "results": [
    {
      "checkpoint": "outputs/runs/direct/best.pt",
      "case": "direct",
      "backend": "compiled",
      "batch_size": 64,
      "n_iters": 100,
      "mean_ms": 1.42,
      "std_ms": 0.12,
      "median_ms": 1.40,
      "min_ms": 1.28,
      "max_ms": 2.15,
      "throughput_graphs_per_sec": 45070.4,
      "peak_memory_mb": 312.5
    }
  ]
}
```

## Generated plots

| # | File | Description | Required data |
|---|------|-------------|---------------|
| 1 | `ler_vs_p` | LER vs physical error prob, per distance, Wilson CI bands | Eval JSONs |
| 2 | `ler_scaling_d` | LER vs code distance at fixed p | Eval JSONs |
| 3 | `ler_vs_latency` | LER vs inference latency (Pareto front) | Eval JSONs + benchmark |
| 4 | `throughput_vs_batch` | Graphs/s vs batch size, by backend | Benchmark |
| 5 | `speedup_bar` | Speedup normalised to pytorch baseline | Benchmark |

Plots 1-2 work with evaluation data alone. Plots 3-5 require `benchmark_report.json` — use `--no-benchmark` to skip them.

## Expected input files

Evaluation results (from `eval_all_cases.py`):

```
outputs/results/
├── mwpm_baseline.json
├── bp_osd_baseline.json      # optional (requires cudaq-qec)
├── gnn_direct.json
├── gnn_edge_mwpm.json
└── gnn_edge_bp_osd.json      # optional (requires cudaq-qec)
```

Benchmark report (from `benchmark_all.py`):

```
outputs/benchmark_report.json
```

## Module layout

```
src/benchmarks/
├── __init__.py
├── runner.py      # Benchmark harness: discovery, timing, memory, JSON report
├── plots.py       # Plot generation: data loading, 5 figure types, PDF+PNG output
└── README.md

scripts/
├── benchmark_all.py   # CLI entry point for runner
└── plot_results.py    # CLI entry point for plots
```

## Full pipeline

```bash
# 1. Train models
uv run scripts/train_all_cases.py -v

# 2. Evaluate all decoders
uv run scripts/eval_all_cases.py -v

# 3. Benchmark inference
uv run scripts/benchmark_all.py -v

# 4. Generate plots
uv run scripts/plot_results.py -v
```
