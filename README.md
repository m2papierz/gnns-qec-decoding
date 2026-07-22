# gnn-surface-code-decoding

End-to-end pipeline for decoding surface codes with graph neural networks: Stim circuit sampling, detector graph construction, GNN training with on-the-fly data generation, multi-backend inference (PyTorch, compiled, TensorRT), statistically rigorous evaluation against classical baselines (MWPM, BP+OSD), and performance benchmarking.

> [!IMPORTANT]
> **Learning project** — built to deepen hands-on understanding of GNN-based QEC decoding, detector graph construction, and the interplay between classical decoders and learned models. Not a production decoder.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest            # verify installation
```

## Dataset generation

Generate Stim circuits and frozen evaluation sets for GNN training and evaluation:

```bash
uv run scripts/generate_circuits.py          # circuits for d∈{3,5,7}, p∈{0.003..0.01}
uv run scripts/generate_eval_sets.py         # frozen eval sets with adaptive sampling
uv run scripts/generate_ci_shard.py          # small CI shard for test suite
```

See [`docs/sampling.md`](docs/sampling.md) for graph construction, circuit metadata, and the sampling API.

## GNN training

Train a GNN decoder using sample-budget training with on-the-fly Stim sampling:

```bash
uv run scripts/train_gnn.py -c configs/train.yaml

# Per-distance configs with tuned budgets
uv run scripts/train_gnn.py -c configs/train_d3.yaml
uv run scripts/train_gnn.py -c configs/train_d5.yaml
uv run scripts/train_gnn.py -c configs/train_d7.yaml
```

See [`docs/model.md`](docs/model.md) for architecture details (GINE encoder, LogicalHead, pooling), hyperparameters, and training configuration.

## Evaluation

Evaluate the GNN against classical baselines (MWPM, BP+OSD) on frozen eval sets with adaptive stopping and paired statistical tests:

```bash
# Full eval harness: GNN vs MWPM vs BP+OSD on frozen sets
uv run scripts/eval_harness.py --checkpoint outputs/runs/direct/best.pt

# Quick sanity check on fresh samples
uv run scripts/eval_sanity.py --checkpoint outputs/runs/direct/best.pt
```

See [`docs/eval_protocol.md`](docs/eval_protocol.md) for the pre-registered stopping rule, McNemar test, and Wilson confidence intervals.

## Deployment and benchmarking

Benchmark inference across backends and export to TensorRT:

```bash
uv run scripts/benchmark_all.py -v           # inference latency/throughput/memory
uv run scripts/export_trt.py --checkpoint outputs/runs/direct/best.pt
```

See [`docs/deployment.md`](docs/deployment.md) for the inference engine API, backend details, and TRT graph partitioning.

See [`docs/kernels.md`](docs/kernels.md) for custom CUDA kernels (inference only).

## Plots

Generate evaluation and benchmark figures:

```bash
uv run scripts/plot_results.py -v            # eval + benchmark figures
uv run scripts/plot_calibration.py           # reliability diagrams and ECE
```

## Development

```bash
make fmt                 # ruff format + import sorting
make lint                # ruff check
make test                # pytest
```

## License

MIT License — see [LICENSE](LICENSE).
