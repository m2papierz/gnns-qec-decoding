# Deployment & Inference

Unified inference engine for trained GNN decoders with multi-backend support and built-in benchmarking.

## Backends

| Backend | What it does | Requires |
|---------|-------------|----------|
| `pytorch` | Vanilla eager-mode inference | Nothing extra |
| `compiled` | `torch.compile` with `reduce-overhead` mode; Triton kernel fusion | GPU |
| `tensorrt` | `torch.compile` with `torch_tensorrt` backend; dense subgraphs (Linear, LayerNorm, MLP) lowered to TRT engines, sparse GNN ops (scatter, gather) stay in PyTorch | `torch-tensorrt`, GPU |

All backends accept the same `torch_geometric.data.Batch` input and produce numerically close outputs. The `tensorrt` backend with FP16 has wider tolerance (~1e-2) than `compiled` (~1e-5) due to reduced precision.

## Quick start

```bash
# Benchmark all backends on a trained checkpoint
uv run scripts/export_trt.py --checkpoint outputs/runs/direct/best.pt

# Skip TensorRT (no torch-tensorrt needed)
uv run scripts/export_trt.py \
    --checkpoint outputs/runs/edge/best.pt \
    --backends pytorch compiled

# Custom batch geometry and iteration count
uv run scripts/export_trt.py \
    --checkpoint outputs/runs/edge/best.pt \
    --n-graphs 8 --n-nodes 100 --n-edges 240 --n-iters 200

# FP32 TensorRT (tighter tolerance, lower speedup)
uv run scripts/export_trt.py \
    --checkpoint outputs/runs/direct/best.pt \
    --precision fp32
```

Produces a JSON report next to the checkpoint (or at `--output`).

## Python API

```python
from deploy.engine import InferenceEngine, load_model_from_checkpoint, make_synthetic_batch

model, cfg = load_model_from_checkpoint("outputs/runs/direct/best.pt")

# Eager baseline
engine = InferenceEngine(model, backend="pytorch", device="cuda")
out = engine.predict(batch)

# TensorRT with FP16
engine = InferenceEngine(model, backend="tensorrt", device="cuda", precision="fp16")
engine.warmup(batch)
metrics = engine.benchmark(batch, n_iters=100)
# metrics = {"mean_ms": 0.42, "throughput_graphs_per_sec": 9523, ...}
```

### `InferenceEngine` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `"pytorch"` | `"pytorch"`, `"compiled"`, or `"tensorrt"` |
| `device` | `"cuda"` | Target device |
| `precision` | `"fp16"` | TRT precision (`"fp32"` or `"fp16"`); ignored for other backends |
| `min_block_size` | `3` | Min consecutive TRT-convertible ops per TRT segment |
| `warmup_iters` | `10` | Forward passes before benchmarking |

### Benchmark output

```json
{
  "backend": "tensorrt",
  "precision": "fp16",
  "n_graphs": 4,
  "n_iters": 100,
  "mean_ms": 0.42,
  "std_ms": 0.03,
  "median_ms": 0.41,
  "min_ms": 0.38,
  "max_ms": 0.55,
  "throughput_graphs_per_sec": 9523.8
}
```

Timing uses CUDA events (sub-ms precision) on GPU, `time.perf_counter` on CPU.

## How TRT partitioning works for GNNs

GNNs mix dense ops (MLP projections, LayerNorm) with sparse graph ops (scatter, gather, message passing). TensorRT can only accelerate the dense parts. `torch.compile` with the `torch_tensorrt` backend handles this automatically:

1. Traces the model graph via `torch.compile`
2. Identifies contiguous subgraphs where all ops are TRT-convertible
3. Subgraphs with ≥ `min_block_size` ops become TRT engines
4. Remaining ops (GINEConv, scatter_add, softmax over graph) stay as PyTorch

This means no manual model surgery is needed — the same `QECDecoder` works across all backends.

## Supported training cases

Both cases work with every backend:

| Case | Head | Output shape |
|------|------|-------------|
| `direct` | `LogicalHead` | `(B, num_observables)` |
| `edge` | `EdgeHead` | `(E_total,)` |

## Testing

```bash
# All deploy tests (pytorch backend runs on CPU, rest need GPU)
uv run pytest tests/test_deploy.py -v

# Just pytorch backend (no GPU needed)
uv run pytest tests/test_deploy.py -k "Pytorch" -v

# TRT tests (need torch-tensorrt + GPU)
uv run pytest tests/test_deploy.py -k "TensorRT" -v
```
