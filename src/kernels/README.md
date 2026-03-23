# Custom CUDA Kernels

Fused CUDA kernels for the GNN encoder and loss computation, targeting NVIDIA Ada Lovelace (sm_89) and Ampere (sm_80) architectures.

## Kernels

| Kernel | Replaces | Speedup source |
|--------|----------|----------------|
| `fused_edge_features` | 2x gather + add + sub+abs + concat (5 launches => 1) | Float4 vectorized loads, single pass |
| `fused_norm_residual` | LayerNorm + Dropout + residual add (3 launches => 1) | Warp-level reductions, fused RNG |
| `graph_norm_bce` | BCE + scatter_add + normalize (3 launches => 2) | Atomic scatter + block reduce |

## Prerequisites

- CUDA Toolkit ≥ 12.0
- PyTorch ≥ 2.5 with CUDA support
- A GPU with compute capability ≥ 8.0 (Ampere or newer)

## Build

### Option A: Ahead-of-time (recommended)

From the **project root**:

```bash
uv run scripts/build_kernels.py build_ext --inplace
```

Verify:

```bash
uv run python -c "import kernels; print(kernels.AVAILABLE)"  # True
```

### Option B: JIT compilation (development)

```python
from kernels.build import build
module = build()  # compiles on first call, cached afterwards
```

JIT is slower on first run but doesn't require the build step.

## Usage

Kernels activate automatically when the compute backend is set to `"cuda"`:

```python
from gnn.models.ops import set_backend
set_backend("cuda")
```

Or via environment variable:

```bash
QECDEC_BACKEND=cuda uv run scripts/train_gnn.py -c configs/train.yaml
```

Or via CLI flag:

```bash
uv run scripts/train_gnn.py --backend cuda -c configs/train.yaml
```

If kernels are not built, `set_backend("cuda")` falls back to PyTorch
with a warning.

## Testing

Equivalence tests verify CUDA kernels match PyTorch output within `atol=1e-5`:

```bash
uv run scripts/build_kernels.py build_ext --inplace
uv run pytest tests/test_ops_equivalence.py -k "CUDA" -v
```

Tests are automatically skipped without a GPU or without built kernels.

## Architecture notes

- **One thread-block per edge** in `fused_edge_features` — each block processes one edge's hidden dimension with float4 vectorized loads. Tail elements (when `hidden_dim % 4 != 0`) handled with scalar fallback.

- **One thread-block per row** in `fused_norm_residual` — two-pass mean/variance via `__shfl_xor_sync` warp reductions. Dropout uses Philox PRNG seeded from `steady_clock`.

- **Atomic scatter** in `graph_norm_bce` — per-edge BCE computed in parallel, `atomicAdd` into per-graph accumulators, then single-block reduce for the final mean-of-means.

## File layout

```
src/kernels/
├── README.md
├── __init__.py          # AVAILABLE flag
├── ops.py               # Python wrappers (fallback to PyTorch on CPU)
├── build.py             # JIT build config
└── cpp/
    ├── warp_reduce.cuh          # Warp/block reduction primitives
    ├── fused_edge_features.cu   # Kernel #1
    ├── fused_norm_residual.cu   # Kernel #2
    ├── graph_norm_bce.cu        # Kernel #3
    └── bindings.cpp             # pybind11 => kernels._C
```
