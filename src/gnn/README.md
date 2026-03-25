# gnn — GNN-Based QEC Decoders

Graph neural network decoders for rotated surface code, operating on the detector graph produced by `qec_generator`.

## Architecture

All training modes share a single encoder backbone; only the prediction head differs.

```
                       ┌─────────────────────────────────────────────┐
                       │              QECDecoder                     │
                       │                                             │
Batch ──► encoder ──►(h, edge_h)                                     │
          (shared)     │                                             │
                       │──► LogicalHead ──► (B, num_obs)             │
                       │    attn_pool ‖ max_pool ‖ edge_pool => MLP  │
                       │                                             │
                       │──► EdgeHead ──► (E_total,)                  │
                       │    [h_u+h_v ‖ |h_u−h_v| ‖ edge_h] => MLP    │
                       └─────────────────────────────────────────────┘
```

### Encoder (`models/encoder.py`)

`DetectorGraphEncoder` runs several rounds of GINEConv message passing on the detector graph with explicit edge co-evolution: each layer updates both node embeddings and edge embeddings.

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Convolution | GINEConv | Edge features (error prob, weight) enter messages natively; 1-WL expressiveness |
| Edge update | `MLP(h_src + h_dst, \|h_src − h_dst\|, e)` | Symmetric, learns edge representations jointly with nodes |
| Normalisation | LayerNorm (nodes and edges) | Stable across variable graph sizes in a batch (d=3 and d=7 mixed) |
| Skip connections | Additive residual per layer (nodes and edges) | Enables deeper networks without degradation |
| Edge projection | Per-layer linear + accumulation | Each layer projects raw edge features and adds previous edge embeddings |

Input/output:

- **In**: `x (N, 1)` syndrome bits, `edge_attr (E, 2)` `[error_prob, weight]`
- **Out**: `h (N, hidden_dim)` node embeddings, `edge_h (E, hidden_dim)` edge embeddings

### Swappable compute operations (`models/ops.py`)

The encoder and heads forward passes delegate compute-intensive patterns to `gnn.models.ops`, which dispatches to one of three backends:

| Backend | Description |
|---------|-------------|
| `pytorch` | Pure PyTorch reference implementations (default) |
| `compiled` | `torch.compile`-wrapped PyTorch — same numerics, compiler-driven kernel fusion |
| `cuda` | Hand-written CUDA kernels |

Set via `QECDEC_BACKEND` env var or `set_backend()` at runtime.

### Heads (`models/heads.py`)

**LogicalHead** — graph-level observable prediction:
1. Attention-weighted sum pooling over nodes => `(B, H)`
2. Max pooling over nodes => `(B, H)`
3. Mean pooling over edge embeddings per graph => `(B, H)`
4. Concatenate all three => two-layer MLP => `(B, num_observables)` logits

**EdgeHead** — per-edge prediction (used by `edge`):
1. Symmetric node pair: `[h_u + h_v, |h_u − h_v|, edge_h]` => `(E, 3H)`
2. Two-layer MLP => `(E,)` logits

### Factory

```python
from gnn.models import build_model

model = build_model("direct", hidden_dim=128, num_layers=6)
model = build_model("edge", hidden_dim=128, num_layers=6)
```

`edge` uses `EdgeHead` — it differs from `direct` in loss
computation, label format, and evaluation protocol.

## Decoders (`decoders/`)

Pluggable decoder interface for converting GNN edge logits to
observable predictions.

| Decoder | Description |
|---------|-------------|
| `MWPMDecoder` | PyMatching-backed MWPM; accepts static or GNN-derived weights |
| `BPOSDDecoder` | NVIDIA CUDA-Q BP+OSD; accepts static or GNN-derived edge probabilities |

All decoders implement `BaseDecoder` with `decode()` and `decode_batch()`.
The evaluator selects a decoder via `--decoder` (`mwpm` or `bp_osd`).

## Training

### Quick start

```bash
# Train from config (recommended)
uv run scripts/train_gnn.py -c configs/train.yaml

# Override case and epochs
uv run scripts/train_gnn.py -c configs/train.yaml --case edge --epochs 50

# Use torch.compile backend
uv run scripts/train_gnn.py -c configs/train.yaml --backend compiled

# Pure CLI (no config file)
uv run scripts/train_gnn.py --case direct --epochs 50

# Custom hyperparameters
uv run scripts/train_gnn.py -c configs/train.yaml \
    --hidden-dim 32 --num-layers 2 --lr 5e-4 --batch-size 128

# Resume from checkpoint
uv run scripts/train_gnn.py -c configs/train.yaml --resume outputs/runs/direct/best.pt

# Train all cases (data generation + direct + edge)
uv run scripts/train_all_cases.py -v
```

### Configuration

All hyperparameters are set in `configs/train.yaml`.  CLI arguments override config values, so the YAML file acts as the base and CLI flags provide per-run tweaks.

```yaml
case: "direct"
backend: "compiled"         # "pytorch" | "compiled" | "cuda"
compile_mode: "reduce-overhead"
model:
  hidden_dim: 128
  num_layers: 6
  dropout: 0.1
optimisation:
  lr: 1.5e-4
  weight_decay: 1.0e-5
  epochs: 750
  batch_size: 128
patience: 75
val_every: 5
seed: 42
```

Programmatic access: `TrainConfig.from_yaml("configs/train.yaml")`.

`train_all_cases.py` uses a lower LR (`1.0e-4`) for the `edge` case via CLI override — the MSE loss landscape is smoother than BCE and converges better at a lower learning rate.

### Loss functions

| Case | Loss | Details |
|------|------|---------|
| `direct` | `BCEWithLogitsLoss` | Graph-level: logits vs observable ground truth |
| `edge` | `_SoftTeacherLoss` | Graph-normalized MSE: `(σ(logit) - target)²` averaged within each graph, then across graphs. Targets are continuous BP marginals in `[0, 1]` |

### Training details

- **Optimiser**: AdamW (weight_decay configurable, default 1e-5)
- **Scheduler**: Linear warmup (5% of epochs, from 1% of peak lr) followed by cosine annealing to lr/50
- **Mixed precision**: AMP enabled automatically on CUDA (GradScaler + autocast)
- **Data loading**: persistent workers + prefetch for reduced overhead
- **Checkpointing**: saves `best.pt` when validation metric improves
  - `direct`: tracks validation LER (lower is better)
  - `edge`: tracks validation loss (lower is better)
- **Reproducibility**: `--seed` sets Python, NumPy, and PyTorch RNGs

### Outputs

```
outputs/runs/{case}/
├── best.pt        # best model checkpoint
├── config.json    # full hyperparameter record
└── history.json   # per-epoch train/val metrics
```

### Hyperparameter defaults

These are the code defaults in `TrainConfig`.  The shipped `configs/train.yaml` overrides several of them (see Configuration above).

| Parameter | Code default | Notes |
|-----------|-------------|-------|
| `hidden_dim` | 64 | Embedding dimensionality |
| `num_layers` | 4 | Message-passing depth |
| `dropout` | 0.1 | Applied in encoder and heads |
| `lr` | 1e-3 | Peak learning rate (after warmup) |
| `weight_decay` | 1e-4 | AdamW L2 regularisation |
| `epochs` | 100 | |
| `batch_size` | 64 | Graphs per batch |
| `num_workers` | 4 | DataLoader parallelism |
| `backend` | `pytorch` | Compute backend |

## Evaluation

### Quick start

```bash
# Evaluate (case inferred from checkpoint)
uv run scripts/eval_gnn.py --checkpoint outputs/runs/direct/best.pt

# Compare with MWPM baseline
uv run scripts/eval_gnn.py --checkpoint outputs/runs/direct/best.pt \
    --baseline outputs/results/mwpm_baseline.json

# Edge model with MWPM decoder
uv run scripts/eval_gnn.py --checkpoint outputs/runs/edge/best.pt \
    --decoder mwpm --baseline outputs/results/mwpm_baseline.json

# Edge model with BP+OSD decoder (CUDA-Q)
uv run scripts/eval_gnn.py --checkpoint outputs/runs/edge/best.pt \
    --decoder bp_osd --baseline outputs/results/bp_osd_baseline.json

# Save report
uv run scripts/eval_gnn.py --checkpoint outputs/runs/direct/best.pt \
    --baseline outputs/results/mwpm_baseline.json \
    -o outputs/results/gnn_logical.json
```

### Evaluation protocols

**`direct`**: for each shot, run the model on (graph, syndrome),
threshold the logit at 0 => predicted observable. Compare with ground
truth. Report LER per setting.

**`edge`**: for each setting, collect per-edge logits across all shots, build a decoder from the GNN-predicted weights, decode all syndromes, compare with ground truth. The decoder backend is selected via `--decoder`:

- `mwpm` (default): GNN edge logits => sigmoid => MWPM weights => PyMatching decode
- `bp_osd`: GNN edge logits => sigmoid => error probabilities => CUDA-Q BP+OSD decode => project onto observables via `observable_flips` mask

### Output format

```
  d   r         p   shots    GNN_LER   MWPM_LER    delta
  3   3   0.00100    3000   0.001333   0.001000  +0.0003
  3   3   0.00500    3000   0.018000   0.019000  -0.0010 *
  ...

GNN better in 12/108 settings (marked with *)
```

JSON reports follow the same schema as the MWPM baseline, with added `mwpm_ler` field per setting.

## Dataset interface

The training and evaluation code consumes data from `MixedSurfaceCodeDataset` (defined in `dataset.py`), which reads the packaged shards produced by `qec_generator`.

Each `__getitem__` returns a PyG `Data` with:

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | `(N, 1)` | Syndrome bits (0/1 float), boundary node = 0 |
| `edge_index` | `(2, E)` | Directed COO (both directions stored) |
| `edge_attr` | `(E, 2)` | `[error_prob, weight]` |
| `y` | varies | `(num_obs,)` for direct; `(E,)` float BP marginals for edge |
| `logical` | `(num_obs,)` | Always present for evaluation |
| `setting_id` | scalar | Setting index for debugging |

Important: do not move graph tensors to GPU in the dataset (breaks `num_workers > 0`).  The training loop calls `batch.to(device)`.
