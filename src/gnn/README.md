# gnn — GNN-Based QEC Decoders

Graph neural network decoders for rotated surface code, operating on the detector graph produced by `qec_generator`.

## Architecture

All three training modes share a single encoder backbone; only the
prediction head differs.

```
                       ┌─────────────────────────────────────────────┐
                       │              QECDecoder                     │
                       │                                             │
Batch ──► encoder ──► h│──► LogicalHead ──► (B, num_obs)             │
          (shared)     │    global_mean_pool => MLP                  │
                       │                                             │
                       │──► EdgeHead ──► (E_total,)                  │
                       │    [h_u ‖ h_v ‖ edge_attr] => MLP           │
                       └─────────────────────────────────────────────┘
```

### Encoder (`models/encoder.py`)

`DetectorGraphEncoder` runs several rounds of GINEConv message passing
on the detector graph.

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Convolution | GINEConv | Edge features (error prob, weight) enter messages natively; 1-WL expressiveness |
| Normalisation | LayerNorm | Stable across variable graph sizes in a batch (d=3 and d=7 mixed) |
| Skip connections | Additive residual per layer | Enables deeper networks without degradation |
| Edge projection | Single linear before layer 1 | Edge attributes are static — no need to re-project per layer |

Input/output:

- **In**: `x (N, 1)` syndrome bits, `edge_attr (E, 2)` `[error_prob, weight]`
- **Out**: `h (N, hidden_dim)` node embeddings

### Heads (`models/heads.py`)

**LogicalHead** — graph-level observable prediction:
1. `global_mean_pool(h, batch)` => `(B, hidden_dim)`
2. Two-layer MLP => `(B, num_observables)` logits

**EdgeHead** — per-edge prediction (used by both `mwpm_teacher` and `hybrid`):
1. Concatenate `[h_u, h_v, edge_attr]` per edge => `(E, 2H + edge_dim)`
2. Two-layer MLP => `(E,)` logits

### Factory

```python
from gnn.models import build_model

model = build_model("logical_head", hidden_dim=64, num_layers=4)
model = build_model("mwpm_teacher", hidden_dim=64, num_layers=4)
model = build_model("hybrid", hidden_dim=64, num_layers=4)
```

`mwpm_teacher` and `hybrid` produce identical architectures — they
differ only in loss computation and evaluation protocol.

## Training

### Quick start

```bash
# Train from config (recommended)
uv run scripts/train_gnn.py -c configs/train.yaml

# Override case and epochs
uv run scripts/train_gnn.py -c configs/train.yaml --case mwpm_teacher --epochs 50

# Pure CLI (no config file)
uv run scripts/train_gnn.py --case logical_head --epochs 50

# Custom hyperparameters
uv run scripts/train_gnn.py -c configs/train.yaml \
    --hidden-dim 32 --num-layers 2 --lr 5e-4 --batch-size 128

# Resume from checkpoint
uv run scripts/train_gnn.py -c configs/train.yaml --resume runs/logical_head/best.pt
```

### Configuration

All hyperparameters are set in `configs/train.yaml`.  CLI arguments
override config values, so the YAML file acts as the base and CLI
flags provide per-run tweaks.

```yaml
case: "logical_head"
model:
  hidden_dim: 64
  num_layers: 4
  dropout: 0.1
optimisation:
  lr: 1.0e-3
  epochs: 100
  batch_size: 64
seed: 42
```

Programmatic access: `TrainConfig.from_yaml("configs/train.yaml")`.

### Loss functions

| Case | Loss | Details |
|------|------|---------|
| `logical_head` | `BCEWithLogitsLoss` | Graph-level: logits vs observable ground truth |
| `mwpm_teacher` | `BCEWithLogitsLoss` + `pos_weight` | Per-edge: logits vs MWPM selections. Extreme class imbalance (most edges are 0), so `pos_weight` is auto-estimated from training data |
| `hybrid` | Same as `mwpm_teacher` | Same architecture and loss; different evaluation |

### Training details

- **Optimiser**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: Cosine annealing to lr/50 over all epochs
- **Checkpointing**: saves `best.pt` when validation metric improves
  - `logical_head`: tracks validation LER (lower is better)
  - edge cases: tracks validation loss (lower is better)
- **Reproducibility**: `--seed` sets Python, NumPy, and PyTorch RNGs

### Outputs

```
runs/{case}/
├── best.pt        # best model checkpoint
├── config.json    # full hyperparameter record
└── history.json   # per-epoch train/val metrics
```

### Hyperparameter defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `hidden_dim` | 64 | Embedding dimensionality |
| `num_layers` | 4 | Message-passing depth |
| `dropout` | 0.1 | Applied in encoder and heads |
| `lr` | 1e-3 | Initial learning rate |
| `weight_decay` | 1e-4 | AdamW L2 regularisation |
| `epochs` | 100 | |
| `batch_size` | 64 | Graphs per batch |
| `num_workers` | 4 | DataLoader parallelism |

## Evaluation

### Quick start

```bash
# Evaluate (case inferred from checkpoint)
uv run scripts/eval_gnn.py --checkpoint runs/logical_head/best.pt

# Compare with MWPM baseline
uv run scripts/eval_gnn.py --checkpoint runs/logical_head/best.pt \
    --baseline results/mwpm_baseline.json

# Save report
uv run scripts/eval_gnn.py --checkpoint runs/logical_head/best.pt \
    --baseline results/mwpm_baseline.json \
    -o results/gnn_logical.json
```

### Evaluation protocols

**`logical_head`**: for each shot, run the model on (graph, syndrome),
threshold the logit at 0 => predicted observable. Compare with ground
truth. Report LER per setting.

**`mwpm_teacher` / `hybrid`**: for each setting, collect per-edge
logits across all shots, average per undirected edge, convert to MWPM
weights via `sigmoid => log((1-p)/p)`, build a PyMatching decoder with
these weights, decode all syndromes, compare with ground truth.  This
tests whether the GNN learned a better noise model than the static DEM
weights.

### Output format

```
  d   r         p   shots    GNN_LER   MWPM_LER    delta
  3   3   0.00100    3000   0.001333   0.001000  +0.0003
  3   3   0.00500    3000   0.018000   0.019000  -0.0010 *
  ...

GNN better in 12/108 settings (marked with *)
```

JSON reports follow the same schema as the MWPM baseline, with added
`mwpm_ler` and `edge_acc` fields per setting.

## Dataset interface

The training and evaluation code consumes data from
`MixedSurfaceCodeDataset` (defined in `dataset.py`), which reads the
packaged shards produced by `qec_generator`.

Each `__getitem__` returns a PyG `Data` with:

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | `(N, 1)` | Syndrome bits (0/1 float), boundary node = 0 |
| `edge_index` | `(2, E)` | Directed COO (both directions stored) |
| `edge_attr` | `(E, 2)` | `[error_prob, weight]` |
| `y` | varies | `(num_obs,)` for logical; `(E,)` for edge cases |
| `logical` | `(num_obs,)` | Always present for evaluation |
| `setting_id` | scalar | Setting index for debugging |

Important: do not move graph tensors to GPU in the dataset (breaks
`num_workers > 0`).  The training loop calls `batch.to(device)`.
