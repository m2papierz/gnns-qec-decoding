# gnns-qec-decoding

An end-to-end project on decoding topological quantum error-correcting codes with Graph Neural Networks (GNNs).

The goal is to build a clean, reproducible pipeline that:
- **simulates surface-code memory experiments** (Stim),
- **turns detector error models into graphs** (the decoding / matching graph),
- **trains GNN decoders** on mixed regimes (distance, rounds, physical error rate),
- and **compares against classical baselines** (e.g., MWPM-style decoding / teacher supervision where relevant).

## Developer setup

Quick start:
1. Install `uv` (if you don't have it):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
2. Synchronize the environment:
  ```bash
  uv sync
  ```
3. Run tests:
  ```bash
  uv run --locked pytest
  ```

Code formatting/linting:
```bash
uv run ruff check --fix .
uv run isort .
uv run black .
```

## Dataset Generation

Generate surface code datasets for GNN-based QEC decoders.

### Dataset Cases

The generator supports three labeling modes (via `--cases`):

- **`logical_head`**: Labels are ground-truth logical outcomes from simulation. For training models to predict logical errors directly.

- **`mwpm_teacher`**: Labels from MWPM decoder predictions. For distillation/imitation learning where GNN mimics classical decoder.

- **`hybrid`**: Combines both logical and MWPM labels. For multi-task learning or auxiliary teacher loss.

*All cases use identical raw samples - only the supervision differs.*

### Usage

**CLI:**
```bash
# Full pipeline (raw + datasets)
uv run scripts/data_generation.py -c configs/data_generation.yaml

# Raw data only
uv run scripts/data_generation.py --mode raw-only

# Specific cases with verbose logging
uv run scripts/data_generation.py --cases logical_head hybrid -v

# Skip MWPM label generation
uv run scripts/data_generation.py --no-mwpm
```

**Python API:**
```python
from qec_generator import Config, generate_raw_data, generate_datasets

cfg = Config.from_yaml("configs/data_generation.yaml")
generate_raw_data(cfg)
generate_datasets(cfg, cases=("logical_head", "mwpm_teacher"))
```

### Output Structure

**Raw data** (shared across all cases):
```
data/raw/
└── d{distance}_r{rounds}_p{error_prob}/
    ├── circuit.stim, model.dem
    ├── graph/
    │   ├── edge_index.npy
    │   ├── edge_error_prob.npy
    │   └── node_*.npy
    └── {split}_syndrome.npy, {split}_logical.npy
```

**Packaged datasets** (per case):
```
data/datasets/{case}/
├── shards/
│   └── d{distance}_r{rounds}_p{error_prob}/
│       ├── graph/              # Graph tensors
│       ├── {split}_syndrome.npy
│       ├── {split}_logical.npy
│       └── {split}_mwpm_*.npy  # Only for mwpm_teacher/hybrid
├── splits/
│   └── {split}_index.npz       # Global indices
└── settings.json               # Metadata
```

### Configuration

See `configs/data_generation.yaml` for full options:
```yaml
surface_code:
  family: "rotated_memory_x"
  distances: [3, 5, 7]
  rounds: [3, 5]
  error_probs: [0.001, 0.005, 0.01]
  num_samples: {train: 50000, val: 10000, test: 10000}
```
