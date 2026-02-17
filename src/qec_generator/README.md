# qec_generator — QEC Dataset Generation

Generate surface-code datasets for GNN-based quantum error correction decoders.

The module wraps [Stim](https://github.com/quantumlib/Stim) circuit
simulation and [PyMatching](https://pymatching.readthedocs.io/) decoding
into a reproducible pipeline that produces ready-to-train NumPy shards.

## Concepts

### Surface code memory experiment

`stim.Circuit.generated("surface_code:rotated_memory_x", ...)` creates a
*rotated surface code memory-X experiment*: the logical qubit is
initialised and measured in the X basis.  During the experiment,
stabiliser measurements run for a configurable number of **rounds**
(syndrome extraction cycles).

### Detector error model (DEM)

A DEM captures which physical faults trigger which *detectors*
(differences between consecutive stabiliser outcomes) and which flip
*observables* (the logical measurement).  The module calls
`circuit.detector_error_model(decompose_errors=True)` to obtain a
graph-like DEM suitable for matching.

### Detector / decoding graph

PyMatching converts the DEM into a weighted graph where nodes are
detectors and edges carry error probabilities and MWPM weights.  The
module adds a single **virtual boundary node** (index = `num_detectors`)
that collapses all boundary edges, making the graph structure stable
across settings.

### Labels

| Label | Source | Use case |
|---|---|---|
| `logical` | Stim simulation ground truth | Direct logical-error prediction |
| `mwpm_edge_selected_packed` | PyMatching MWPM solution | Teacher / distillation supervision |

### Rounds

`rounds` controls how many stabiliser measurement cycles run in each
memory experiment.  More rounds => longer temporal axis => more detectors
and edges.  By default the pipeline generates `rounds ∈ {d, 2·d}` for
each distance `d`.  Override this via `rounds:` in the YAML config.

## Quick start

```bash
# Full pipeline (raw samples + packaged datasets)
uv run scripts/data_generation.py -c configs/data_generation.yaml

# Raw data only
uv run scripts/data_generation.py --mode raw-only

# Specific cases with verbose logging
uv run scripts/data_generation.py --cases logical_head hybrid -v

# Overwrite existing data
uv run scripts/data_generation.py --overwrite
```

### Python API

```python
from qec_generator import Config, generate_raw_data, generate_datasets

cfg = Config.from_yaml("configs/data_generation.yaml")

# Step 1: sample circuits
generate_raw_data(cfg)

# Step 2: package for ML
generate_datasets(cfg, cases=("logical_head", "mwpm_teacher", "hybrid"))
```

## Configuration (YAML)

```yaml
surface_code:
  family: "rotated_memory_x"
  distances: [3, 5, 7]
  error_probs: [0.001, 0.005, 0.01]

  # Optional — omit to auto-generate [d, 2*d] per distance.
  # rounds: [3, 5, 6, 7, 10, 14]

  num_samples:
    train: 15000
    val: 3000
    test: 3000

  # Stim noise channels.  "p" is substituted with error_prob.
  noise:
    after_clifford_depolarization: "p"
    after_reset_flip_probability: "p"
    before_measure_flip_probability: "p"
    before_round_data_depolarization: "p"

output:
  raw_data_dir: "./data/raw_data"
  datasets_dir: "./data/datasets"
  compress: true

generation:
  chunk_size: 10000
  seed: 42
```

## Output structure

### Raw data (shared across all cases)

```
data/raw_data/
└── d{distance}_r{rounds}_p{error_prob}/
    ├── circuit.stim              # Stim circuit (debug artifact)
    ├── model.dem                 # Detector error model
    ├── graph/
    │   ├── edge_index.npy        # (2, E) int64 — directed COO
    │   ├── edge_error_prob.npy   # (E,) float32
    │   ├── edge_weight.npy       # (E,) float32 — MWPM weight
    │   ├── node_coords.npy       # (N, C) float32
    │   ├── node_is_boundary.npy  # (N,) bool
    │   └── meta.json
    ├── {split}_syndrome.npy      # (S, D) uint8
    ├── {split}_logical.npy       # (S, O) uint8
    └── {split}_meta.json
```

### Packaged datasets (per case)

```
data/datasets/{case}/
├── shards/
│   └── d{distance}_r{rounds}_p{error_prob}/
│       ├── graph/
│       │   ├── edge_index.npy       # (2, E) int64
│       │   ├── edge_attr.npy        # (E, 2) float32 [error_prob, weight]
│       │   ├── node_coords.npy      # (N, C) float32
│       │   ├── node_is_boundary.npy # (N,) bool
│       │   └── graph_meta.json
│       ├── {split}_syndrome.npy
│       ├── {split}_logical.npy
│       ├── {split}_mwpm_edge_selected_packed.npy   # mwpm_teacher / hybrid only
│       └── mwpm/                                    # mwpm_teacher / hybrid only
│           ├── undirected_edge_endpoints.npy
│           ├── dir_to_undir.npy
│           └── teacher_meta.json
├── splits/
│   └── {split}_index.npz    # arrays: setting_id, shot_id
├── settings.json
└── build_meta.json
```

### Dataset cases

All cases share identical raw samples — only the supervision differs.

| Case | Labels included | Purpose |
|---|---|---|
| `logical_head` | `logical` | Train model to predict logical errors directly |
| `mwpm_teacher` | `logical` + MWPM edge labels | Distillation / imitation learning |
| `hybrid` | `logical` + MWPM edge labels | Multi-task or auxiliary teacher loss |

## Module layout

```
src/qec_generator/
├── __init__.py        # Public API
├── _constants.py      # Shared constants (splits, cases, defaults)
├── config.py          # YAML => Config dataclass
├── sampler.py         # Stim circuit building and sampling
├── graph.py           # DEM => DetectorGraph conversion
├── datasets.py        # Raw => ML-ready shards + global indices
├── utils.py           # I/O helpers, deterministic seeding
└── py.typed           # PEP 561 marker
```
