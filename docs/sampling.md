# sampling — Surface Code Sampling & Graph Construction

Stim circuit sampling and detector graph construction for GNN-based QEC decoders.

The module wraps [Stim](https://github.com/quantumlib/Stim) circuit
simulation and [PyMatching](https://pymatching.readthedocs.io/) decoding
into a reproducible sampling pipeline.

## Concepts

### Surface code memory experiment

`stim.Circuit.generated("surface_code:rotated_memory_x", ...)` creates a *rotated surface code memory-X experiment*: the logical qubit is initialised and measured in the X basis.  During the experiment, stabiliser measurements run for a configurable number of **rounds** (syndrome extraction cycles).

### Detector error model (DEM)

A DEM captures which physical faults trigger which *detectors* (differences between consecutive stabiliser outcomes) and which flip *observables* (the logical measurement).  The module calls `circuit.detector_error_model(decompose_errors=True)` to obtain a graph-like DEM suitable for matching.

### Detector / decoding graph

PyMatching converts the DEM into a weighted graph where nodes are detectors and edges carry error probabilities and MWPM weights.  The module adds a single **virtual boundary node** (index = `num_detectors`) that collapses all boundary edges, making the graph structure stable across settings.

Per-edge **observable flip masks** are extracted from the DEM and stored alongside the graph.  These indicate which logical observables are flipped when a given edge's error occurs — required by BP+OSD decoding.

### Labels

| Label | Source | Use case |
|---|---|---|
| `logical` | Stim simulation ground truth | Direct logical-error prediction |
| `bp_soft_labels` | Sum-product belief propagation marginals | Edge model supervision (continuous P(e=1 \| syndrome) per edge) |

### Rounds

`rounds` controls how many stabiliser measurement cycles run in each memory experiment. More rounds => longer temporal axis => more detectors and edges.

## Quick start

```bash
# Full pipeline (raw samples + packaged datasets)
uv run scripts/data_generation.py -c configs/data_generation.yaml

# Raw data only
uv run scripts/data_generation.py --mode raw-only

# Specific cases with verbose logging
uv run scripts/data_generation.py --cases direct edge -v

# Overwrite existing data
uv run scripts/data_generation.py --overwrite
```

### Python API

```python
from sampling.sampler import CircuitSetting, WorkerSampler
from sampling.graph import build_fired_detector_graph, extract_circuit_metadata
```

## Configuration (YAML)

```yaml
surface_code:
  family: "rotated_memory_x"
  distances: [3, 5, 7]
  error_probs: [0.001, 0.005, 0.01]

  rounds: [3, 5, 7]

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
    │   ├── observable_flips.npy  # (E, num_obs) bool — per-edge observable mask
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
│       │   ├── edge_index.npy          # (2, E) int64
│       │   ├── edge_attr.npy           # (E, 2) float32 [error_prob, weight]
│       │   ├── node_coords.npy         # (N, C) float32
│       │   ├── node_is_boundary.npy    # (N,) bool
│       │   ├── observable_flips.npy    # (E, num_obs) bool
│       │   └── graph_meta.json
│       ├── {split}_syndrome.npy
│       ├── {split}_logical.npy
│       ├── {split}_bp_soft_labels.npy  # edge only — (S, U) float32
│       └── mwpm/                       # edge only
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
| `direct` | `logical` | Train model to predict logical errors directly |
| `edge` | `logical` + BP marginals (soft labels) | GNN learns per-edge error probabilities fed back into a decoder |

## Module layout

```
src/sampling/
├── __init__.py
├── graph.py           # DEM => DetectorGraph conversion (incl. observable_flips)
├── sampler.py         # Stim circuit building and sampling
└── seeding.py         # Deterministic BLAKE2b seed derivation
```
