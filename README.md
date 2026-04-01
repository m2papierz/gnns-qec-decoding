# gnns-qec-decoding

An end-to-end project on decoding topological quantum error-correcting codes with Graph Neural Networks (GNNs).

> [!IMPORTANT]
> **Learning project** - built to deepen hands-on understanding of GNN-based QEC decoding, detector graph construction, and the interplay between classical decoders (MWPM, BP+OSD) and learned models. Simplifications and approximations often appear in the design; this is not a production decoder.

## Background

Physical qubits are fragile, environmental noise introduces random bit-flip, phase-flip, and depolarisation errors. Quantum error correction (QEC) protects a **logical qubit** by encoding it across many physical qubits arranged on a lattice and periodically measuring parity checks called **stabilisers**.

**Surface code** is the leading QEC scheme. Physical qubits sit on a 2-D grid; stabiliser measurements produce a binary **syndrome** that reveals *where* errors likely occurred without collapsing the encoded quantum state. Three parameters govern the setup:

| Parameter | Meaning |
|-----------|---------|
| distance `d` | Grid size (d×d). More redundancy => better protection, but more qubits. |
| rounds `r` | How many times stabilisers are measured. Measurements themselves are noisy, so repetition helps. |
| error prob `p` | Physical error rate per operation. |

A **decoder** takes the syndrome and infers whether the logical qubit has been corrupted (an **observable flip**). The fraction of shots where the decoder is wrong is the **logical error rate (LER)** - the single metric that matters.

### Error-correction threshold

Below a critical physical error rate (the **threshold**, ~0.5–1 % for circuit-level depolarising noise), increasing `d` exponentially suppresses LER. Above threshold, larger codes perform *worse* because more qubits mean more uncorrectable errors.

### From syndromes to graphs

Stim's detector error model (DEM) describes which physical faults trigger which detectors and which flip the logical observable. This maps naturally to a **graph**: nodes are detectors, edges connect detectors that a single fault can trigger simultaneously, and edge weights encode the fault likelihood. This graph is the input to both classical decoders and our GNN.

### Decoding approaches

**MWPM (Minimum-Weight Perfect Matching)** is the standard classical decoder. It pairs up triggered detectors (or pairs them with the boundary) to find the lowest-weight explanation of the syndrome, then reads off whether the logical observable flipped. PyMatching implements this efficiently.

**GNN decoders** operate on the same detector graph but can learn richer error structure. This project explores two training modes:

| Mode | What the GNN predicts | Why |
|------|----------------------|-----|
| `direct` | Observable flip directly from the graph + syndrome | Simplest end-to-end approach |
| `edge` | Per-edge error probabilities fed into a downstream decoder | Can outperform MWPM by learning a better noise model |

The `edge` model is trained against BP marginals (soft labels from belief propagation). At evaluation time, its per-edge probabilities can be fed into either **MWPM** (PyMatching) or **BP+OSD** (NVIDIA CUDA-Q) for decoding.

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
   uv run pytest
   ```

Code formatting/linting:
```bash
uv run isort .
uv run black .
```

## Dataset generation

Generate surface code datasets for GNN-based QEC decoders:

```bash
# Full pipeline (raw + datasets)
uv run scripts/data_generation.py -c configs/data_generation.yaml

# Raw data only
uv run scripts/data_generation.py --mode raw-only

# Specific cases with verbose logging
uv run scripts/data_generation.py --cases direct edge -v
```

See [`src/qec_generator/README.md`](src/qec_generator/README.md) for
configuration reference, output structure, and Python API.

## MWPM baseline evaluation

Before training any GNN, establish reference LER curves with the classical MWPM decoder:

```bash
# Evaluate on test split (default)
uv run scripts/eval_mwpm.py -c configs/data_generation.yaml

# Multiple splits + JSON export
uv run scripts/eval_mwpm.py --splits test val -o outputs/results/mwpm_baseline.json

# Rebuild circuits from config (if .stim files are missing)
uv run scripts/eval_mwpm.py --regenerate
```

The script reports a per-setting LER table, summary statistics by distance, an estimated error-correction threshold, and automatic sanity checks (LER monotonicity in `p`, scaling with `d` below threshold).

## GNN training and evaluation

Train a GNN decoder and evaluate against the MWPM baseline:

```bash
# Train (start with direct)
uv run scripts/train_gnn.py -c configs/train.yaml

# Override case or backend
uv run scripts/train_gnn.py -c configs/train.yaml --case edge
uv run scripts/train_gnn.py -c configs/train.yaml --backend compiled

# Evaluate direct model with MWPM comparison
uv run scripts/eval_gnn.py --checkpoint outputs/runs/direct/best.pt \
    --baseline outputs/results/mwpm_baseline.json

# Evaluate edge model - GNN weights fed into MWPM
uv run scripts/eval_gnn.py --checkpoint outputs/runs/edge/best.pt \
    --decoder mwpm --baseline outputs/results/mwpm_baseline.json

# Evaluate edge model - GNN weights fed into BP+OSD (CUDA-Q)
uv run scripts/eval_gnn.py --checkpoint outputs/runs/edge/best.pt \
    --decoder bp_osd --baseline outputs/results/bp_osd_baseline.json
```

See [`src/gnn/README.md`](src/gnn/README.md) for architecture details,
hyperparameters, training modes, and evaluation protocols.

## Deployment and benchmarking

Train all cases:

```bash
uv run scripts/train_all_cases.py -v
```

Then benchmark inference across backends:

```bash
# All backends (requires torch-tensorrt for TRT)
uv run src/deploy/export_trt.py --checkpoint outputs/runs/direct/best.pt

# PyTorch and compiled only
uv run src/deploy/export_trt.py --checkpoint outputs/runs/direct/best.pt \
    --backends pytorch compiled

# Custom batch size
uv run src/deploy/export_trt.py --checkpoint outputs/runs/edge/best.pt \
    --n-graphs 8 --n-iters 200
```

The TensorRT backend uses `torch.compile` with the `torch_tensorrt` backend, which automatically partitions the GNN: dense subgraphs (MLP, Linear, LayerNorm) are lowered to TRT engines, while sparse ops (scatter, gather) remain in PyTorch.

See [`src/deploy/README.md`](src/deploy/README.md) for Python API, benchmark output format, and details on TRT graph partitioning.

See [`src/kernels/README.md`](src/kernels/README.md) for custom CUDA kernels used with the `cuda` compute backend (inference only).

## Benchmarks and plots

After training and evaluation, benchmark inference and generate figures:

```bash
# Inference benchmarks (latency, throughput, memory)
uv run scripts/benchmark_all.py -v

# Generate all plots (requires: pip install matplotlib)
uv run scripts/plot_results.py -v

# Eval-only plots (no benchmark data needed)
uv run scripts/plot_results.py --no-benchmark -v
```

See [`src/benchmarks/README.md`](src/benchmarks/README.md) for report format, plot descriptions, and the full pipeline from training to figures.
