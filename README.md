# gnns-qec-decoding

An end-to-end project on decoding topological quantum error-correcting codes with Graph Neural Networks (GNNs).

## Background

Physical qubits are fragile, environmental noise introduces random bit-flip, phase-flip, and depolarisation errors. Quantum error correction (QEC) protects a **logical qubit** by encoding it across many physical qubits arranged on a lattice and periodically measuring parity checks called **stabilisers**.

**Surface code** is the leading QEC scheme. Physical qubits sit on a 2-D grid; stabiliser measurements produce a binary **syndrome** that reveals *where* errors likely occurred without collapsing the encoded quantum state. Three parameters govern the setup:

| Parameter | Meaning |
|-----------|---------|
| distance `d` | Grid size (d×d). More redundancy → better protection, but more qubits. |
| rounds `r` | How many times stabilisers are measured. Measurements themselves are noisy, so repetition helps. |
| error prob `p` | Physical error rate per operation. |

A **decoder** takes the syndrome and infers whether the logical qubit has been corrupted (an **observable flip**). The fraction of shots where the decoder is wrong is the **logical error rate (LER)** - the single metric that matters.

### Error-correction threshold

Below a critical physical error rate (the **threshold**, ~0.5–1 % for circuit-level depolarising noise), increasing `d` exponentially suppresses LER. Above threshold, larger codes perform *worse* because more qubits mean more uncorrectable errors.

### From syndromes to graphs

Stim's detector error model (DEM) describes which physical faults trigger which detectors and which flip the logical observable. This maps naturally to a **graph**: nodes are detectors, edges connect detectors that a single fault can trigger simultaneously, and edge weights encode the fault likelihood. This graph is the input to both classical decoders and our GNN.

### Decoding approaches

**MWPM (Minimum-Weight Perfect Matching)** is the standard classical decoder. It pairs up triggered detectors (or pairs them with the boundary) to find the lowest-weight explanation of the syndrome, then reads off whether the logical observable flipped. PyMatching implements this efficiently.

**GNN decoders** operate on the same detector graph but can learn richer error structure. This project explores three modes:

| Mode | What the GNN predicts | Why |
|------|----------------------|-----|
| `logical_head` | Observable flip directly from the graph + syndrome | Simplest end-to-end approach |
| `mwpm_teacher` | Which edges MWPM selected (distillation) | Faster inference with comparable accuracy |
| `hybrid` | New edge weights fed back into MWPM | Can outperform MWPM by learning a better noise model |

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
uv run ruff check --fix .
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
uv run scripts/data_generation.py --cases logical_head hybrid -v
```

See [`src/qec_generator/README.md`](src/qec_generator/README.md) for
configuration reference, output structure, and Python API.

## MWPM baseline evaluation

Before training any GNN, establish reference LER curves with the classical MWPM decoder:

```bash
# Evaluate on test split (default)
uv run scripts/eval_mwpm.py -c configs/data_generation.yaml

# Multiple splits + JSON export
uv run scripts/eval_mwpm.py --splits test val -o results/mwpm_baseline.json

# Rebuild circuits from config (if .stim files are missing)
uv run scripts/eval_mwpm.py --regenerate
```

The script reports a per-setting LER table, summary statistics by distance, an estimated error-correction threshold, and automatic sanity checks (LER monotonicity in `p`, scaling with `d` below threshold).

## GNN training and evaluation

Train a GNN decoder and evaluate against the MWPM baseline:

```bash
# Train (start with logical_head)
uv run python -m gnn.train --case logical_head --epochs 50

# Evaluate with MWPM comparison
uv run python -m gnn.eval --checkpoint runs/logical_head/best.pt \
    --baseline results/mwpm_baseline.json
```

See [`src/gnn/README.md`](src/gnn/README.md) for architecture details, hyperparameters, all three training modes, and evaluation protocols.
