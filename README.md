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
