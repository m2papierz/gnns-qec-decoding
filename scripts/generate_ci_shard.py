"""Generate a small frozen CI data shard for integration tests.

Produces pre-sampled syndromes and observable flips for d=3 at a single
error probability, along with detector metadata.  The resulting files
can be loaded by tests without any Stim dependency.

Usage
-----
    uv run python scripts/generate_ci_shard.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import stim

from sampling.graph import extract_circuit_metadata


CIRCUIT_PATH = Path("data/circuits/d3_r3_p0_01.stim")
OUTPUT_DIR = Path("data/ci_shard")
SEED = 20240101
NUM_SHOTS = 256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    circuit = stim.Circuit.from_file(str(CIRCUIT_PATH))
    dem = circuit.detector_error_model(decompose_errors=True)

    sampler = circuit.compile_detector_sampler(seed=SEED)
    syndromes, observables = sampler.sample(
        shots=NUM_SHOTS, separate_observables=True, bit_packed=False
    )
    syndromes = syndromes.astype(np.uint8)
    observables = observables.astype(np.uint8)

    meta = extract_circuit_metadata(circuit, distance=3, rounds=3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "syndromes.npy", syndromes)
    np.save(OUTPUT_DIR / "observables.npy", observables)
    np.save(OUTPUT_DIR / "detector_coords.npy", meta.detector_coords)

    manifest = {
        "circuit_file": str(CIRCUIT_PATH),
        "circuit_sha256": _sha256(CIRCUIT_PATH),
        "stim_version": stim.__version__,
        "seed": SEED,
        "num_shots": NUM_SHOTS,
        "distance": 3,
        "rounds": 3,
        "error_prob": 0.01,
        "num_detectors": int(dem.num_detectors),
        "num_observables": int(dem.num_observables),
        "syndromes_shape": list(syndromes.shape),
        "observables_shape": list(observables.shape),
        "positive_count": int(observables.any(axis=1).sum()),
        "generation_command": "uv run python scripts/generate_ci_shard.py",
    }

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"CI shard written to {OUTPUT_DIR}/")
    print(f"  syndromes: {syndromes.shape}")
    print(f"  observables: {observables.shape}")
    print(f"  positives: {manifest['positive_count']}/{NUM_SHOTS}")


if __name__ == "__main__":
    main()
