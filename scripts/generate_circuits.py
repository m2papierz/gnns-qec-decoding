"""Generate and commit Stim circuit files for all training/eval settings.

Settings: d∈{3,5,7}, r=d, p∈{0.003, 0.005, 0.008, 0.01}.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import stim


logger = logging.getLogger(__name__)

DISTANCES: list[int] = [3, 5, 7]
ERROR_PROBS: list[float] = [0.003, 0.005, 0.008, 0.01]
FAMILY: str = "rotated_memory_x"

CIRCUITS_DIR: Path = Path("data/circuits")


def circuit_filename(distance: int, rounds: int, p: float) -> str:
    """Return canonical circuit filename for a setting.

    Parameters
    ----------
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    p : float
        Physical error probability.

    Returns
    -------
    str
        Filename like ``d3_r3_p0_003.stim``.
    """
    p_tag = f"p{p:.6g}".replace(".", "_")
    return f"d{distance}_r{rounds}_{p_tag}.stim"


def generate_circuit(distance: int, rounds: int, p: float) -> stim.Circuit:
    """Build a Stim surface code circuit for given parameters.

    Parameters
    ----------
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    p : float
        Physical error probability.

    Returns
    -------
    stim.Circuit
        Generated circuit.
    """
    return stim.Circuit.generated(
        f"surface_code:{FAMILY}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=CIRCUITS_DIR,
        help="Output directory for circuit files (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing circuit files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for d in DISTANCES:
        r = d  # rounds match distance for training
        for p in ERROR_PROBS:
            fname = circuit_filename(d, r, p)
            path = out_dir / fname

            if path.exists() and not args.overwrite:
                logger.info("Exists, skipping: %s", path)
                continue

            circuit = generate_circuit(d, r, p)
            dem = circuit.detector_error_model(decompose_errors=True)

            circuit.to_file(path)
            generated += 1
            logger.info(
                "Generated %s  (detectors=%d, observables=%d)",
                fname,
                dem.num_detectors,
                dem.num_observables,
            )

    logger.info(
        "Done: %d circuits generated in %s (stim==%s)",
        generated,
        out_dir,
        stim.__version__,
    )


if __name__ == "__main__":
    main()
