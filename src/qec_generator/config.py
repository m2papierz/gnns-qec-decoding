"""Configuration management for QEC dataset generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import yaml

from qec_generator._constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESS


logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration for generating surface code datasets.

    Parameters
    ----------
    family : str
        Stim surface code family (e.g., ``"rotated_memory_x"``).
    distances : list[int]
        Code distances to generate.
    rounds : list[int]
        Stabilizer measurement round counts.  Each ``(distance, rounds,
        error_prob)`` combination becomes one setting.  More rounds means
        a longer temporal axis in the detector graph.
    error_probs : list[float]
        Physical error probabilities.
    num_samples : Dict[str, int]
        Samples per split: ``{"train": N, "val": N, "test": N}``.
    noise : Dict[str, Any]
        Stim noise parameters.  Use ``"p"`` for error_prob substitution.
    raw_data_dir : Path
        Output directory for raw sampled data.
    datasets_dir : Path
        Output directory for processed datasets.
    compress : bool
        Use compressed numpy format.
    chunk_size : int
        Samples per Stim sampler call.
    seed : int or None
        Global RNG seed for reproducibility.
    """

    family: str
    distances: list[int]
    rounds: list[int]
    error_probs: list[float]
    num_samples: Dict[str, int]
    noise: Dict[str, Any]
    raw_data_dir: Path
    datasets_dir: Path
    compress: bool = DEFAULT_COMPRESS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    seed: int | None = None

    def __post_init__(self) -> None:
        """Resolve paths and validate fields."""
        self.raw_data_dir = Path(self.raw_data_dir).expanduser().resolve()
        self.datasets_dir = Path(self.datasets_dir).expanduser().resolve()

        if not self.distances:
            raise ValueError("At least one distance required")
        if any(d < 2 for d in self.distances):
            raise ValueError("Distances must be >= 2")
        if not self.rounds:
            raise ValueError("At least one round count required")
        if any(r < 1 for r in self.rounds):
            raise ValueError("All round counts must be >= 1")
        if not self.error_probs:
            raise ValueError("At least one error probability required")
        if any(p <= 0 or p >= 1 for p in self.error_probs):
            raise ValueError("Error probabilities must be in (0, 1)")

        logger.debug(
            "Initialized config: family=%s, %d settings",
            self.family,
            len(list(self.iter_settings())),
        )

    def iter_settings(self) -> Iterator[Tuple[int, int, float]]:
        """
        Yield all ``(distance, rounds, error_prob)`` combinations.

        Yields
        ------
        Tuple[int, int, float]
            ``(distance, rounds, error_prob)`` for every setting.
        """
        for d in self.distances:
            for r in self.rounds:
                for p in self.error_probs:
                    yield d, r, p

    def setting_dir(self, distance: int, rounds: int, p: float) -> Path:
        """
        Return the output directory for a specific setting.

        Parameters
        ----------
        distance : int
            Code distance.
        rounds : int
            Number of rounds.
        p : float
            Error probability.

        Returns
        -------
        Path
            Directory path: ``raw_data_dir/d{distance}_r{rounds}_p{p}``.
        """
        p_tag = f"p{p:.6g}".replace(".", "_")
        return self.raw_data_dir / f"d{distance}_r{rounds}_{p_tag}"

    def resolve_noise(self, p: float) -> dict[str, float]:
        """
        Resolve noise parameters, substituting ``"p"`` with *p*.

        Parameters
        ----------
        p : float
            Error probability to substitute.

        Returns
        -------
        dict[str, float]
            Noise parameters with ``"p"`` replaced by actual value.
        """
        return {
            k: float(p) if (isinstance(v, str) and v.lower() == "p") else float(v)
            for k, v in self.noise.items()
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file.

        Returns
        -------
        Config
            Parsed configuration.

        Raises
        ------
        FileNotFoundError
            If config file does not exist.
        ValueError
            If YAML structure is invalid.
        KeyError
            If required fields are missing.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config not found: {path}")

        logger.info("Loading config from %s", path)

        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")

        sc = raw.get("surface_code", {})
        out = raw.get("output", {})
        gen = raw.get("generation", {})

        return cls(
            family=sc["family"],
            distances=list(sc["distances"]),
            rounds=list(sc["rounds"]),
            error_probs=list(sc["error_probs"]),
            num_samples={k: int(v) for k, v in sc["num_samples"].items()},
            noise=dict(sc.get("noise", {})),
            raw_data_dir=Path(out["raw_data_dir"]),
            datasets_dir=Path(out["datasets_dir"]),
            compress=bool(out.get("compress", DEFAULT_COMPRESS)),
            chunk_size=int(gen.get("chunk_size", DEFAULT_CHUNK_SIZE)),
            seed=int(gen["seed"]) if gen.get("seed") is not None else None,
        )


def load_config(path: str | Path) -> Config:
    """
    Load a :class:`Config` from YAML.

    Parameters
    ----------
    path : str or Path
        Path to YAML configuration.

    Returns
    -------
    Config
        Parsed configuration.
    """
    return Config.from_yaml(path)
