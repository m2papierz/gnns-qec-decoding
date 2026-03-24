"""Project-level constants for QEC dataset generation."""

from typing import Literal


# Dataset splits
SPLITS = ("train", "val", "test")
Split = Literal["train", "val", "test"]

# Dataset cases/variants
CASES = ("direct", "edge")
Case = Literal["direct", "edge"]

# File extensions
NPY_EXT = ".npy"
NPZ_EXT = ".npz"
JSON_EXT = ".json"
STIM_EXT = ".stim"
DEM_EXT = ".dem"

# Default values
DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_COMPRESS = True
