"""Project-level constants for QEC dataset generation."""

from typing import Literal


# Dataset splits
SPLITS = ("train", "val", "test")
Split = Literal["train", "val", "test"]

# Dataset cases/variants
CASES = ("logical_head", "mwpm_teacher", "hybrid")
Case = Literal["logical_head", "mwpm_teacher", "hybrid"]

# File extensions
NPY_EXT = ".npy"
NPZ_EXT = ".npz"
JSON_EXT = ".json"
STIM_EXT = ".stim"
DEM_EXT = ".dem"

# MWPM configuration
MWPM_BITORDER = "little"
MWPM_LABEL = "edge_selected"

# Default values
DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_COMPRESS = True
