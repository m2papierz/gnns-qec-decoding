"""QEC Dataset Generator — Generate QEC datasets for GNN-based decoders."""

from constants import CASES
from qec_generator.config import Config, load_config
from qec_generator.datasets import generate_datasets
from qec_generator.graph import DetectorGraph, build_detector_graph
from qec_generator.sampler import build_circuit, generate_for_setting, generate_raw_data


__all__ = [
    # Configuration
    "Config",
    "load_config",
    # Constants
    "CASES",
    # Graph construction
    "DetectorGraph",
    "build_detector_graph",
    "build_circuit",
    # Data generation
    "generate_for_setting",
    "generate_raw_data",
    "generate_datasets",
]
