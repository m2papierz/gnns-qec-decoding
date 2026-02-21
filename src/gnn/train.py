"""Training configuration for GNN-based QEC decoders."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from constants import Case


@dataclass
class TrainConfig:
    """
    Training hyperparameters.

    Attributes
    ----------
    case : str
        Training case: ``"logical_head"``, ``"mwpm_teacher"``, or ``"hybrid"``.
    datasets_dir : Path
        Root directory of packaged datasets.
    output_dir : Path
        Directory for checkpoints and logs.
    hidden_dim : int
        Encoder hidden dimensionality.
    num_layers : int
        Number of message-passing layers.
    dropout : float
        Dropout probability.
    lr : float
        Peak learning rate (after warmup).
    weight_decay : float
        AdamW weight decay.
    epochs : int
        Number of training epochs.
    batch_size : int
        Graphs per training batch.
    num_workers : int
        DataLoader worker processes.
    edge_pos_weight : float or None
        Positive-class weight for edge BCE loss.  If None, estimated
        from training data.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    patience : int
        Early stopping patience (validation runs without improvement).
        Set to 0 to disable early stopping.
    val_every : int
        Run validation every this many epochs.
    seed : int
        Random seed for reproducibility.
    resume : Path or None
        Path to checkpoint to resume from.
    max_samples : int or None
        Cap on training samples (validation capped at ``max_samples // 5``).
    """

    case: Case = "logical_head"
    datasets_dir: Path = Path("data/datasets")
    output_dir: Path = Path("outputs/runs")
    hidden_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    edge_pos_weight: float | None = None
    max_grad_norm: float = 1.0
    patience: int = 15
    val_every: int = 1
    seed: int = 42
    resume: Path | None = None
    max_samples: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load configuration from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        flat: Dict[str, Any] = {}

        for key in (
            "case",
            "datasets_dir",
            "output_dir",
            "num_workers",
            "edge_pos_weight",
            "max_grad_norm",
            "patience",
            "val_every",
            "seed",
            "max_samples",
        ):
            if key in raw:
                flat[key] = raw[key]

        model = raw.get("model", {})
        for key in ("hidden_dim", "num_layers", "dropout"):
            if key in model:
                flat[key] = model[key]

        optim = raw.get("optimisation", {})
        for key in ("lr", "weight_decay", "epochs", "batch_size"):
            if key in optim:
                flat[key] = optim[key]

        for key in ("datasets_dir", "output_dir"):
            if key in flat:
                flat[key] = Path(flat[key])

        return cls(**flat)

    def __post_init__(self) -> None:
        if self.val_every < 1:
            raise ValueError(f"val_every must be >= 1, got {self.val_every}")


def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
