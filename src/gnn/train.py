"""
Training loop for GNN-based QEC decoders.

Supports all three training cases:

- ``logical_head``: graph-level binary classification of observable flips.
- ``mwpm_teacher``: per-edge imitation of MWPM edge selections.
- ``hybrid``: same architecture as ``mwpm_teacher`` (different eval protocol).
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from gnn.dataset import MixedSurfaceCodeDataset
from gnn.models.heads import Case, QECDecoder, build_model


logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes
    ----------
    case : str
        Training case: ``"logical_head"``, ``"mwpm_teacher"``, or
        ``"hybrid"``.
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
        Initial learning rate.
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
        Early stopping patience (epochs without improvement).
        Set to 0 to disable early stopping.
    seed : int
        Random seed for reproducibility.
    resume : Path or None
        Path to checkpoint to resume from.
    max_samples : int or None
        Cap on training samples (validation capped at ``max_samples // 5``).
        If None, use all available data.
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
    seed: int = 42
    resume: Path | None = None
    max_samples: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """
        Load configuration from a YAML file.

        YAML keys are mapped to dataclass fields.  Nested sections
        (``model``, ``optimisation``) are flattened.  Any field not
        present in the file keeps its default value.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        TrainConfig
            Parsed configuration.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        flat: Dict[str, Any] = {}

        # Top-level scalars
        for key in (
            "case",
            "datasets_dir",
            "output_dir",
            "num_workers",
            "edge_pos_weight",
            "max_grad_norm",
            "patience",
            "seed",
            "max_samples",
        ):
            if key in raw:
                flat[key] = raw[key]

        # Nested: model.*
        model = raw.get("model", {})
        for key in ("hidden_dim", "num_layers", "dropout"):
            if key in model:
                flat[key] = model[key]

        # Nested: optimisation.*
        optim = raw.get("optimisation", {})
        for key in ("lr", "weight_decay", "epochs", "batch_size"):
            if key in optim:
                flat[key] = optim[key]

        # Path coercion
        for key in ("datasets_dir", "output_dir"):
            if key in flat:
                flat[key] = Path(flat[key])

        return cls(**flat)


def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _estimate_edge_pos_weight(
    loader: DataLoader,
    max_batches: int = 50,
) -> float:
    """
    Estimate positive-class weight for edge BCE from training data.

    Scans a subset of the training data and computes the ratio of
    negative to positive edge labels: ``num_neg / num_pos``.

    Parameters
    ----------
    loader : DataLoader
        Training data loader.
    max_batches : int
        Maximum number of batches to scan.

    Returns
    -------
    float
        Estimated ``pos_weight``.  Clamped to ``[1.0, 200.0]``.
    """
    total_pos = 0
    total_neg = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        labels = batch.y
        pos = int(labels.sum().item())
        total_pos += pos
        total_neg += int(labels.numel()) - pos

    if total_pos == 0:
        logger.warning("No positive edge labels found; using pos_weight=10.0")
        return 10.0

    raw = total_neg / total_pos
    clamped = float(np.clip(raw, 1.0, 200.0))
    logger.info(
        "Edge label balance: %d pos / %d neg → raw=%.1f, pos_weight=%.1f",
        total_pos,
        total_neg,
        raw,
        clamped,
    )
    return clamped


def build_criterion(
    case: Case,
    pos_weight: float | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Build the loss function for a given training case.

    Parameters
    ----------
    case : str
        Training case.
    pos_weight : float or None
        Positive-class weight for edge BCE.  Ignored for
        ``logical_head``.
    device : torch.device or None
        Device for the pos_weight tensor.

    Returns
    -------
    nn.Module
        Loss function (``BCEWithLogitsLoss``).
    """
    if case == "logical_head":
        return nn.BCEWithLogitsLoss()

    # mwpm_teacher / hybrid: imbalanced edge labels
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.BCEWithLogitsLoss()


def _validate_dataset(ds: MixedSurfaceCodeDataset, label: str) -> None:
    """Smoke-test dataset by loading first sample; raises on bad shapes."""
    sample = ds[0]
    if sample.x.ndim != 2 or sample.x.shape[1] != 1:
        raise ValueError(
            f"{label} dataset: expected x shape (N, 1), got {tuple(sample.x.shape)}"
        )
    if sample.edge_index.shape[0] != 2:
        raise ValueError(
            f"{label} dataset: expected edge_index shape (2, E), "
            f"got {tuple(sample.edge_index.shape)}"
        )
    if sample.edge_attr.ndim != 2:
        raise ValueError(
            f"{label} dataset: expected edge_attr shape (E, D), "
            f"got {tuple(sample.edge_attr.shape)}"
        )
    if sample.y.ndim < 1:
        raise ValueError(f"{label} dataset: y is scalar, expected ≥1D")
    logger.debug(
        "%s dataset validated: x=%s, edges=%d, y=%s",
        label,
        tuple(sample.x.shape),
        sample.edge_index.shape[1],
        tuple(sample.y.shape),
    )


def train_one_epoch(
    model: QECDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    case: Case,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Parameters
    ----------
    model : QECDecoder
        Model to train.
    loader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function.
    optimizer : Optimizer
        Parameter optimizer.
    device : torch.device
        Compute device.
    case : str
        Training case (determines how predictions map to targets).
    max_grad_norm : float
        Maximum gradient norm for clipping.

    Returns
    -------
    dict
        Training metrics: ``loss``, and for ``logical_head`` also
        ``ler`` (logical error rate).
    """
    model.train()

    total_loss = 0.0
    num_batches = 0
    # logical_head LER tracking
    total_graphs = 0
    total_errors = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)

        if case == "logical_head":
            # logits: (B, num_obs), batch.y: (B * num_obs,)
            target = batch.y
            loss = criterion(logits.view(-1), target)

            # LER: any observable wrong → logical error
            with torch.no_grad():
                pred = (logits > 0.0).float()  # (B, num_obs)
                target_2d = target.view_as(pred)
                total_graphs += pred.shape[0]
                total_errors += int((pred != target_2d).any(dim=1).sum().item())
        else:
            # logits: (E_total,), batch.y: (E_total,)
            loss = criterion(logits, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    metrics: Dict[str, float] = {"loss": total_loss / max(num_batches, 1)}

    if case == "logical_head" and total_graphs > 0:
        metrics["ler"] = total_errors / total_graphs

    return metrics


@torch.no_grad()
def validate(
    model: QECDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    case: Case,
) -> Dict[str, float]:
    """Run validation and compute metrics.

    Parameters
    ----------
    model : QECDecoder
        Model to evaluate.
    loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    case : str
        Training case.

    Returns
    -------
    dict
        Validation metrics: ``loss``, and for ``logical_head`` also
        ``ler``.  For edge cases, includes ``edge_acc`` (per-edge
        accuracy).
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0
    total_graphs = 0
    total_errors = 0
    # Edge accuracy tracking
    edge_correct = 0
    edge_total = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)

        if case == "logical_head":
            target = batch.y
            loss = criterion(logits.view(-1), target)

            pred = (logits > 0.0).float()  # (B, num_obs)
            target_2d = target.view_as(pred)
            total_graphs += pred.shape[0]
            total_errors += int((pred != target_2d).any(dim=1).sum().item())
        else:
            loss = criterion(logits, batch.y)

            pred = (logits > 0.0).float()
            edge_correct += int((pred == batch.y).sum().item())
            edge_total += int(batch.y.numel())

        total_loss += loss.item()
        num_batches += 1

    metrics: Dict[str, float] = {"loss": total_loss / max(num_batches, 1)}

    if case == "logical_head" and total_graphs > 0:
        metrics["ler"] = total_errors / total_graphs
    if case != "logical_head" and edge_total > 0:
        metrics["edge_acc"] = edge_correct / edge_total

    return metrics


def save_checkpoint(
    path: Path,
    model: QECDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    best_metric: float,
    config: TrainConfig,
) -> None:
    """
    Save a training checkpoint.

    Parameters
    ----------
    path : Path
        Output file path.
    model : QECDecoder
        Model to save.
    optimizer : Optimizer
        Optimizer state.
    scheduler : object
        LR scheduler (must have ``state_dict``).
    epoch : int
        Current epoch number.
    best_metric : float
        Best validation metric so far.
    config : TrainConfig
        Training configuration for reproducibility.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),  # type: ignore[attr-defined]
            "best_metric": best_metric,
            "config": asdict(config),
        },
        path,
    )
    logger.debug("Saved checkpoint to %s", path)


def load_checkpoint(
    path: Path,
    model: QECDecoder,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Parameters
    ----------
    path : Path
        Checkpoint file path.
    model : QECDecoder
        Model to load weights into.
    optimizer : Optimizer or None
        If provided, restore optimizer state.
    scheduler : object or None
        If provided, restore scheduler state.

    Returns
    -------
    dict
        Checkpoint metadata: ``epoch``, ``best_metric``, ``config``.
    """
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])  # type: ignore[attr-defined]

    logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
    return {
        "epoch": ckpt.get("epoch", 0),
        "best_metric": ckpt.get("best_metric", float("inf")),
        "config": ckpt.get("config", {}),
    }


def _validate_resume_config(cfg: TrainConfig, ckpt_cfg: Dict[str, Any]) -> None:
    """Raise ValueError if checkpoint architecture doesn't match current config."""
    for key in ("case", "hidden_dim", "num_layers"):
        current = getattr(cfg, key)
        saved = ckpt_cfg.get(key)
        if saved is not None and saved != current:
            raise ValueError(
                f"Config mismatch on resume: {key}={current}, "
                f"checkpoint has {key}={saved}"
            )


def _best_metric_key(case: Case) -> str:
    """Return 'ler' for logical_head, 'loss' otherwise."""
    return "ler" if case == "logical_head" else "loss"


def _format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict as a compact log string."""
    parts = []
    for k, v in metrics.items():
        if k == "loss":
            parts.append(f"loss={v:.4f}")
        elif k == "ler":
            parts.append(f"LER={v:.4f}")
        elif k == "edge_acc":
            parts.append(f"edge_acc={v:.4f}")
        else:
            parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)


def train(cfg: TrainConfig) -> Path:
    """
    Execute the full training pipeline.

    Parameters
    ----------
    cfg : TrainConfig
        Training configuration.

    Returns
    -------
    Path
        Path to the best model checkpoint.
    """
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────

    run_dir = cfg.output_dir / cfg.case
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets from %s (case=%s)", cfg.datasets_dir, cfg.case)

    train_ds = MixedSurfaceCodeDataset(
        datasets_dir=cfg.datasets_dir,
        case=cfg.case,
        split="train",
    )
    val_ds = MixedSurfaceCodeDataset(
        datasets_dir=cfg.datasets_dir,
        case=cfg.case,
        split="val",
    )

    # Fail-fast: validate data shapes before training
    _validate_dataset(train_ds, "train")
    _validate_dataset(val_ds, "val")

    # Optional sample cap for fast iteration
    if cfg.max_samples is not None:
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, range(min(cfg.max_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(cfg.max_samples // 5, len(val_ds))))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    # ── Model ─────────────────────────────────────────────────────

    model = build_model(
        cfg.case,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable parameters", num_params)

    # ── Optimizer + scheduler ─────────────────────────────────────

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    eta_min = cfg.lr / 50
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=eta_min)
    logger.info(
        "Scheduler: CosineAnnealingLR(T_max=%d, eta_min=%.1e)",
        cfg.epochs,
        eta_min,
    )

    # ── Loss ──────────────────────────────────────────────────────

    pos_weight = cfg.edge_pos_weight
    if cfg.case != "logical_head" and pos_weight is None:
        pos_weight = _estimate_edge_pos_weight(train_loader)

    criterion = build_criterion(
        cfg.case,
        pos_weight=pos_weight,
        device=device,
    )

    # ── Resume ────────────────────────────────────────────────────

    start_epoch = 0
    best_metric = float("inf")

    if cfg.resume is not None:
        ckpt_info = load_checkpoint(cfg.resume, model, optimizer, scheduler)
        _validate_resume_config(cfg, ckpt_info["config"])
        start_epoch = ckpt_info["epoch"] + 1
        best_metric = ckpt_info["best_metric"]
        logger.info(
            "Resuming from epoch %d (best_metric=%.6f)",
            start_epoch,
            best_metric,
        )

    # ── Save config ───────────────────────────────────────────────

    config_path = run_dir / "config.json"
    config_dict = asdict(cfg)
    config_dict["datasets_dir"] = str(cfg.datasets_dir)
    config_dict["output_dir"] = str(cfg.output_dir)
    config_dict["resume"] = str(cfg.resume) if cfg.resume else None
    config_path.write_text(
        json.dumps(config_dict, indent=2),
        encoding="utf-8",
    )

    # ── Training loop ─────────────────────────────────────────────

    metric_key = _best_metric_key(cfg.case)
    best_path = run_dir / "best.pt"
    history: list[Dict[str, Any]] = []
    epochs_without_improvement = 0

    logger.info("Starting training: %d epochs", cfg.epochs)

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            case=cfg.case,
            max_grad_norm=cfg.max_grad_norm,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            case=cfg.case,
        )

        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Check improvement
        current = val_metrics[metric_key]
        improved = current < best_metric
        if improved:
            best_metric = current
            epochs_without_improvement = 0
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                config=cfg,
            )
        else:
            epochs_without_improvement += 1

        # Log
        marker = " *" if improved else ""
        logger.info(
            "Epoch %3d/%d [%.1fs, lr=%.1e]  " "train: %s  val: %s%s",
            epoch + 1,
            cfg.epochs,
            elapsed,
            lr,
            _format_metrics(train_metrics),
            _format_metrics(val_metrics),
            marker,
        )

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "elapsed_s": round(elapsed, 2),
                "train": train_metrics,
                "val": val_metrics,
                "best": improved,
            }
        )

        # Early stopping
        if cfg.patience > 0 and epochs_without_improvement >= cfg.patience:
            logger.info(
                "Early stopping at epoch %d (%d epochs without improvement)",
                epoch + 1,
                cfg.patience,
            )
            break

    # ── Save history ──────────────────────────────────────────────

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("Training complete. Best %s=%.6f", metric_key, best_metric)
    logger.info("Best checkpoint: %s", best_path)

    return best_path
