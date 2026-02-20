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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

from gnn.dataset import MixedSurfaceCodeDataset
from gnn.eval import ValLEREstimator
from gnn.models.heads import Case, QECDecoder, build_model


logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────


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
        Peak learning rate (after warmup).
    weight_decay : float
        AdamW weight decay.
    epochs : int
        Number of training epochs.
    batch_size : int
        Graphs per training batch.
    num_workers : int
        DataLoader worker processes.
    edge_loss : str
        Loss for edge cases: ``"focal"`` or ``"bce"``.
    focal_gamma : float
        Focal loss focusing parameter.
    edge_pos_weight : float or None
        Positive-class weight (alpha for focal, pos_weight for BCE).
        If None, estimated from training data.
    warmup_epochs : int
        Linear LR warmup duration.
    ler_every : int
        Compute validation LER every N epochs.  For ``logical_head``
        this is cheap; for edge cases it runs PyMatching.
    ler_max_shots : int
        Max val shots per setting for LER estimation.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    patience : int
        Early stopping patience (LER evaluations without improvement).
        Set to 0 to disable.
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
    edge_loss: str = "focal"
    focal_gamma: float = 2.0
    edge_pos_weight: float | None = None
    warmup_epochs: int = 5
    ler_every: int = 5
    ler_max_shots: int = 500
    max_grad_norm: float = 1.0
    patience: int = 10
    seed: int = 42
    resume: Path | None = None
    max_samples: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load configuration from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        flat: Dict[str, Any] = {}

        # Top-level scalars
        for key in (
            "case",
            "datasets_dir",
            "output_dir",
            "num_workers",
            "edge_loss",
            "focal_gamma",
            "edge_pos_weight",
            "warmup_epochs",
            "ler_every",
            "ler_max_shots",
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


# ── Utilities ─────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
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
    Estimate positive-class weight from training label distribution.

    Returns ``num_neg / num_pos``, clamped to ``[1, 200]``.
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
        "Edge label balance: %d pos / %d neg → raw=%.1f, clamped=%.1f",
        total_pos,
        total_neg,
        raw,
        clamped,
    )
    return clamped


# ── Loss ──────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification with extreme class imbalance.W

    Standard choice for foreground/background ratios ~1:100+.
    Down-weights easy negatives so gradient signal is dominated by
    hard positives.

    Parameters
    ----------
    alpha : float
        Weight for the positive class.
    gamma : float
        Focusing parameter.  ``gamma=0`` recovers weighted BCE.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * 1.0
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * ce).mean()


def build_criterion(
    case: Case,
    edge_loss: str = "focal",
    focal_gamma: float = 2.0,
    pos_weight: float | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Build loss function for a given training case."""
    if case == "logical_head":
        return nn.BCEWithLogitsLoss()

    alpha = pos_weight if pos_weight is not None else 1.0

    if edge_loss == "focal":
        logger.info("Using FocalLoss(alpha=%.1f, gamma=%.1f)", alpha, focal_gamma)
        return FocalLoss(alpha=alpha, gamma=focal_gamma)

    # BCE fallback
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.BCEWithLogitsLoss()


# ── Dataset validation ────────────────────────────────────────────────


def _validate_dataset(ds: MixedSurfaceCodeDataset, label: str) -> None:
    """Smoke-test dataset by loading first sample; raises on bad shapes."""
    sample = ds[0]
    if sample.x.ndim != 2 or sample.x.shape[1] != 1:
        raise ValueError(
            f"{label} dataset: expected x shape (N, 1), got {tuple(sample.x.shape)}"
        )
    if sample.edge_attr.ndim != 2 or sample.edge_attr.shape[1] != 2:
        raise ValueError(
            f"{label} dataset: expected edge_attr shape (E, 2), "
            f"got {tuple(sample.edge_attr.shape)}"
        )


# ── Training / validation steps ──────────────────────────────────────


def train_one_epoch(
    model: QECDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    case: Case,
    max_grad_norm: float = 1.0,
) -> float:
    """Run one training epoch.  Returns mean loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        if case == "logical_head":
            loss = criterion(logits.view(-1), batch.y)
        else:
            loss = criterion(logits, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: QECDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    case: Case,
) -> float:
    """Run validation.  Returns mean loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)

        if case == "logical_head":
            loss = criterion(logits.view(-1), batch.y)
        else:
            loss = criterion(logits, batch.y)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ── Checkpoint management ────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    model: QECDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    best_ler: float,
    config: TrainConfig,
) -> None:
    """Save training state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)
    config_dict["datasets_dir"] = str(config.datasets_dir)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["resume"] = str(config.resume) if config.resume else None

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
            ),
            "best_metric": best_ler,
            "config": config_dict,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: QECDecoder,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
) -> Dict[str, Any]:
    """Load training state from disk."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if (
        scheduler is not None
        and hasattr(scheduler, "load_state_dict")
        and ckpt.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "best_metric": ckpt.get("best_metric", float("inf")),
        "config": ckpt.get("config", {}),
    }


def _validate_resume_config(cfg: TrainConfig, ckpt_cfg: Dict[str, Any]) -> None:
    """Raise ValueError if checkpoint architecture doesn't match."""
    for key in ("case", "hidden_dim", "num_layers"):
        current = getattr(cfg, key)
        saved = ckpt_cfg.get(key)
        if saved is not None and saved != current:
            raise ValueError(
                f"Config mismatch on resume: {key}={current}, "
                f"checkpoint has {key}={saved}"
            )


# ── Main training function ───────────────────────────────────────────


def train(cfg: TrainConfig) -> Path:
    """
    Execute the full training pipeline.

    Returns the path to the best model checkpoint (selected on val LER).
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

    _validate_dataset(train_ds, "train")
    _validate_dataset(val_ds, "val")

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

    # ── LER monitor ──────────────────────────────────────────────

    ler_monitor = ValLEREstimator(
        datasets_dir=cfg.datasets_dir,
        case=cfg.case,
        max_shots_per_setting=cfg.ler_max_shots,
        batch_size=cfg.batch_size * 4,  # inference can use larger batches
    )

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

    # ── Optimizer + scheduler (warmup → cosine) ──────────────────

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    warmup = cfg.warmup_epochs
    cosine_epochs = max(cfg.epochs - warmup, 1)
    eta_min = cfg.lr / 50

    warmup_sched = LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=warmup,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup],
    )

    logger.info(
        "Schedule: %d warmup → cosine(%d, eta_min=%.1e)",
        warmup,
        cosine_epochs,
        eta_min,
    )

    # ── Loss ──────────────────────────────────────────────────────

    pos_weight = cfg.edge_pos_weight
    if cfg.case != "logical_head" and pos_weight is None:
        pos_weight = _estimate_edge_pos_weight(train_loader)

    criterion = build_criterion(
        cfg.case,
        edge_loss=cfg.edge_loss,
        focal_gamma=cfg.focal_gamma,
        pos_weight=pos_weight,
        device=device,
    )

    # ── Resume ────────────────────────────────────────────────────

    start_epoch = 0
    best_ler = float("inf")

    if cfg.resume is not None:
        ckpt_info = load_checkpoint(cfg.resume, model, optimizer, scheduler)
        _validate_resume_config(cfg, ckpt_info["config"])
        start_epoch = ckpt_info["epoch"] + 1
        best_ler = ckpt_info["best_metric"]
        logger.info("Resuming from epoch %d (best LER=%.6f)", start_epoch, best_ler)

    # ── Save config ───────────────────────────────────────────────

    config_dict = asdict(cfg)
    config_dict["datasets_dir"] = str(cfg.datasets_dir)
    config_dict["output_dir"] = str(cfg.output_dir)
    config_dict["resume"] = str(cfg.resume) if cfg.resume else None
    (run_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )

    # ── Training loop ─────────────────────────────────────────────

    best_path = run_dir / "best.pt"
    history: list[Dict[str, Any]] = []
    evals_without_improvement = 0
    last_ler: float | None = None

    logger.info("Training %d epochs (LER eval every %d)", cfg.epochs, cfg.ler_every)

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg.case,
            cfg.max_grad_norm,
        )
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            cfg.case,
        )

        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        # ── Periodic LER evaluation ──────────────────────────────

        do_ler = (
            epoch == start_epoch
            or (epoch + 1) % cfg.ler_every == 0
            or epoch == cfg.epochs - 1
        )

        improved = False
        if do_ler:
            t_ler = time.perf_counter()
            current_ler = ler_monitor.compute(model, device)
            ler_time = time.perf_counter() - t_ler
            elapsed += ler_time
            last_ler = current_ler

            if current_ler < best_ler:
                best_ler = current_ler
                improved = True
                evals_without_improvement = 0
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_ler,
                    cfg,
                )
            else:
                evals_without_improvement += 1

        # ── Log ──────────────────────────────────────────────────

        ler_str = f"  LER={last_ler:.6f}" if last_ler is not None else ""
        marker = " *" if improved else ""

        logger.info(
            "Epoch %3d/%d [%.1fs, lr=%.1e]  " "train_loss=%.4f  val_loss=%.4f%s%s",
            epoch + 1,
            cfg.epochs,
            elapsed,
            lr,
            train_loss,
            val_loss,
            ler_str,
            marker,
        )

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "elapsed_s": round(elapsed, 2),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ler": last_ler,
                "best": improved,
            }
        )

        # ── Early stopping (counts LER evals, not epochs) ────────

        if cfg.patience > 0 and do_ler and evals_without_improvement >= cfg.patience:
            logger.info(
                "Early stopping at epoch %d " "(%d LER evals without improvement)",
                epoch + 1,
                cfg.patience,
            )
            break

    # ── Save history ──────────────────────────────────────────────

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    logger.info("Training complete. Best LER=%.6f", best_ler)
    logger.info("Best checkpoint: %s", best_path)

    return best_path
