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
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import Batch
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
        Early stopping patience (epochs without improvement).
        Set to 0 to disable early stopping.
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
    Estimate positive-class weight for edge BCE from training data.

    Returns ``num_neg / num_pos``, clamped to ``[1.0, 200.0]``.
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


class _GraphNormalizedBCE(nn.Module):
    """Per-graph normalized BCE with logits for edge predictions.

    Computes per-edge BCE, averages within each graph, then averages
    across graphs.  Prevents larger graphs from dominating the
    gradient in mixed-distance batches.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        raw = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        edge_graph = batch.batch[batch.edge_index[0]]
        n_graphs = int(batch.batch.max()) + 1
        graph_loss = torch.zeros(n_graphs, device=logits.device)
        graph_count = torch.zeros(n_graphs, device=logits.device)
        graph_loss.scatter_add_(0, edge_graph, raw)
        graph_count.scatter_add_(0, edge_graph, torch.ones_like(raw))
        return (graph_loss / graph_count.clamp(min=1)).mean()


def build_criterion(
    case: Case,
    pos_weight: float | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Build the loss function for a given training case."""
    if case == "logical_head":
        return nn.BCEWithLogitsLoss()

    pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    return _GraphNormalizedBCE(pos_weight=pw)


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


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    eta_min: float,
) -> tuple[object, str]:
    """Build linear warmup + cosine annealing scheduler."""
    warmup_epochs = max(1, epochs // 20)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
    desc = (
        f"LinearWarmup({warmup_epochs}ep) → "
        f"CosineAnnealing(T_max={epochs - warmup_epochs}, eta_min={eta_min:.1e})"
    )
    return scheduler, desc


def train_one_epoch(
    model: QECDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    case: Case,
    max_grad_norm: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
) -> Dict[str, float]:
    """Run one training epoch with optional mixed-precision."""
    model.train()

    if scaler is None:
        scaler = torch.amp.GradScaler(enabled=False)
    use_amp = scaler.is_enabled()

    total_loss = 0.0
    num_batches = 0
    total_graphs = 0
    total_errors = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch)

            if case == "logical_head":
                target = batch.y
                loss = criterion(logits.view(-1), target)
            else:
                loss = criterion(logits, batch.y, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if case == "logical_head":
            with torch.no_grad():
                pred = (logits > 0.0).float()
                target_2d = batch.y.view_as(pred)
                total_graphs += pred.shape[0]
                total_errors += int((pred != target_2d).any(dim=1).sum().item())

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
    """Run validation and compute metrics."""
    model.eval()
    use_amp = device.type == "cuda"

    total_loss = 0.0
    num_batches = 0
    total_graphs = 0
    total_errors = 0
    edge_correct = 0
    edge_total = 0

    for batch in loader:
        batch = batch.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch)

            if case == "logical_head":
                target = batch.y
                loss = criterion(logits.view(-1), target)
            else:
                loss = criterion(logits, batch.y, batch)

        if case == "logical_head":
            pred = (logits > 0.0).float()
            target_2d = batch.y.view_as(pred)
            total_graphs += pred.shape[0]
            total_errors += int((pred != target_2d).any(dim=1).sum().item())
        else:
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
    """Save a training checkpoint."""
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
    """Load a training checkpoint."""
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
    return "ler" if case == "logical_head" else "loss"


def _format_metrics(metrics: Dict[str, float]) -> str:
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
    scheduler, sched_desc = _build_scheduler(optimizer, cfg.epochs, eta_min)
    logger.info("Scheduler: %s", sched_desc)

    # ── Loss ──────────────────────────────────────────────────────

    pos_weight = cfg.edge_pos_weight
    if cfg.case != "logical_head" and pos_weight is None:
        pos_weight = _estimate_edge_pos_weight(train_loader)

    criterion = build_criterion(
        cfg.case,
        pos_weight=pos_weight,
        device=device,
    )

    # ── AMP scaler ────────────────────────────────────────────────

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    if scaler.is_enabled():
        logger.info("Mixed precision training enabled (AMP)")

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
            scaler=scaler,
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
