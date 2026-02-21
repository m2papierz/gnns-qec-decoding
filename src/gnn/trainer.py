"""
Trainer for GNN-based QEC decoders.

Encapsulates training state (model, optimizer, scheduler, criterion) and
provides a clean ``fit()`` entry point.
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

from constants import Case
from gnn.dataset import MixedSurfaceCodeDataset
from gnn.models.heads import QECDecoder, build_model


logger = logging.getLogger(__name__)


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

        for key in ("hidden_dim", "num_layers", "dropout"):
            if key in raw.get("model", {}):
                flat[key] = raw["model"][key]

        for key in ("lr", "weight_decay", "epochs", "batch_size"):
            if key in raw.get("optimisation", {}):
                flat[key] = raw["optimisation"][key]

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


class _GraphNormalizedBCE(nn.Module):
    """
    Per-graph normalized BCE with logits for edge predictions.

    Computes per-edge BCE, averages within each graph, then averages
    across graphs.  Prevents larger graphs from dominating the gradient
    in mixed-distance batches.
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


def _build_criterion(
    case: Case,
    pos_weight: float | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Build the loss function for a given training case."""
    if case == "logical_head":
        return nn.BCEWithLogitsLoss()
    pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    return _GraphNormalizedBCE(pos_weight=pw)


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
        pos = int(batch.y.sum().item())
        total_pos += pos
        total_neg += int(batch.y.numel()) - pos

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


class Trainer:
    """
    Encapsulates GNN training state and loop.

    Parameters
    ----------
    cfg : TrainConfig
        Full training configuration.

    Example
    -------
    >>> trainer = Trainer(cfg)
    >>> best_path = trainer.fit()
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: QECDecoder
        self.optimizer: AdamW
        self.scheduler: object
        self.criterion: nn.Module
        self.scaler: torch.amp.GradScaler

        self.train_loader: DataLoader
        self.val_loader: DataLoader

        self.start_epoch: int = 0
        self.best_metric: float = float("inf")
        self.history: list[Dict[str, Any]] = []

        self._run_dir: Path
        self._best_path: Path

    def _setup_data(self) -> None:
        """Load datasets and build data loaders."""
        logger.info(
            "Loading datasets from %s (case=%s)",
            self.cfg.datasets_dir,
            self.cfg.case,
        )
        train_ds = MixedSurfaceCodeDataset(
            datasets_dir=self.cfg.datasets_dir,
            case=self.cfg.case,
            split="train",
        )
        val_ds = MixedSurfaceCodeDataset(
            datasets_dir=self.cfg.datasets_dir,
            case=self.cfg.case,
            split="val",
        )

        _validate_dataset(train_ds, "train")
        _validate_dataset(val_ds, "val")

        if self.cfg.max_samples is not None:
            from torch.utils.data import Subset

            train_ds = Subset(train_ds, range(min(self.cfg.max_samples, len(train_ds))))
            val_ds = Subset(val_ds, range(min(self.cfg.max_samples // 5, len(val_ds))))

        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            pin_memory=pin,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=pin,
        )
        logger.info(
            "Train: %d samples, Val: %d samples",
            len(train_ds),
            len(val_ds),
        )

    def _setup_model(self) -> None:
        """Build model and move to device."""
        self.model = build_model(
            self.cfg.case,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Model: %d trainable parameters", num_params)

    def _setup_optimizer(self) -> None:
        """Build optimizer and LR scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        eta_min = self.cfg.lr / 50
        warmup_epochs = max(1, self.cfg.epochs // 20)
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.epochs - warmup_epochs,
            eta_min=eta_min,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        logger.info(
            "Scheduler: LinearWarmup(%dep) → CosineAnnealing(T_max=%d, eta_min=%.1e)",
            warmup_epochs,
            self.cfg.epochs - warmup_epochs,
            eta_min,
        )

    def _setup_criterion(self) -> None:
        """Build loss function, estimating pos_weight if needed."""
        pos_weight = self.cfg.edge_pos_weight
        if self.cfg.case != "logical_head" and pos_weight is None:
            pos_weight = _estimate_edge_pos_weight(self.train_loader)

        self.criterion = _build_criterion(
            self.cfg.case,
            pos_weight=pos_weight,
            device=self.device,
        )

    def _setup_amp(self) -> None:
        """Configure automatic mixed precision."""
        self.scaler = torch.amp.GradScaler(
            enabled=(self.device.type == "cuda"),
        )
        if self.scaler.is_enabled():
            logger.info("Mixed precision training enabled (AMP)")

    def _maybe_resume(self) -> None:
        """Restore training state from checkpoint if configured."""
        if self.cfg.resume is None:
            return

        ckpt = torch.load(self.cfg.resume, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])  # type: ignore[attr-defined]

        ckpt_cfg = ckpt.get("config", {})
        for key in ("case", "hidden_dim", "num_layers"):
            current = getattr(self.cfg, key)
            saved = ckpt_cfg.get(key)
            if saved is not None and saved != current:
                raise ValueError(
                    f"Config mismatch on resume: {key}={current}, "
                    f"checkpoint has {key}={saved}"
                )

        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric", float("inf"))
        logger.info(
            "Resuming from epoch %d (best_metric=%.6f)",
            self.start_epoch,
            self.best_metric,
        )

    def _save_config(self) -> None:
        """Persist run configuration to JSON."""
        config_dict = asdict(self.cfg)
        config_dict["datasets_dir"] = str(self.cfg.datasets_dir)
        config_dict["output_dir"] = str(self.cfg.output_dir)
        config_dict["resume"] = str(self.cfg.resume) if self.cfg.resume else None

        path = self._run_dir / "config.json"
        path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.

        Returns
        -------
        dict
            Training metrics (``loss``, optionally ``ler``).
        """
        self.model.train()
        use_amp = self.scaler.is_enabled()

        total_loss = 0.0
        num_batches = 0
        total_graphs = 0
        total_errors = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=use_amp,
            ):
                logits = self.model(batch)
                if self.cfg.case == "logical_head":
                    loss = self.criterion(logits.view(-1), batch.y)
                else:
                    loss = self.criterion(logits, batch.y, batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.cfg.case == "logical_head":
                with torch.no_grad():
                    pred = (logits > 0.0).float()
                    target_2d = batch.y.view_as(pred)
                    total_graphs += pred.shape[0]
                    total_errors += int((pred != target_2d).any(dim=1).sum().item())

            total_loss += loss.item()
            num_batches += 1

        metrics: Dict[str, float] = {
            "loss": total_loss / max(num_batches, 1),
        }
        if self.cfg.case == "logical_head" and total_graphs > 0:
            metrics["ler"] = total_errors / total_graphs

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation and compute metrics.

        Returns
        -------
        dict
            Validation metrics (``loss``, optionally ``ler`` or ``edge_acc``).
        """
        self.model.eval()
        use_amp = self.device.type == "cuda"

        total_loss = 0.0
        num_batches = 0
        total_graphs = 0
        total_errors = 0
        edge_correct = 0
        edge_total = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=use_amp,
            ):
                logits = self.model(batch)
                if self.cfg.case == "logical_head":
                    loss = self.criterion(logits.view(-1), batch.y)
                else:
                    loss = self.criterion(logits, batch.y, batch)

            if self.cfg.case == "logical_head":
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

        metrics: Dict[str, float] = {
            "loss": total_loss / max(num_batches, 1),
        }
        if self.cfg.case == "logical_head" and total_graphs > 0:
            metrics["ler"] = total_errors / total_graphs
        if self.cfg.case != "logical_head" and edge_total > 0:
            metrics["edge_acc"] = edge_correct / edge_total

        return metrics

    def save_checkpoint(self, path: Path, epoch: int, best_metric: float) -> None:
        """Save a training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),  # type: ignore[attr-defined]
                "best_metric": best_metric,
                "config": asdict(self.cfg),
            },
            path,
        )
        logger.debug("Saved checkpoint to %s", path)

    def fit(self) -> Path:
        """
        Execute the full training pipeline.

        Returns
        -------
        Path
            Path to the best model checkpoint.
        """
        seed_everything(self.cfg.seed)
        logger.info("Device: %s", self.device)

        self._run_dir = self.cfg.output_dir / self.cfg.case
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._best_path = self._run_dir / "best.pt"

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_criterion()
        self._setup_amp()
        self._maybe_resume()
        self._save_config()

        metric_key = "ler" if self.cfg.case == "logical_head" else "loss"
        epochs_without_improvement = 0

        logger.info(
            "Starting training: %d epochs (val_every=%d, patience=%d)",
            self.cfg.epochs,
            self.cfg.val_every,
            self.cfg.patience,
        )

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.perf_counter()

            train_metrics = self.train_epoch()
            self.scheduler.step()  # type: ignore[attr-defined]

            elapsed = time.perf_counter() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            is_last = epoch + 1 == self.cfg.epochs
            do_validate = (epoch + 1) % self.cfg.val_every == 0 or is_last

            if do_validate:
                val_metrics = self.validate()
                current = val_metrics[metric_key]
                improved = current < self.best_metric

                if improved:
                    self.best_metric = current
                    epochs_without_improvement = 0
                    self.save_checkpoint(
                        self._best_path,
                        epoch,
                        self.best_metric,
                    )
                else:
                    epochs_without_improvement += 1

                logger.info(
                    "Epoch %3d/%d [%.1fs, lr=%.1e]  train: %s  val: %s%s",
                    epoch + 1,
                    self.cfg.epochs,
                    elapsed,
                    lr,
                    _format_metrics(train_metrics),
                    _format_metrics(val_metrics),
                    " *" if improved else "",
                )
                self.history.append(
                    {
                        "epoch": epoch,
                        "lr": lr,
                        "elapsed_s": round(elapsed, 2),
                        "train": train_metrics,
                        "val": val_metrics,
                        "best": improved,
                    }
                )

                if (
                    self.cfg.patience > 0
                    and epochs_without_improvement >= self.cfg.patience
                ):
                    logger.info(
                        "Early stopping at epoch %d (%d val rounds w/o improvement)",
                        epoch + 1,
                        self.cfg.patience,
                    )
                    break
            else:
                logger.info(
                    "Epoch %3d/%d [%.1fs, lr=%.1e]  train: %s",
                    epoch + 1,
                    self.cfg.epochs,
                    elapsed,
                    lr,
                    _format_metrics(train_metrics),
                )
                self.history.append(
                    {
                        "epoch": epoch,
                        "lr": lr,
                        "elapsed_s": round(elapsed, 2),
                        "train": train_metrics,
                    }
                )

        history_path = self._run_dir / "history.json"
        history_path.write_text(
            json.dumps(self.history, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Training complete. Best %s=%.6f",
            metric_key,
            self.best_metric,
        )
        logger.info("Best checkpoint: %s", self._best_path)

        return self._best_path


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


def _format_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    for k, v in metrics.items():
        if k == "ler":
            parts.append(f"LER={v:.4f}")
        elif k == "edge_acc":
            parts.append(f"edge_acc={v:.4f}")
        else:
            parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)
