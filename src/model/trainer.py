"""
Trainer for GNN-based QEC decoders.

Sample-budget training with streaming on-the-fly Stim sampling and frozen
validation sets.  The training loop halts after consuming
``sample_budget`` training samples; cumulative ``samples_consumed`` is
logged at every validation checkpoint and persisted in saved checkpoints.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from model.dataset import StreamingSurfaceCodeDataset
from model.decoder import QECDecoder, build_model
from sampling.sampler import settings_from_circuit_dir
from sampling.seeding import stable_seed


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyperparameters for sample-budget training.

    Parameters
    ----------
    circuit_dir : Path
        Directory containing committed ``.stim`` circuit files.
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
    sample_budget : int
        Total training samples to consume before halting.
    batch_size : int
        Graphs per training batch.
    num_workers : int
        DataLoader worker processes.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    patience : int
        Early stopping patience (validation checks without improvement).
        Set to 0 to disable early stopping.
    val_interval_samples : int
        Run validation every this many training samples consumed.
    val_size : int
        Number of samples in the frozen validation set, pre-sampled
        once at training start with a deterministic seed.
    seed : int
        Master random seed.
    resume : Path or None
        Path to checkpoint to resume from.
    backend : str
        Compute backend: ``"pytorch"`` (default) or ``"compiled"``
        (recommended on GPU).  The ``"cuda"`` backend is inference-only
        and must not be used for training (no autograd backward).
    compile_mode : str
        ``torch.compile`` mode (only used when backend is ``"compiled"``).
        Use ``"default"`` for training — GNN batches have dynamic shapes
        (variable N/E) which cause ``"reduce-overhead"`` to record
        excessive CUDA graphs and degrade performance over time.
    amp_dtype : str
        Autocast dtype for mixed precision: ``"bfloat16"`` (default,
        recommended on Ampere+ GPUs) or ``"float16"``.  Only used
        when training on CUDA.  Model weights, optimizer state, and
        loss remain in float32 regardless.
    focal_alpha : float
        Focal loss balancing factor for the positive class.
    focal_gamma : float
        Focal loss focusing exponent.
    warmup_fraction : float
        Fraction of sample budget for LR warmup (linear ramp).
    include_p_feature : bool
        Attach physical error probability as a graph-level feature.
    """

    circuit_dir: Path = Path("data/circuits")
    output_dir: Path = Path("outputs/runs")
    hidden_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    sample_budget: int = 1_000_000
    batch_size: int = 64
    num_workers: int = 4
    max_grad_norm: float = 1.0
    patience: int = 10
    val_interval_samples: int = 50_000
    val_size: int = 10_000
    seed: int = 42
    resume: Path | None = None
    backend: str = "pytorch"
    compile_mode: str = "default"
    amp_dtype: str = "bfloat16"
    focal_alpha: float = 0.75
    focal_gamma: float = 1.0
    warmup_fraction: float = 0.05
    include_p_feature: bool = False
    distances: list[int] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load configuration from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        flat: dict[str, Any] = {}

        for key in (
            "circuit_dir",
            "output_dir",
            "num_workers",
            "max_grad_norm",
            "patience",
            "seed",
            "backend",
            "compile_mode",
            "amp_dtype",
            "focal_alpha",
            "focal_gamma",
            "sample_budget",
            "val_interval_samples",
            "val_size",
            "warmup_fraction",
            "include_p_feature",
            "distances",
        ):
            if key in raw:
                flat[key] = raw[key]

        for key in ("hidden_dim", "num_layers", "dropout"):
            if key in raw.get("model", {}):
                flat[key] = raw["model"][key]

        for key in ("lr", "weight_decay", "batch_size"):
            if key in raw.get("optimisation", {}):
                flat[key] = raw["optimisation"][key]

        for key in ("circuit_dir", "output_dir"):
            if key in flat:
                flat[key] = Path(flat[key])

        return cls(**flat)

    def __post_init__(self) -> None:
        if self.sample_budget < 1:
            raise ValueError(f"sample_budget must be >= 1, got {self.sample_budget}")
        if self.val_interval_samples < 1:
            raise ValueError(
                f"val_interval_samples must be >= 1, got {self.val_interval_samples}"
            )
        if self.val_size < 1:
            raise ValueError(f"val_size must be >= 1, got {self.val_size}")
        if not (0.0 < self.warmup_fraction < 1.0):
            raise ValueError(
                f"warmup_fraction must be in (0, 1), got {self.warmup_fraction}"
            )


def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class FocalBCEWithLogitsLoss(nn.Module):
    """Sigmoid focal loss for imbalanced binary classification.

    Down-weights well-classified examples, focusing training on hard
    positives/negatives.  Standard choice for rare-positive detection
    tasks (logical flip rate is low at small ``p`` and large ``d``).

    Parameters
    ----------
    alpha : float
        Balancing factor for the positive class (default: 0.25).
    gamma : float
        Focusing exponent- higher values suppress easy examples more
        aggressively (default: 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()


def _build_criterion(cfg: TrainConfig) -> nn.Module:
    """Build the loss function."""
    return FocalBCEWithLogitsLoss(
        alpha=cfg.focal_alpha,
        gamma=cfg.focal_gamma,
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """GNN trainer with sample-budget semantics.

    Training halts after consuming ``cfg.sample_budget`` training
    samples.  The learning-rate scheduler is parameterized by budget
    fraction (total optimizer steps = budget / batch_size), not epochs.
    Validation runs on a frozen set pre-sampled once at training start.

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

        self.samples_consumed: int = 0
        self.best_metric: float = float("inf")
        self._decision_threshold: float = 0.0
        self.history: list[dict[str, Any]] = []

        self._node_dim: int
        self._edge_dim: int
        self._run_dir: Path
        self._best_path: Path

    def _setup_data(self) -> None:
        """Build streaming training loader and frozen validation set.

        The training DataLoader wraps an infinite
        ``StreamingSurfaceCodeDataset``.  The validation set is
        pre-sampled once (deterministic seed) and cached as a finite
        list — it does not change between validation checks.
        """
        settings = settings_from_circuit_dir(
            self.cfg.circuit_dir, distances=self.cfg.distances
        )

        train_ds = StreamingSurfaceCodeDataset(
            settings=settings,
            master_seed=self.cfg.seed,
            include_p_feature=self.cfg.include_p_feature,
        )
        self._node_dim = train_ds.node_dim
        self._edge_dim = train_ds.edge_dim

        pin = self.device.type == "cuda"
        persistent = self.cfg.num_workers > 0
        prefetch = 2 if self.cfg.num_workers > 0 else None
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )

        # Frozen val set: sample once with a deterministic seed.
        val_seed = stable_seed("val", f"seed={self.cfg.seed}", base=self.cfg.seed)
        val_ds = StreamingSurfaceCodeDataset(
            settings=settings,
            master_seed=val_seed,
            include_p_feature=self.cfg.include_p_feature,
        )
        val_samples = list(itertools.islice(val_ds, self.cfg.val_size))
        self.val_loader = DataLoader(
            val_samples,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(
            "Training: streaming from %d settings, budget=%d samples",
            len(settings),
            self.cfg.sample_budget,
        )
        logger.info("Validation: %d frozen samples", len(val_samples))

    def _setup_model(self) -> None:
        """Build model, set compute backend, and move to device."""
        from model.ops import set_backend

        set_backend(self.cfg.backend)

        self.model = build_model(
            node_dim=self._node_dim,
            edge_dim=self._edge_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)

        if self.cfg.backend == "compiled" and self.device.type == "cuda":
            self.model = torch.compile(
                self.model,
                mode=self.cfg.compile_mode,
                dynamic=True,
                fullgraph=False,
            )
            logger.info(
                "torch.compile enabled (mode=%s, dynamic=True)",
                self.cfg.compile_mode,
            )

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Model: %d trainable parameters", num_params)

    def _setup_optimizer(self) -> None:
        """Build optimizer and step-level LR scheduler.

        The scheduler is parameterized by total optimizer steps
        (``sample_budget / batch_size``), not epochs.  It steps after
        every training batch.
        """
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        total_steps = max(1, self.cfg.sample_budget // self.cfg.batch_size)
        warmup_steps = max(1, int(self.cfg.warmup_fraction * total_steps))

        eta_min = self.cfg.lr / 50
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=eta_min,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        logger.info(
            "Scheduler: LinearWarmup(%d steps) => "
            "CosineAnnealing(T_max=%d, eta_min=%.1e)",
            warmup_steps,
            total_steps - warmup_steps,
            eta_min,
        )

    def _setup_criterion(self) -> None:
        """Build loss function."""
        self.criterion = _build_criterion(self.cfg)

    def _setup_amp(self) -> None:
        """Configure automatic mixed precision."""
        use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self._amp_dtype = getattr(torch, self.cfg.amp_dtype, torch.bfloat16)
        if use_amp:
            torch.set_float32_matmul_precision("high")
            logger.info(
                "Mixed precision training enabled (AMP, dtype=%s, TF32=high)",
                self.cfg.amp_dtype,
            )

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
        for key in ("hidden_dim", "num_layers"):
            current = getattr(self.cfg, key)
            saved = ckpt_cfg.get(key)
            if saved is not None and saved != current:
                raise ValueError(
                    f"Config mismatch on resume: {key}={current}, "
                    f"checkpoint has {key}={saved}"
                )

        self.samples_consumed = ckpt.get("samples_consumed", 0)
        self.best_metric = ckpt.get("best_metric", float("inf"))
        self._decision_threshold = ckpt.get("decision_threshold", 0.0)
        logger.info(
            "Resuming from %d samples (best_metric=%.6f, threshold=%.3f)",
            self.samples_consumed,
            self.best_metric,
            self._decision_threshold,
        )

    def _save_config(self) -> None:
        """Persist run configuration to JSON."""
        config_dict = asdict(self.cfg)
        config_dict["circuit_dir"] = str(self.cfg.circuit_dir)
        config_dict["output_dir"] = str(self.cfg.output_dir)
        config_dict["resume"] = str(self.cfg.resume) if self.cfg.resume else None
        config_dict["node_dim"] = self._node_dim
        config_dict["edge_dim"] = self._edge_dim
        config_dict["distances"] = self.cfg.distances

        path = self._run_dir / "config.json"
        path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation on the frozen val set.

        Returns
        -------
        dict
            Validation metrics: ``loss`` and ``ler``.
        """
        self.model.eval()
        use_amp = self.scaler.is_enabled()

        total_loss = 0.0
        num_batches = 0
        total_graphs = 0
        total_errors = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=use_amp,
                dtype=self._amp_dtype,
            ):
                logits = self.model(batch)
                loss = self.criterion(logits.view(-1), batch.y)

            pred = (logits > 0.0).float()
            target_2d = batch.y.view_as(pred)
            total_graphs += pred.shape[0]
            total_errors += int((pred != target_2d).any(dim=1).sum().item())

            total_loss += loss.item()
            num_batches += 1

        metrics: dict[str, float] = {
            "loss": total_loss / max(num_batches, 1),
        }
        if total_graphs > 0:
            metrics["ler"] = total_errors / total_graphs

        return metrics

    @torch.no_grad()
    def calibrate_threshold(self) -> float:
        """Sweep decision thresholds on the validation set for ``direct``.

        Collects all per-graph logits from the validation loader, then
        evaluates LER at each candidate threshold in logit space.
        Returns the threshold that minimises LER.

        Returns
        -------
        float
            Optimal logit threshold (0.0 = sigmoid 0.5 default).
        """
        self.model.eval()
        use_amp = self.scaler.is_enabled()

        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in self.val_loader:
            batch = batch.to(self.device)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=use_amp,
                dtype=self._amp_dtype,
            ):
                logits = self.model(batch)
            all_logits.append(logits.cpu())
            all_targets.append(batch.y.view_as(logits).cpu())

        if not all_logits:
            return 0.0

        logits_cat = torch.cat(all_logits, dim=0)  # (N_val, num_obs)
        targets_cat = torch.cat(all_targets, dim=0)  # (N_val, num_obs)

        # Sweep thresholds in logit space.
        thresholds = torch.linspace(-4.0, 4.0, steps=81)
        best_threshold = 0.0
        best_ler = float("inf")

        for thr in thresholds:
            pred = (logits_cat > thr.item()).float()
            errors = (pred != targets_cat).any(dim=1).sum().item()
            ler = errors / logits_cat.shape[0]
            if ler < best_ler:
                best_ler = ler
                best_threshold = thr.item()

        # Log improvement over default.
        default_pred = (logits_cat > 0.0).float()
        default_ler = (default_pred != targets_cat).any(
            dim=1
        ).sum().item() / logits_cat.shape[0]

        logger.info(
            "Threshold calibration: default(0.0) LER=%.6f, "
            "best(%.3f) LER=%.6f (delta=%.6f)",
            default_ler,
            best_threshold,
            best_ler,
            best_ler - default_ler,
        )

        return best_threshold

    def save_checkpoint(
        self, path: Path, samples_consumed: int, best_metric: float
    ) -> None:
        """Save a training checkpoint.

        Parameters
        ----------
        path : Path
            Checkpoint file path.
        samples_consumed : int
            Cumulative training samples consumed at this point.
        best_metric : float
            Best validation metric achieved so far.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = asdict(self.cfg)
        config_dict["node_dim"] = self._node_dim
        config_dict["edge_dim"] = self._edge_dim
        torch.save(
            {
                "samples_consumed": samples_consumed,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),  # type: ignore[attr-defined]
                "best_metric": best_metric,
                "decision_threshold": self._decision_threshold,
                "config": config_dict,
            },
            path,
        )
        logger.debug("Saved checkpoint to %s (samples=%d)", path, samples_consumed)

    def fit(self) -> Path:
        """Execute sample-budget training.

        Consumes up to ``cfg.sample_budget`` training samples from the
        streaming dataset.  Validation runs every
        ``cfg.val_interval_samples`` samples on a frozen validation set.
        Early stopping triggers after ``cfg.patience`` consecutive
        validation checks without improvement (disabled when patience=0).

        Returns
        -------
        Path
            Path to the best model checkpoint.
        """
        seed_everything(self.cfg.seed)
        logger.info("Device: %s", self.device)

        self._run_dir = self.cfg.output_dir / "direct"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._best_path = self._run_dir / "best.pt"

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_criterion()
        self._setup_amp()
        self._maybe_resume()
        self._save_config()

        metric_key = "ler"
        checks_without_improvement = 0

        running_loss = 0.0
        running_batches = 0
        running_graphs = 0
        running_errors = 0

        next_val_at = self.samples_consumed + self.cfg.val_interval_samples
        t0 = time.perf_counter()

        logger.info(
            "Starting training: budget=%d samples (val every %d, patience=%d)",
            self.cfg.sample_budget,
            self.cfg.val_interval_samples,
            self.cfg.patience,
        )

        train_iter = iter(self.train_loader)
        use_amp = self.scaler.is_enabled()

        while self.samples_consumed < self.cfg.sample_budget:
            batch = next(train_iter)
            batch = batch.to(self.device)

            # --- Forward / backward ---
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=use_amp,
                dtype=self._amp_dtype,
            ):
                logits = self.model(batch)
                loss = self.criterion(logits.view(-1), batch.y)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()  # type: ignore[attr-defined]

            # --- Accumulate running stats ---
            self.samples_consumed += batch.num_graphs
            running_loss += loss.item()
            running_batches += 1

            with torch.no_grad():
                pred = (logits > 0.0).float()
                target_2d = batch.y.view_as(pred)
                running_graphs += pred.shape[0]
                running_errors += int((pred != target_2d).any(dim=1).sum().item())

            # --- Validation check ---
            do_val = (
                self.samples_consumed >= next_val_at
                or self.samples_consumed >= self.cfg.sample_budget
            )
            if not do_val:
                continue

            elapsed = time.perf_counter() - t0
            train_metrics: dict[str, float] = {
                "loss": running_loss / max(running_batches, 1),
            }
            if running_graphs > 0:
                train_metrics["ler"] = running_errors / running_graphs

            val_metrics = self.validate()
            current = val_metrics[metric_key]
            improved = current < self.best_metric

            if improved:
                self.best_metric = current
                checks_without_improvement = 0
                self.save_checkpoint(
                    self._best_path,
                    self.samples_consumed,
                    self.best_metric,
                )
            else:
                checks_without_improvement += 1

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Samples %d/%d [%.1fs, lr=%.1e]  train: %s  val: %s%s",
                self.samples_consumed,
                self.cfg.sample_budget,
                elapsed,
                lr,
                _format_metrics(train_metrics),
                _format_metrics(val_metrics),
                " *" if improved else "",
            )

            self.history.append(
                {
                    "samples_consumed": self.samples_consumed,
                    "lr": lr,
                    "elapsed_s": round(elapsed, 2),
                    "train": train_metrics,
                    "val": val_metrics,
                    "best": improved,
                }
            )

            # Reset running stats for the next interval.
            running_loss = 0.0
            running_batches = 0
            running_graphs = 0
            running_errors = 0
            t0 = time.perf_counter()
            next_val_at = self.samples_consumed + self.cfg.val_interval_samples

            if (
                self.cfg.patience > 0
                and checks_without_improvement >= self.cfg.patience
            ):
                logger.info(
                    "Early stopping at %d samples (%d checks w/o improvement)",
                    self.samples_consumed,
                    self.cfg.patience,
                )
                break

        # --- Post-training ---
        history_path = self._run_dir / "history.json"
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        logger.info(
            "Training complete. Best %s=%.6f, samples_consumed=%d",
            metric_key,
            self.best_metric,
            self.samples_consumed,
        )
        logger.info("Best checkpoint: %s", self._best_path)

        if self._best_path.exists():
            ckpt = torch.load(self._best_path, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

            self._decision_threshold = self.calibrate_threshold()

            self.save_checkpoint(
                self._best_path,
                ckpt["samples_consumed"],
                self.best_metric,
            )
            logger.info(
                "Checkpoint updated with decision_threshold=%.3f",
                self._decision_threshold,
            )

        return self._best_path


def _format_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for k, v in metrics.items():
        if k == "ler":
            parts.append(f"LER={v:.4f}")
        elif k == "loss" and abs(v) < 1e-3:
            parts.append(f"loss={v:.2e}")
        else:
            parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)
