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
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from constants import Case
from gnn.dataset import MixedSurfaceCodeDataset
from gnn.models.heads import QECDecoder, build_model


logger = logging.getLogger(__name__)


def build_stratified_subset(
    setting_ids: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Build a stratified random subset balanced across setting IDs.

    Allocates an equal quota to each unique setting.  Leftover capacity
    (from settings with fewer samples than their quota) is filled by
    sampling uniformly from the remaining pool, preserving overall
    balance as much as possible.

    Parameters
    ----------
    setting_ids : ndarray, shape ``(N,)``, int
        Per-sample setting identifier (e.g. from
        ``MixedSurfaceCodeDataset._setting_id``).
    max_samples : int
        Maximum number of samples to select.  Must be >= 1.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape ``(M,)``, int64
        Selected indices (shuffled), where ``M = min(max_samples, N)``.

    Raises
    ------
    ValueError
        If *max_samples* < 1 or *setting_ids* is empty.
    """
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")
    n_total = len(setting_ids)
    if n_total == 0:
        raise ValueError("setting_ids must be non-empty")

    # Fast path: requesting everything - just shuffle.
    if max_samples >= n_total:
        rng = np.random.default_rng(seed)
        idx = np.arange(n_total, dtype=np.int64)
        rng.shuffle(idx)
        return idx

    rng = np.random.default_rng(seed)
    unique_sids = np.unique(setting_ids)
    n_settings = len(unique_sids)

    per_setting = max_samples // n_settings

    chosen_parts: list[np.ndarray] = []
    leftover_parts: list[np.ndarray] = []

    for sid in unique_sids:
        indices = np.flatnonzero(setting_ids == sid).astype(np.int64)
        rng.shuffle(indices)
        take = min(per_setting, len(indices))
        chosen_parts.append(indices[:take])
        if take < len(indices):
            leftover_parts.append(indices[take:])

    chosen = np.concatenate(chosen_parts)
    remaining = max_samples - len(chosen)

    if remaining > 0 and leftover_parts:
        leftovers = np.concatenate(leftover_parts)
        rng.shuffle(leftovers)
        chosen = np.concatenate([chosen, leftovers[:remaining]])

    rng.shuffle(chosen)
    return chosen


def build_direct_sampler(
    ds: MixedSurfaceCodeDataset,
    indices: Sequence[int] | None,
    pos_oversample_cap: float = 10.0,
) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` for the ``direct`` training case.

    Combines two sources of per-sample reweighting:

    1. **Distance weight** - larger ``d`` gets higher weight so that
       harder settings are not drowned out by easy small-distance ones.
       Formula: ``w_d = distance / d_min``.
    2. **Class weight** - positive samples (any observable flipped) are
       up-weighted to counter the rarity of logical errors at low ``p``.
       Formula: ``w_cls = min(n_neg / n_pos, pos_oversample_cap)`` for
       positive samples, ``1.0`` for negatives.

    Final per-sample weight: ``w_d * w_cls``.

    Parameters
    ----------
    ds : MixedSurfaceCodeDataset
        The underlying (non-subsetted) dataset.
    indices : sequence of int or None
        If the training set is a ``Subset``, pass the subset indices.
        ``None`` means use all samples.
    pos_oversample_cap : float
        Maximum positive-class weight to avoid extreme oversampling.

    Returns
    -------
    WeightedRandomSampler
        Sampler with ``replacement=True`` and ``num_samples=len(indices)``.
    """
    if indices is None:
        idx_arr = np.arange(len(ds), dtype=np.int64)
    else:
        idx_arr = np.asarray(indices, dtype=np.int64)

    n_samples = len(idx_arr)
    setting_ids = ds._setting_id[idx_arr]
    shot_ids = ds._shot_id[idx_arr]

    # --- Precompute per-sample: distance and is_positive ---
    distances = np.empty(n_samples, dtype=np.float64)
    is_positive = np.empty(n_samples, dtype=bool)

    # Group by setting for efficient bulk logical label lookup.
    unique_sids = np.unique(setting_ids)
    for sid in unique_sids:
        mask = setting_ids == sid
        info = ds.settings[int(sid)]
        distances[mask] = info.distance

        # Load logical labels (memory-mapped).
        _, logical_mm = ds._get_split_arrays(int(sid))
        shots = shot_ids[mask]
        logical_rows = np.asarray(logical_mm[shots], dtype=np.float32)
        if logical_rows.ndim == 1:
            is_positive[mask] = logical_rows > 0.5
        else:
            is_positive[mask] = logical_rows.any(axis=1)

    # --- Distance weight ---
    d_min = distances.min()
    w_distance = distances / max(d_min, 1.0)

    # --- Class weight ---
    n_pos = int(is_positive.sum())
    n_neg = n_samples - n_pos
    if n_pos > 0:
        raw_pos_weight = n_neg / n_pos
        pos_weight = min(raw_pos_weight, pos_oversample_cap)
    else:
        pos_weight = 1.0

    w_class = np.where(is_positive, pos_weight, 1.0)

    weights = w_distance * w_class

    logger.info(
        "Direct sampler: %d samples, %d pos (%.2f%%), "
        "pos_weight=%.2f, d_weights=[%.2f..%.2f]",
        n_samples,
        n_pos,
        100.0 * n_pos / max(n_samples, 1),
        pos_weight,
        w_distance.min(),
        w_distance.max(),
    )

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=n_samples,
        replacement=True,
    )


@dataclass
class TrainConfig:
    """
    Training hyperparameters.

    Attributes
    ----------
    case : str
        Training case: ``"direct"`` or ``"edge"``.
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
        from training data.  Only used for ``edge``.
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
    backend : str
        Compute backend: ``"pytorch"`` (default) or ``"compiled"``
        (recommended on GPU).  The ``"cuda"`` backend is inference-only
        and must not be used for training (no autograd backward).
    compile_mode : str
        ``torch.compile`` mode (only used when backend is ``"compiled"``).
        Use ``"default"`` for training- GNN batches have dynamic shapes
        (variable N/E) which cause ``"reduce-overhead"`` to record
        excessive CUDA graphs and degrade performance over time.
    """

    case: Case = "direct"
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
    backend: str = "pytorch"
    compile_mode: str = "default"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    direct_pos_oversample_cap: float = 10.0

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
            "backend",
            "compile_mode",
            "focal_alpha",
            "focal_gamma",
            "direct_pos_oversample_cap",
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


class GraphNormalizedSoftBCELoss(nn.Module):
    """
    Graph-normalized soft BCE between logits and soft teacher marginals.

    For each edge, computes ``BCE_with_logits(logit, target)`` where
    targets are continuous BP marginals in ``[0, 1]``.  Losses are
    averaged within each graph, then across graphs, so that larger
    graphs do not dominate the gradient in mixed-distance batches.

    This replaces the previous MSE-on-sigmoid approach, operating
    directly in logit space for better numerical stability with the
    residual prior + delta prediction scheme.
    """

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        per_edge = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        edge_graph = batch.batch[batch.edge_index[0]]
        n_graphs = int(batch.batch.max()) + 1

        graph_loss = torch.zeros(n_graphs, device=logits.device)
        graph_count = torch.zeros(n_graphs, device=logits.device)
        graph_loss.scatter_add_(0, edge_graph, per_edge)
        graph_count.scatter_add_(0, edge_graph, torch.ones_like(per_edge))
        return (graph_loss / graph_count.clamp(min=1)).mean()


def _build_criterion(case: Case, cfg: "TrainConfig") -> nn.Module:
    """Build the loss function for a given training case."""
    if case == "direct":
        return FocalBCEWithLogitsLoss(
            alpha=cfg.focal_alpha,
            gamma=cfg.focal_gamma,
        )
    # edge case: graph-normalized soft BCE against BP soft labels
    return GraphNormalizedSoftBCELoss()


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
        self._decision_threshold: float = 0.0
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

        # Feature dimensions (fixed by dataset).
        self._node_dim = train_ds.node_dim
        self._edge_dim = train_ds.edge_dim

        _validate_dataset(train_ds, "train")
        _validate_dataset(val_ds, "val")

        if self.cfg.max_samples is not None:
            train_idx = build_stratified_subset(
                train_ds._setting_id,
                self.cfg.max_samples,
                self.cfg.seed,
            )
            val_cap = min(
                len(val_ds),
                max(20_000, self.cfg.max_samples // 5),
            )
            val_idx = build_stratified_subset(
                val_ds._setting_id,
                val_cap,
                self.cfg.seed + 1,
            )
            logger.info(
                "Stratified subset: train %d -> %d, val %d -> %d",
                len(train_ds),
                len(train_idx),
                len(val_ds),
                len(val_idx),
            )
            train_ds = Subset(train_ds, train_idx.tolist())
            val_ds = Subset(val_ds, val_idx.tolist())

        # Build weighted sampler for direct (imbalanced pos class).
        train_sampler = None
        train_shuffle = True
        if self.cfg.case == "direct":
            # Resolve underlying dataset and indices for sampler.
            if isinstance(train_ds, Subset):
                raw_ds = train_ds.dataset
                subset_indices = train_ds.indices
            else:
                raw_ds = train_ds
                subset_indices = None
            train_sampler = build_direct_sampler(
                raw_ds,
                subset_indices,
                pos_oversample_cap=self.cfg.direct_pos_oversample_cap,
            )
            train_shuffle = False  # mutually exclusive with sampler

        pin = self.device.type == "cuda"
        persistent = self.cfg.num_workers > 0
        prefetch = 2 if self.cfg.num_workers > 0 else None
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
        logger.info(
            "Train: %d samples, Val: %d samples",
            len(train_ds),
            len(val_ds),
        )

    def _setup_model(self) -> None:
        """Build model, set compute backend, and move to device."""
        from gnn.models.ops import set_backend

        set_backend(self.cfg.backend)

        self.model = build_model(
            self.cfg.case,
            node_dim=self._node_dim,
            edge_dim=self._edge_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)

        if self.cfg.backend == "compiled" and self.device.type == "cuda":
            # GNN batches have dynamic shapes (variable nodes/edges per
            # batch due to mixed code distances). dynamic=True avoids
            # recompilation per shape; fullgraph=False lets scatter ops
            # cause graph breaks without error.
            self.model = torch.compile(
                self.model,
                mode=self.cfg.compile_mode,
                dynamic=True,
                fullgraph=False,
            )
            logger.info(
                "torch.compile enabled (mode=%s, dynamic=True)", self.cfg.compile_mode
            )

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
            "Scheduler: LinearWarmup(%dep) => CosineAnnealing(T_max=%d, eta_min=%.1e)",
            warmup_epochs,
            self.cfg.epochs - warmup_epochs,
            eta_min,
        )

    def _setup_criterion(self) -> None:
        """Build loss function."""
        self.criterion = _build_criterion(self.cfg.case, self.cfg)

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
        self._decision_threshold = ckpt.get("decision_threshold", 0.0)
        logger.info(
            "Resuming from epoch %d (best_metric=%.6f, threshold=%.3f)",
            self.start_epoch,
            self.best_metric,
            self._decision_threshold,
        )

    def _save_config(self) -> None:
        """Persist run configuration to JSON."""
        config_dict = asdict(self.cfg)
        config_dict["datasets_dir"] = str(self.cfg.datasets_dir)
        config_dict["output_dir"] = str(self.cfg.output_dir)
        config_dict["resume"] = str(self.cfg.resume) if self.cfg.resume else None
        config_dict["node_dim"] = self._node_dim
        config_dict["edge_dim"] = self._edge_dim

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
        total_mae = 0.0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=use_amp,
            ):
                logits = self.model(batch)
                if self.cfg.case == "direct":
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

            with torch.no_grad():
                if self.cfg.case == "direct":
                    pred = (logits > 0.0).float()
                    target_2d = batch.y.view_as(pred)
                    total_graphs += pred.shape[0]
                    total_errors += int((pred != target_2d).any(dim=1).sum().item())
                else:
                    total_mae += float(
                        (torch.sigmoid(logits) - batch.y).abs().mean().item()
                    )

            total_loss += loss.item()
            num_batches += 1

        metrics: Dict[str, float] = {
            "loss": total_loss / max(num_batches, 1),
        }
        if self.cfg.case == "direct" and total_graphs > 0:
            metrics["ler"] = total_errors / total_graphs
        if self.cfg.case == "edge" and num_batches > 0:
            metrics["mae"] = total_mae / num_batches

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation and compute metrics.

        Returns
        -------
        dict
            Validation metrics: ``loss`` always; ``ler`` for direct.
        """
        self.model.eval()
        use_amp = self.device.type == "cuda"

        total_loss = 0.0
        num_batches = 0
        total_graphs = 0
        total_errors = 0
        total_mae = 0.0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=use_amp,
            ):
                logits = self.model(batch)
                if self.cfg.case == "direct":
                    loss = self.criterion(logits.view(-1), batch.y)
                else:
                    loss = self.criterion(logits, batch.y, batch)

            if self.cfg.case == "direct":
                pred = (logits > 0.0).float()
                target_2d = batch.y.view_as(pred)
                total_graphs += pred.shape[0]
                total_errors += int((pred != target_2d).any(dim=1).sum().item())
            else:
                total_mae += float(
                    (torch.sigmoid(logits) - batch.y).abs().mean().item()
                )

            total_loss += loss.item()
            num_batches += 1

        metrics: Dict[str, float] = {
            "loss": total_loss / max(num_batches, 1),
        }
        if self.cfg.case == "direct" and total_graphs > 0:
            metrics["ler"] = total_errors / total_graphs
        if self.cfg.case == "edge" and num_batches > 0:
            metrics["mae"] = total_mae / num_batches

        return metrics

    @torch.no_grad()
    def calibrate_threshold(self) -> float:
        """Sweep decision thresholds on the validation set for ``direct``.

        Collects all per-graph logits from the validation loader, then
        evaluates LER at each candidate threshold in logit space.
        Returns the threshold that minimises LER.

        This is cheap (single forward pass over val + vectorised sweep)
        and should be called once after training completes.

        Returns
        -------
        float
            Optimal logit threshold (0.0 = sigmoid 0.5 default).
        """
        if self.cfg.case != "direct":
            return 0.0

        self.model.eval()
        use_amp = self.device.type == "cuda"

        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in self.val_loader:
            batch = batch.to(self.device)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=use_amp,
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

    def save_checkpoint(self, path: Path, epoch: int, best_metric: float) -> None:
        """Save a training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = asdict(self.cfg)
        config_dict["node_dim"] = self._node_dim
        config_dict["edge_dim"] = self._edge_dim
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),  # type: ignore[attr-defined]
                "best_metric": best_metric,
                "decision_threshold": self._decision_threshold,
                "config": config_dict,
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

        metric_key = "ler" if self.cfg.case == "direct" else "loss"
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

        # Calibrate decision threshold for direct case.
        if self.cfg.case == "direct" and self._best_path.exists():
            # Reload best model weights for calibration.
            ckpt = torch.load(self._best_path, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

            self._decision_threshold = self.calibrate_threshold()

            # Re-save with calibrated threshold.
            self.save_checkpoint(
                self._best_path,
                ckpt["epoch"],
                self.best_metric,
            )
            logger.info(
                "Checkpoint updated with decision_threshold=%.3f",
                self._decision_threshold,
            )

        return self._best_path


def _validate_dataset(ds: MixedSurfaceCodeDataset, label: str) -> None:
    """Smoke-test dataset by loading first sample; raises on bad shapes."""
    from gnn.dataset import EDGE_DIM, NODE_DIM

    sample = ds[0]
    if sample.x.ndim != 2 or sample.x.shape[1] != NODE_DIM:
        raise ValueError(
            f"{label} dataset: expected x shape (N, {NODE_DIM}), "
            f"got {tuple(sample.x.shape)}"
        )
    if sample.edge_index.shape[0] != 2:
        raise ValueError(
            f"{label} dataset: expected edge_index shape (2, E), "
            f"got {tuple(sample.edge_index.shape)}"
        )
    if sample.edge_attr.ndim != 2 or sample.edge_attr.shape[1] != EDGE_DIM:
        raise ValueError(
            f"{label} dataset: expected edge_attr shape (E, {EDGE_DIM}), "
            f"got {tuple(sample.edge_attr.shape)}"
        )
    if sample.y.ndim < 1:
        raise ValueError(f"{label} dataset: y is scalar, expected ≥1D")


def _format_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    for k, v in metrics.items():
        if k == "ler":
            parts.append(f"LER={v:.4f}")
        elif k == "mae":
            parts.append(f"MAE={v:.2e}")
        elif k == "loss" and abs(v) < 1e-3:
            parts.append(f"loss={v:.2e}")
        else:
            parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)
