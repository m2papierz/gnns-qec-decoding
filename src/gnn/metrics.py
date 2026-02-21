"""
Shared evaluation result types for QEC decoders.

Provides :class:`SettingResult` and :class:`EvalReport` used by both
the GNN evaluator and the MWPM baseline script, eliminating the
previous duplication across modules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class SettingResult:
    """
    Evaluation result for a single (distance, rounds, error_prob) setting.

    Attributes
    ----------
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    split : str
        Evaluated split name.
    num_shots : int
        Total number of shots evaluated.
    num_errors : int
        Shots where the decoder predicted the wrong observable.
    logical_error_rate : float
        ``num_errors / num_shots``.
    elapsed_s : float
        Wall-clock seconds for this setting.
    mwpm_ler : float or None
        MWPM baseline LER for comparison (if available).
    edge_acc : float or None
        Per-edge accuracy (edge cases only).
    """

    distance: int
    rounds: int
    error_prob: float
    split: str
    num_shots: int
    num_errors: int
    logical_error_rate: float
    elapsed_s: float
    mwpm_ler: float | None = None
    edge_acc: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dictionary."""
        d: Dict[str, Any] = {
            "distance": self.distance,
            "rounds": self.rounds,
            "error_prob": self.error_prob,
            "split": self.split,
            "num_shots": self.num_shots,
            "num_errors": self.num_errors,
            "logical_error_rate": self.logical_error_rate,
            "elapsed_s": round(self.elapsed_s, 4),
        }
        if self.mwpm_ler is not None:
            d["mwpm_ler"] = self.mwpm_ler
        if self.edge_acc is not None:
            d["edge_acc"] = self.edge_acc
        return d


@dataclass
class EvalReport:
    """
    Aggregated decoder evaluation report.

    Attributes
    ----------
    results : list of SettingResult
        Per-setting results.
    metadata : dict
        Run-level metadata (checkpoint, case, device, etc.).
    """

    results: List[SettingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full report."""
        return {
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path) -> None:
        """Save report to a JSON file.

        Parameters
        ----------
        path : Path
            Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info("Report saved to %s", path)


def print_report(report: EvalReport) -> None:
    """
    Log a formatted summary table.

    Parameters
    ----------
    report : EvalReport
        Evaluation report to display.
    """
    results = report.results
    if not results:
        logger.warning("No results to display")
        return

    has_baseline = any(r.mwpm_ler is not None for r in results)
    case = report.metadata.get("case", report.metadata.get("decoder", "unknown"))

    if has_baseline:
        header = (
            f"{'d':>3} {'r':>4} {'p':>9} "
            f"{'shots':>7} {'GNN_LER':>10} {'MWPM_LER':>10} {'delta':>8}"
        )
    else:
        header = f"{'d':>3} {'r':>4} {'p':>9} " f"{'shots':>7} {'LER':>10}"

    sep = "-" * len(header)
    lines: list[str] = []

    lines.append("")
    lines.append(sep)
    lines.append(f"Evaluation — {case}")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    sorted_results = sorted(
        results, key=lambda r: (r.distance, r.rounds, r.error_prob)
    )

    prev_d = None
    total_better = 0
    total_compared = 0

    for r in sorted_results:
        if prev_d is not None and r.distance != prev_d:
            lines.append("")
        prev_d = r.distance

        if has_baseline and r.mwpm_ler is not None:
            delta = r.logical_error_rate - r.mwpm_ler
            marker = " +" if delta > 0 else " "
            if delta < -1e-9:
                marker = " *"
                total_better += 1
            total_compared += 1
            lines.append(
                f"{r.distance:>3} {r.rounds:>4} {r.error_prob:>9.5f} "
                f"{r.num_shots:>7} {r.logical_error_rate:>10.6f} "
                f"{r.mwpm_ler:>10.6f} {delta:>+8.4f}{marker}"
            )
        else:
            lines.append(
                f"{r.distance:>3} {r.rounds:>4} {r.error_prob:>9.5f} "
                f"{r.num_shots:>7} {r.logical_error_rate:>10.6f}"
            )

    lines.append(sep)

    if has_baseline and total_compared > 0:
        lines.append(
            f"\nGNN better in {total_better}/{total_compared} settings "
            f"(marked with *)"
        )

    lines.append("\nSummary by distance:")
    for d in sorted(set(r.distance for r in results)):
        subset = [r for r in results if r.distance == d]
        lers = [r.logical_error_rate for r in subset]
        line = (
            f"  d={d}: "
            f"min_LER={min(lers):.6f}  "
            f"max_LER={max(lers):.6f}  "
            f"mean_LER={np.mean(lers):.6f}"
        )
        if has_baseline:
            bl_lers = [r.mwpm_ler for r in subset if r.mwpm_ler is not None]
            if bl_lers:
                line += f"  (MWPM mean={np.mean(bl_lers):.6f})"
        lines.append(line)

    lines.append("")
    print("\n".join(lines))
