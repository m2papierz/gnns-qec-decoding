"""Tests for decision threshold calibration on the direct case."""

import torch


def _sweep_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor,
) -> float:
    """Pure-function version of the threshold sweep logic from Trainer.

    Returns the threshold that minimises LER.
    """
    best_thr = 0.0
    best_ler = float("inf")
    for thr in thresholds:
        pred = (logits > thr.item()).float()
        errors = (pred != targets).any(dim=1).sum().item()
        ler = errors / logits.shape[0]
        if ler < best_ler:
            best_ler = ler
            best_thr = thr.item()
    return best_thr


class TestThresholdCalibration:
    """Tests for the logit-space threshold sweep."""

    def test_perfect_model_returns_zero(self) -> None:
        """When the model is perfectly calibrated, threshold ≈ 0."""
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        logits = torch.tensor([[5.0], [-5.0], [3.0], [-3.0]])
        thr = _sweep_threshold(logits, targets, torch.linspace(-4, 4, 81))
        # Any threshold in (-3, 3) gives LER=0; sweep should pick one.
        assert thr >= -3.0 and thr <= 3.0

    def test_biased_model_shifts_threshold(self) -> None:
        """Model with a +2 logit bias → optimal threshold > 0."""
        n = 200
        targets = torch.zeros(n, 1)
        targets[:50] = 1.0
        # Unbiased correct logits would be +3 / -3.
        # Add +2 bias: positives → +5, negatives → -1.
        # At default threshold 0, negatives at -1 are correct,
        # but if we shift further: positives at +5 and negatives at -1
        # are best separated around +2 (midpoint).
        # Actually this still gives LER=0 at thr=0.
        # Better test: make an overlap where threshold matters.
        logits = torch.randn(n, 1) * 0.5 + 2.0  # all biased positive
        logits[:50] += 1.5  # positives are higher
        # At thr=0: almost everything predicted positive → many FP.
        # At thr ≈ 2.0: better separation.
        thr = _sweep_threshold(logits, targets, torch.linspace(-4, 4, 81))
        assert thr > 1.0, f"Expected threshold > 1.0, got {thr}"

    def test_all_positive_targets(self) -> None:
        """All targets 1 → very negative threshold optimal (always predict 1)."""
        targets = torch.ones(100, 1)
        logits = torch.randn(100, 1)
        thr = _sweep_threshold(logits, targets, torch.linspace(-4, 4, 81))
        assert thr <= logits.min().item() + 0.2

    def test_all_negative_targets(self) -> None:
        """All targets 0 → very positive threshold optimal (always predict 0)."""
        targets = torch.zeros(100, 1)
        logits = torch.randn(100, 1)
        thr = _sweep_threshold(logits, targets, torch.linspace(-4, 4, 81))
        assert thr >= logits.max().item() - 0.2

    def test_optimal_threshold_minimises_ler(self) -> None:
        """Verify the returned threshold actually gives the minimum LER."""
        n = 500
        targets = torch.zeros(n, 1)
        targets[:100] = 1.0
        logits = torch.randn(n, 1) + 0.5 * (2 * targets - 1)

        thresholds = torch.linspace(-4, 4, 81)
        best_thr = _sweep_threshold(logits, targets, thresholds)

        # Compute LER at best and default.
        def ler_at(t: float) -> float:
            pred = (logits > t).float()
            return (pred != targets).any(dim=1).sum().item() / n

        best_ler = ler_at(best_thr)
        for thr in thresholds:
            assert ler_at(thr.item()) >= best_ler - 1e-9

    def test_multi_observable(self) -> None:
        """Works with multiple observables (error if ANY mismatch)."""
        targets = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )
        # Good predictions shifted by +1.
        logits = torch.where(targets == 1.0, torch.tensor(3.0), torch.tensor(-1.0))
        thr = _sweep_threshold(logits, targets, torch.linspace(-4, 4, 81))
        # Should find a threshold in (-1, 3) that gives LER=0.
        pred = (logits > thr).float()
        assert (pred == targets).all()

    def test_returns_float(self) -> None:
        logits = torch.randn(50, 1)
        targets = torch.randint(0, 2, (50, 1)).float()
        thr = _sweep_threshold(logits, targets, torch.linspace(-4, 4, 81))
        assert isinstance(thr, float)
