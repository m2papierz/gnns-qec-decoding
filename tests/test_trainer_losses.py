"""Tests for trainer loss functions and metrics."""

import numpy as np
import pytest

from gnn.trainer import _roc_auc_score


class TestRocAucScore:
    """Tests of ROC AUC implementation."""

    def test_perfect_separation(self) -> None:
        """All positives scored higher than all negatives => AUC=1."""
        targets = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _roc_auc_score(targets, scores) == pytest.approx(1.0)

    def test_perfect_inverse(self) -> None:
        """All positives scored lower than all negatives => AUC=0."""
        targets = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _roc_auc_score(targets, scores) == pytest.approx(0.0)

    def test_random_baseline(self) -> None:
        """Identical scores => AUC=0.5."""
        targets = np.array([0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        assert _roc_auc_score(targets, scores) == pytest.approx(0.5)

    def test_all_positive_returns_half(self) -> None:
        targets = np.array([1, 1, 1])
        scores = np.array([0.1, 0.5, 0.9])
        assert _roc_auc_score(targets, scores) == 0.5

    def test_all_negative_returns_half(self) -> None:
        targets = np.array([0, 0, 0])
        scores = np.array([0.1, 0.5, 0.9])
        assert _roc_auc_score(targets, scores) == 0.5

    def test_known_value(self) -> None:
        """Hand-computed: 2 pos at ranks 3,4 out of 4 items, 2 neg.
        U = (3+4) - 2*3/2 = 4. AUC = 4/(2*2) = 1.0.
        """
        targets = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert _roc_auc_score(targets, scores) == pytest.approx(1.0)

    def test_tied_scores_handled(self) -> None:
        """Ties must be averaged, not arbitrarily broken."""
        targets = np.array([0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        # All tied => AUC should be exactly 0.5
        assert _roc_auc_score(targets, scores) == pytest.approx(0.5)

    def test_matches_sklearn_on_nontrivial(self) -> None:
        """Cross-check against sklearn on a non-trivial case."""
        try:
            from sklearn.metrics import roc_auc_score as sklearn_auc
        except ImportError:
            pytest.skip("sklearn not installed")

        rng = np.random.RandomState(42)
        targets = rng.randint(0, 2, size=200).astype(float)
        scores = rng.randn(200)

        ours = _roc_auc_score(targets, scores)
        theirs = sklearn_auc(targets, scores)
        assert ours == pytest.approx(theirs, abs=1e-10)

    def test_two_samples(self) -> None:
        """Minimal case: one pos, one neg."""
        assert _roc_auc_score(np.array([0, 1]), np.array([0.3, 0.7])) == pytest.approx(
            1.0
        )
        assert _roc_auc_score(np.array([1, 0]), np.array([0.3, 0.7])) == pytest.approx(
            0.0
        )
