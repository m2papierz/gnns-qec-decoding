"""Tests for focal loss and weighted sampler for the direct training case."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from gnn.trainer import FocalBCEWithLogitsLoss, build_direct_sampler


class TestFocalBCEWithLogitsLoss:
    """Tests for the sigmoid focal loss."""

    @pytest.fixture
    def focal(self) -> FocalBCEWithLogitsLoss:
        return FocalBCEWithLogitsLoss(alpha=0.25, gamma=2.0)

    def test_output_is_scalar(self, focal: FocalBCEWithLogitsLoss) -> None:
        logits = torch.randn(16)
        target = torch.randint(0, 2, (16,)).float()
        loss = focal(logits, target)
        assert loss.ndim == 0

    def test_loss_nonnegative(self, focal: FocalBCEWithLogitsLoss) -> None:
        logits = torch.randn(32)
        target = torch.randint(0, 2, (32,)).float()
        assert focal(logits, target).item() >= 0.0

    def test_gradient_flows(self, focal: FocalBCEWithLogitsLoss) -> None:
        logits = torch.randn(16, requires_grad=True)
        target = torch.randint(0, 2, (16,)).float()
        focal(logits, target).backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_perfect_positive_near_zero(
        self,
        focal: FocalBCEWithLogitsLoss,
    ) -> None:
        """Confident correct positive prediction → near-zero loss."""
        loss = focal(torch.full((10,), 10.0), torch.ones(10))
        assert loss.item() < 1e-4

    def test_perfect_negative_near_zero(
        self,
        focal: FocalBCEWithLogitsLoss,
    ) -> None:
        """Confident correct negative prediction → near-zero loss."""
        loss = focal(torch.full((10,), -10.0), torch.zeros(10))
        assert loss.item() < 1e-4

    def test_suppresses_easy_negatives(
        self,
        focal: FocalBCEWithLogitsLoss,
    ) -> None:
        """Focal loss on easy examples is strictly less than BCE."""
        bce = nn.BCEWithLogitsLoss()
        logits = torch.full((100,), -3.0)
        target = torch.zeros(100)
        assert focal(logits, target).item() < bce(logits, target).item()

    def test_hard_example_ratio(
        self,
        focal: FocalBCEWithLogitsLoss,
    ) -> None:
        """At p=0.5: focal = alpha * (1-0.5)^gamma * BCE = 0.0625 * BCE."""
        bce = nn.BCEWithLogitsLoss()
        logits = torch.zeros(100)
        target = torch.ones(100)
        ratio = focal(logits, target).item() / bce(logits, target).item()
        assert ratio == pytest.approx(0.0625, abs=0.01)

    def test_gamma_zero_is_weighted_bce(self) -> None:
        """With gamma=0, focal reduces to alpha-weighted BCE."""
        focal_g0 = FocalBCEWithLogitsLoss(alpha=0.5, gamma=0.0)
        bce = nn.BCEWithLogitsLoss()
        logits = torch.randn(50)
        target = torch.randint(0, 2, (50,)).float()
        assert focal_g0(logits, target).item() == pytest.approx(
            0.5 * bce(logits, target).item(), abs=0.01
        )

    def test_multidim_target(self, focal: FocalBCEWithLogitsLoss) -> None:
        """Works with multi-observable targets."""
        logits = torch.randn(20)
        target = torch.randint(0, 2, (20,)).float()
        loss = focal(logits, target)
        assert loss.ndim == 0
        assert loss.item() >= 0.0


class TestBuildDirectSampler:
    """Tests for WeightedRandomSampler construction."""

    @staticmethod
    def _make_mock_dataset(
        n_per_setting: int = 100,
        distances: tuple[int, ...] = (3, 5, 7),
        positive_rates: tuple[float, ...] = (0.15, 0.10, 0.05),
    ) -> object:
        """Build a minimal mock that satisfies build_direct_sampler's API.

        Returns an object with ``_setting_id``, ``_shot_id``, ``settings``,
        and ``_get_split_arrays`` matching ``MixedSurfaceCodeDataset``.
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Info:
            distance: int

        n_settings = len(distances)
        total = n_per_setting * n_settings

        setting_ids = np.repeat(np.arange(n_settings, dtype=np.int32), n_per_setting)
        shot_ids = np.tile(np.arange(n_per_setting, dtype=np.int32), n_settings)

        # Build per-setting logical arrays.
        split_cache = {}
        settings = {}
        rng = np.random.default_rng(42)

        for i, (d, pr) in enumerate(zip(distances, positive_rates)):
            settings[i] = _Info(distance=d)
            logical = (rng.random(n_per_setting) < pr).astype(np.float32)
            syndrome = np.zeros((n_per_setting, 1), dtype=np.float32)  # unused
            split_cache[i] = (syndrome, logical)

        class MockDS:
            def __len__(self) -> int:
                return total

        ds = MockDS()
        ds._setting_id = setting_ids
        ds._shot_id = shot_ids
        ds.settings = settings
        ds._split_cache = split_cache
        ds._get_split_arrays = lambda sid: split_cache[sid]

        return ds

    def test_returns_sampler(self) -> None:
        from torch.utils.data import WeightedRandomSampler

        ds = self._make_mock_dataset()
        sampler = build_direct_sampler(ds, indices=None, pos_oversample_cap=10.0)
        assert isinstance(sampler, WeightedRandomSampler)

    def test_num_samples_matches(self) -> None:
        ds = self._make_mock_dataset(n_per_setting=50)
        sampler = build_direct_sampler(ds, indices=None)
        assert sampler.num_samples == 150  # 3 settings × 50

    def test_num_samples_with_subset(self) -> None:
        ds = self._make_mock_dataset(n_per_setting=100)
        indices = list(range(0, 300, 3))  # 100 samples
        sampler = build_direct_sampler(ds, indices=indices)
        assert sampler.num_samples == 100

    def test_all_weights_positive(self) -> None:
        ds = self._make_mock_dataset()
        sampler = build_direct_sampler(ds, indices=None)
        weights = list(sampler.weights)
        assert all(w > 0 for w in weights)

    def test_larger_distance_gets_higher_weight(self) -> None:
        """Mean weight for d=7 > mean weight for d=3."""
        ds = self._make_mock_dataset(
            n_per_setting=200,
            positive_rates=(0.0, 0.0, 0.0),  # no class weight effect
        )
        sampler = build_direct_sampler(ds, indices=None)
        w = np.array(list(sampler.weights))
        w_d3 = w[:200].mean()
        w_d7 = w[400:600].mean()
        assert w_d7 > w_d3

    def test_positive_gets_higher_weight(self) -> None:
        """Positive samples weighed higher than negatives in same setting."""
        ds = self._make_mock_dataset(
            n_per_setting=200,
            distances=(3,),
            positive_rates=(0.10,),
        )
        sampler = build_direct_sampler(ds, indices=None)
        w = np.array(list(sampler.weights))

        # Find pos/neg from the logical array.
        _, logical = ds._get_split_arrays(0)
        pos_mask = logical > 0.5
        assert w[pos_mask].mean() > w[~pos_mask].mean()

    def test_cap_limits_positive_weight(self) -> None:
        """Positive weight is capped at pos_oversample_cap."""
        ds = self._make_mock_dataset(
            n_per_setting=1000,
            distances=(3,),
            positive_rates=(0.001,),  # ~1 positive out of 1000
        )
        sampler = build_direct_sampler(ds, indices=None, pos_oversample_cap=5.0)
        w = np.array(list(sampler.weights))
        # Max ratio should be ~5.0 (cap), not 999.
        assert w.max() / w[w > 0].min() <= 5.5
