"""Tests for stratified subset selection in trainer."""

import numpy as np
import pytest

from gnn.trainer import build_stratified_subset


class TestBuildStratifiedSubset:
    """Tests for build_stratified_subset."""

    def test_exact_count(self) -> None:
        """Returned array has exactly min(max_samples, N) elements."""
        ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        result = build_stratified_subset(ids, 6, seed=0)
        assert len(result) == 6

    def test_all_settings_represented(self) -> None:
        """Every setting ID appears in the result when quota allows."""
        ids = np.array([0] * 100 + [1] * 100 + [2] * 100, dtype=np.int32)
        result = build_stratified_subset(ids, 30, seed=42)
        selected_sids = set(ids[result])
        assert selected_sids == {0, 1, 2}

    def test_balanced_allocation(self) -> None:
        """Settings get roughly equal share of the budget."""
        ids = np.array([0] * 500 + [1] * 500 + [2] * 500, dtype=np.int32)
        result = build_stratified_subset(ids, 300, seed=7)
        counts = {s: int(np.sum(ids[result] == s)) for s in [0, 1, 2]}
        assert all(90 <= c <= 110 for c in counts.values()), counts

    def test_small_setting_gets_all(self) -> None:
        """A setting with fewer samples than its quota donates leftovers."""
        ids = np.array([0] * 5 + [1] * 500, dtype=np.int32)
        result = build_stratified_subset(ids, 100, seed=3)
        assert len(result) == 100
        count_0 = int(np.sum(ids[result] == 0))
        assert count_0 == 5  # all of setting 0

    def test_max_samples_exceeds_total(self) -> None:
        """When max_samples >= N, return all indices."""
        ids = np.array([0, 0, 1, 1], dtype=np.int32)
        result = build_stratified_subset(ids, 100, seed=0)
        assert len(result) == 4
        assert set(result) == {0, 1, 2, 3}

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical output."""
        ids = np.array([0] * 50 + [1] * 50, dtype=np.int32)
        r1 = build_stratified_subset(ids, 30, seed=99)
        r2 = build_stratified_subset(ids, 30, seed=99)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_differs(self) -> None:
        """Different seeds produce different orderings (with high probability)."""
        ids = np.array([0] * 200 + [1] * 200, dtype=np.int32)
        r1 = build_stratified_subset(ids, 100, seed=1)
        r2 = build_stratified_subset(ids, 100, seed=2)
        assert not np.array_equal(r1, r2)

    def test_indices_in_bounds(self) -> None:
        """All returned indices are valid for the original array."""
        ids = np.array([0] * 40 + [1] * 60 + [2] * 30, dtype=np.int32)
        result = build_stratified_subset(ids, 50, seed=5)
        assert result.min() >= 0
        assert result.max() < len(ids)

    def test_no_duplicates(self) -> None:
        """Returned indices are unique."""
        ids = np.array([0] * 100 + [1] * 100 + [2] * 100, dtype=np.int32)
        result = build_stratified_subset(ids, 150, seed=8)
        assert len(result) == len(set(result))

    def test_single_setting(self) -> None:
        """Works with only one setting ID."""
        ids = np.array([0] * 100, dtype=np.int32)
        result = build_stratified_subset(ids, 30, seed=0)
        assert len(result) == 30
        assert len(set(result)) == 30

    def test_max_samples_one(self) -> None:
        """Edge case: requesting exactly 1 sample."""
        ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        result = build_stratified_subset(ids, 1, seed=0)
        assert len(result) == 1

    def test_invalid_max_samples(self) -> None:
        """Raises on max_samples < 1."""
        ids = np.array([0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="max_samples"):
            build_stratified_subset(ids, 0, seed=0)

    def test_empty_setting_ids(self) -> None:
        """Raises on empty input."""
        with pytest.raises(ValueError, match="non-empty"):
            build_stratified_subset(np.array([], dtype=np.int32), 10, seed=0)

    def test_many_settings_few_samples(self) -> None:
        """When num_settings > max_samples, still works without error."""
        ids = np.arange(20, dtype=np.int32)  # 20 settings, 1 sample each
        result = build_stratified_subset(ids, 5, seed=0)
        assert len(result) == 5
        assert len(set(ids[result])) == 5  # 5 distinct settings

    def test_dtype_int64(self) -> None:
        """Output dtype is int64 for direct indexing."""
        ids = np.array([0, 0, 1, 1], dtype=np.int32)
        result = build_stratified_subset(ids, 3, seed=0)
        assert result.dtype == np.int64
