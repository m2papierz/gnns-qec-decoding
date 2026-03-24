"""Tests for decoder implementations."""

import numpy as np
import pytest

from decoders.base import DecoderConfig
from decoders.mwpm import MWPMDecoder


@pytest.fixture
def simple_config() -> DecoderConfig:
    """Minimal d=3 decoder config."""
    return DecoderConfig(
        num_detectors=8,
        num_observables=1,
        has_boundary=True,
        boundary_node=8,
    )


@pytest.fixture
def simple_graph() -> tuple[np.ndarray, np.ndarray]:
    """Small undirected edge set with weights."""
    und_pairs = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 8]],
        dtype=np.int64,
    )
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)
    return und_pairs, weights


class TestDecoderConfig:
    def test_frozen(self, simple_config: DecoderConfig) -> None:
        with pytest.raises(Exception):
            simple_config.num_detectors = 10  # type: ignore[misc]

    def test_fields(self, simple_config: DecoderConfig) -> None:
        assert simple_config.num_detectors == 8
        assert simple_config.num_observables == 1
        assert simple_config.has_boundary is True
        assert simple_config.boundary_node == 8


class TestMWPMDecoder:
    def test_decode_returns_correct_shape(
        self,
        simple_config: DecoderConfig,
        simple_graph: tuple,
    ) -> None:
        und_pairs, weights = simple_graph
        decoder = MWPMDecoder(simple_config, und_pairs, weights)
        syndrome = np.zeros(8, dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert result.shape == (1,)

    def test_decode_batch_shape(
        self,
        simple_config: DecoderConfig,
        simple_graph: tuple,
    ) -> None:
        und_pairs, weights = simple_graph
        decoder = MWPMDecoder(simple_config, und_pairs, weights)
        syndromes = np.zeros((5, 8), dtype=np.uint8)
        results = decoder.decode_batch(syndromes)
        assert results.shape == (5, 1)

    def test_zero_syndrome_gives_zero(
        self,
        simple_config: DecoderConfig,
        simple_graph: tuple,
    ) -> None:
        und_pairs, weights = simple_graph
        decoder = MWPMDecoder(simple_config, und_pairs, weights)
        syndrome = np.zeros(8, dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert np.all(result == 0)

    def test_name(
        self,
        simple_config: DecoderConfig,
        simple_graph: tuple,
    ) -> None:
        und_pairs, weights = simple_graph
        decoder = MWPMDecoder(simple_config, und_pairs, weights)
        assert decoder.name == "MWPMDecoder"

    def test_from_gnn_logits(self) -> None:
        config = DecoderConfig(
            num_detectors=3,
            num_observables=1,
            has_boundary=True,
            boundary_node=3,
        )
        und_pairs = np.array([[0, 1], [1, 2], [0, 3]], dtype=np.int64)
        directed_logits = np.array([-2.0, -2.0, -1.0, -1.0, 0.5, 0.5], dtype=np.float64)
        dir_to_undir = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

        decoder = MWPMDecoder.from_gnn_logits(
            config, und_pairs, directed_logits, dir_to_undir, 3
        )
        assert isinstance(decoder, MWPMDecoder)
        result = decoder.decode(np.zeros(3, dtype=np.uint8))
        assert result.shape == (1,)
