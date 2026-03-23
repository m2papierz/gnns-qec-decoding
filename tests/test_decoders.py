"""Tests for decoder implementations."""

import numpy as np
import pytest

from decoders.base import DecoderConfig
from decoders.mwpm import MWPMDecoder


def _tn_importable() -> bool:
    try:
        from decoders.tensor_network import _check_cuquantum

        return _check_cuquantum()
    except ImportError:
        return False


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
        self, simple_config: DecoderConfig, simple_graph: tuple
    ) -> None:
        und_pairs, weights = simple_graph
        decoder = MWPMDecoder(simple_config, und_pairs, weights)
        syndrome = np.zeros(8, dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert result.shape == (1,)

    def test_decode_batch_shape(
        self, simple_config: DecoderConfig, simple_graph: tuple
    ) -> None:
        und_pairs, weights = simple_graph
        decoder = MWPMDecoder(simple_config, und_pairs, weights)
        syndromes = np.zeros((5, 8), dtype=np.uint8)
        results = decoder.decode_batch(syndromes)
        assert results.shape == (5, 1)

    def test_zero_syndrome_gives_zero(
        self, simple_config: DecoderConfig, simple_graph: tuple
    ) -> None:
        und_pairs, weights = simple_graph
        decoder = MWPMDecoder(simple_config, und_pairs, weights)
        syndrome = np.zeros(8, dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert np.all(result == 0)

    def test_name(self, simple_config: DecoderConfig, simple_graph: tuple) -> None:
        und_pairs, weights = simple_graph
        decoder = MWPMDecoder(simple_config, und_pairs, weights)
        assert decoder.name == "MWPMDecoder"

    def test_from_gnn_logits(self, simple_config: DecoderConfig) -> None:
        und_pairs = np.array([[0, 1], [1, 2], [0, 8]], dtype=np.int64)
        directed_logits = np.array([-2.0, -2.0, -1.0, -1.0, 0.5, 0.5], dtype=np.float64)
        dir_to_undir = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

        decoder = MWPMDecoder.from_gnn_logits(
            simple_config, und_pairs, directed_logits, dir_to_undir, 3
        )
        assert isinstance(decoder, MWPMDecoder)
        result = decoder.decode(np.zeros(8, dtype=np.uint8))
        assert result.shape == (1,)


class TestTNDecoder:
    """Tests for TNDecoder (requires cuquantum)."""

    def test_instantiation_requires_cuquantum(self) -> None:
        """TNDecoder raises ImportError if cuquantum is absent."""
        if _tn_importable():
            pytest.skip("cuquantum is installed")
        from decoders.tensor_network import TNDecoder

        config = DecoderConfig(
            num_detectors=4,
            num_observables=1,
            has_boundary=False,
        )
        with pytest.raises(ImportError, match="cuquantum"):
            TNDecoder(
                config,
                np.array([[0, 1]], dtype=np.int64),
                np.array([0.1]),
            )

    @pytest.mark.skipif(
        not _tn_importable(),
        reason="cuquantum not installed",
    )
    def test_decode_shape(self, simple_config: DecoderConfig) -> None:
        from decoders.tensor_network import TNDecoder

        und_pairs = np.array([[0, 1], [1, 2], [0, 8]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.1, 0.05])
        decoder = TNDecoder(simple_config, und_pairs, edge_probs)
        result = decoder.decode(np.zeros(8, dtype=np.uint8))
        assert result.shape == (1,)

    @pytest.mark.skipif(
        not _tn_importable(),
        reason="cuquantum not installed",
    )
    def test_decode_batch_shape(self, simple_config: DecoderConfig) -> None:
        from decoders.tensor_network import TNDecoder

        und_pairs = np.array([[0, 1], [1, 2], [0, 8]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.1, 0.05])
        decoder = TNDecoder(simple_config, und_pairs, edge_probs)
        results = decoder.decode_batch(np.zeros((3, 8), dtype=np.uint8))
        assert results.shape == (3, 1)

    @pytest.mark.skipif(
        not _tn_importable(),
        reason="cuquantum not installed",
    )
    def test_soft_label_generator(self, simple_config: DecoderConfig) -> None:
        from decoders.tensor_network import TNSoftLabelGenerator

        und_pairs = np.array([[0, 1], [1, 2], [0, 8]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.1, 0.05])
        gen = TNSoftLabelGenerator(simple_config, und_pairs, edge_probs)
        syndromes = np.zeros((5, 8), dtype=np.uint8)
        labels = gen.generate_soft_labels(syndromes)
        assert labels.shape == (5, 3)
        assert labels.dtype == np.float32
        assert np.all(labels >= 0) and np.all(labels <= 1)
