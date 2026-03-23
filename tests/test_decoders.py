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


def _gpu_available() -> bool:
    """Check if cupy + CUDA GPU is available."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


_skip_no_gpu = pytest.mark.skipif(not _gpu_available(), reason="no CUDA GPU")


@_skip_no_gpu
class TestTNDecoder:
    """Tests for TNDecoder (requires GPU for cuTensorNet contraction)."""

    def test_decode_shape(self, simple_config: DecoderConfig) -> None:
        from decoders.tensor_network import TNDecoder

        und_pairs = np.array([[0, 1], [1, 2], [0, 8]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.1, 0.05])
        decoder = TNDecoder(simple_config, und_pairs, edge_probs)
        result = decoder.decode(np.zeros(8, dtype=np.uint8))
        assert result.shape == (1,)

    def test_decode_batch_shape(self, simple_config: DecoderConfig) -> None:
        from decoders.tensor_network import TNDecoder

        und_pairs = np.array([[0, 1], [1, 2], [0, 8]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.1, 0.05])
        decoder = TNDecoder(simple_config, und_pairs, edge_probs)
        results = decoder.decode_batch(np.zeros((3, 8), dtype=np.uint8))
        assert results.shape == (3, 1)

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

    def test_zero_syndrome_low_marginals(self) -> None:
        from decoders.tensor_network import TNDecoder

        config = DecoderConfig(
            num_detectors=4,
            num_observables=1,
            has_boundary=False,
        )
        und_pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.1])
        decoder = TNDecoder(config, und_pairs, edge_probs)

        m = decoder._compute_edge_marginals(np.zeros(4, dtype=np.uint8))
        assert np.all(m < 0.5), "Zero syndrome should yield low marginals"

    def test_name(self) -> None:
        from decoders.tensor_network import TNDecoder

        config = DecoderConfig(
            num_detectors=2,
            num_observables=1,
            has_boundary=False,
        )
        decoder = TNDecoder(
            config,
            np.array([[0, 1]], dtype=np.int64),
            np.array([0.1]),
        )
        assert decoder.name == "TNDecoder"


class TestCheckTensor:
    """Unit tests for parity-check tensor construction."""

    def test_degree1_parity0(self) -> None:
        from decoders.tensor_network import TNDecoder

        t = TNDecoder._build_check_tensor(degree=1, target_parity=0)
        assert t.shape == (2,)
        np.testing.assert_array_equal(t, [1.0, 0.0])

    def test_degree1_parity1(self) -> None:
        from decoders.tensor_network import TNDecoder

        t = TNDecoder._build_check_tensor(degree=1, target_parity=1)
        np.testing.assert_array_equal(t, [0.0, 1.0])

    def test_degree2_parity0(self) -> None:
        from decoders.tensor_network import TNDecoder

        t = TNDecoder._build_check_tensor(degree=2, target_parity=0)
        assert t.shape == (2, 2)
        np.testing.assert_array_equal(t, [[1, 0], [0, 1]])

    def test_degree2_parity1(self) -> None:
        from decoders.tensor_network import TNDecoder

        t = TNDecoder._build_check_tensor(degree=2, target_parity=1)
        np.testing.assert_array_equal(t, [[0, 1], [1, 0]])

    def test_degree3_count(self) -> None:
        from decoders.tensor_network import TNDecoder

        t = TNDecoder._build_check_tensor(degree=3, target_parity=0)
        assert t.shape == (2, 2, 2)
        assert t.sum() == 4

    def test_degree4_symmetry(self) -> None:
        from decoders.tensor_network import TNDecoder

        t0 = TNDecoder._build_check_tensor(degree=4, target_parity=0)
        t1 = TNDecoder._build_check_tensor(degree=4, target_parity=1)
        assert t0.sum() == 8
        assert t1.sum() == 8
        np.testing.assert_array_equal(t0 + t1, np.ones([2] * 4))


@_skip_no_gpu
class TestExactMarginals:
    """Analytically verified exact marginal tests."""

    def test_line_graph_both_fire(self) -> None:
        """det0 -- e0 -- det1 -- e1 -- det2, syndrome [1,0,1].

        Only consistent solution: e0=1, e1=1.
        """
        from decoders.tensor_network import TNDecoder

        config = DecoderConfig(
            num_detectors=3,
            num_observables=1,
            has_boundary=False,
        )
        und_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.2])
        decoder = TNDecoder(config, und_pairs, edge_probs)

        m = decoder._compute_edge_marginals(np.array([1, 0, 1], dtype=np.uint8))
        np.testing.assert_allclose(m[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(m[1], 1.0, atol=1e-6)

    def test_line_graph_zero_syndrome(self) -> None:
        """det0 -- e0 -- det1 -- e1 -- det2, syndrome [0,0,0].

        Only consistent solution: e0=0, e1=0.
        """
        from decoders.tensor_network import TNDecoder

        config = DecoderConfig(
            num_detectors=3,
            num_observables=1,
            has_boundary=False,
        )
        und_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.2])
        decoder = TNDecoder(config, und_pairs, edge_probs)

        m = decoder._compute_edge_marginals(np.zeros(3, dtype=np.uint8))
        np.testing.assert_allclose(m[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(m[1], 0.0, atol=1e-6)

    def test_line_graph_single_error(self) -> None:
        """det0 -- e0 -- det1 -- e1 -- det2, syndrome [1,1,0].

        Only consistent solution: e0=1, e1=0.
        """
        from decoders.tensor_network import TNDecoder

        config = DecoderConfig(
            num_detectors=3,
            num_observables=1,
            has_boundary=False,
        )
        und_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.2])
        decoder = TNDecoder(config, und_pairs, edge_probs)

        m = decoder._compute_edge_marginals(np.array([1, 1, 0], dtype=np.uint8))
        np.testing.assert_allclose(m[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(m[1], 0.0, atol=1e-6)

    def test_cycle_ambiguous(self) -> None:
        """4-node cycle, syndrome [1,1,0,0].

        Two solutions:
          A: e0=1, rest=0  -> weight = 0.1 * 0.9^3
          B: e1=e2=e3=1    -> weight = 0.9 * 0.1^3
        """
        from decoders.tensor_network import TNDecoder

        config = DecoderConfig(
            num_detectors=4,
            num_observables=1,
            has_boundary=False,
        )
        und_pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
        edge_probs = np.full(4, 0.1)
        decoder = TNDecoder(config, und_pairs, edge_probs)

        m = decoder._compute_edge_marginals(np.array([1, 1, 0, 0], dtype=np.uint8))

        wa = 0.1 * 0.9**3  # solution A
        wb = 0.9 * 0.1**3  # solution B
        z = wa + wb

        np.testing.assert_allclose(m[0], wa / z, atol=1e-6)
        np.testing.assert_allclose(m[1], wb / z, atol=1e-6)
        np.testing.assert_allclose(m[2], wb / z, atol=1e-6)
        np.testing.assert_allclose(m[3], wb / z, atol=1e-6)

    def test_boundary_edge(self) -> None:
        """det0 -- e0 -- det1 -- e1 -- boundary(2), syndrome [1,1].

        check_0: e0 = 1.  check_1: e0 XOR e1 = 1 -> e1 = 0.
        """
        from decoders.tensor_network import TNDecoder

        config = DecoderConfig(
            num_detectors=2,
            num_observables=1,
            has_boundary=True,
            boundary_node=2,
        )
        und_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.1])
        decoder = TNDecoder(config, und_pairs, edge_probs)

        m = decoder._compute_edge_marginals(np.array([1, 1], dtype=np.uint8))
        np.testing.assert_allclose(m[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(m[1], 0.0, atol=1e-6)

    def test_boundary_edge_flipped(self) -> None:
        """det0 -- e0 -- det1 -- e1 -- boundary(2), syndrome [0,1].

        check_0: e0 = 0.  check_1: e0 XOR e1 = 1 -> e1 = 1.
        """
        from decoders.tensor_network import TNDecoder

        config = DecoderConfig(
            num_detectors=2,
            num_observables=1,
            has_boundary=True,
            boundary_node=2,
        )
        und_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.1])
        decoder = TNDecoder(config, und_pairs, edge_probs)

        m = decoder._compute_edge_marginals(np.array([0, 1], dtype=np.uint8))
        np.testing.assert_allclose(m[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(m[1], 1.0, atol=1e-6)

    def test_soft_label_generator_matches_decoder(self) -> None:
        """TNSoftLabelGenerator produces same marginals as TNDecoder."""
        from decoders.tensor_network import TNDecoder, TNSoftLabelGenerator

        config = DecoderConfig(
            num_detectors=3,
            num_observables=1,
            has_boundary=False,
        )
        und_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.2])

        gen = TNSoftLabelGenerator(config, und_pairs, edge_probs)
        decoder = TNDecoder(config, und_pairs, edge_probs)

        syndromes = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 0]], dtype=np.uint8)
        labels = gen.generate_soft_labels(syndromes)

        for i in range(syndromes.shape[0]):
            expected = decoder._compute_edge_marginals(syndromes[i])
            np.testing.assert_allclose(labels[i], expected, atol=1e-6)
