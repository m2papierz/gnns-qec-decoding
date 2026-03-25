"""Sum-product belief propagation on detector factor graphs.

Computes posterior edge-error marginals ``P(e=1 | syndrome)`` using
LLR-domain message passing.  All inner loops are vectorised with NumPy;
a batch interface amortises overhead across syndrome shots sharing the
same graph topology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)

# Numerical constants
_CLIP: float = 1.0 - 1e-15
_TANH_CLIP: float = 30.0
_LOG_FLOOR: float = -60.0
_TINY: float = 1e-30


@dataclass(frozen=True)
class BPFactorGraph:
    """Precomputed bipartite factor-graph topology for BP.

    Variables = undirected edges (*U*), checks = detectors (*D*).
    Each variable touches at most two checks (its endpoint detectors);
    boundary endpoints (index >= *num_detectors*) are excluded.

    Build once per ``(distance, rounds, error_prob)`` setting via
    :meth:`build`, then pass to :func:`compute_bp_marginals` or
    :func:`compute_bp_marginals_batch` as ``_factor_graph``.

    Attributes
    ----------
    prior_llr : ndarray (U,) float64
        ``log((1-p)/p)`` per undirected edge.
    num_detectors : int
        Number of detector (check) nodes.
    num_edges : int
        Number of undirected variable edges (*U*).
    conn_check : ndarray (K,) int32
        Check (detector) index for each connection.
    conn_var : ndarray (K,) int32
        Variable (edge) index for each connection.
    num_connections : int
        Total connections (*K*) in the bipartite graph.
    """

    prior_llr: np.ndarray
    num_detectors: int
    num_edges: int
    conn_check: np.ndarray
    conn_var: np.ndarray
    num_connections: int

    @classmethod
    def build(
        cls,
        und_pairs: np.ndarray,
        edge_probs: np.ndarray,
        num_detectors: int,
    ) -> BPFactorGraph:
        """Construct factor graph from undirected edge structure.

        Parameters
        ----------
        und_pairs : ndarray (U, 2) int
            Undirected edge endpoints.
        edge_probs : ndarray (U,) float
            Prior error probability per edge.
        num_detectors : int
            Detector count; endpoints >= this value are boundary.
        """
        U = und_pairs.shape[0]

        # Vectorised connection building — no per-edge Python loop.
        ep0 = und_pairs[:, 0].astype(np.int32)
        ep1 = und_pairs[:, 1].astype(np.int32)
        eidx = np.arange(U, dtype=np.int32)

        valid0 = (ep0 >= 0) & (ep0 < num_detectors)
        valid1 = (ep1 >= 0) & (ep1 < num_detectors)

        conn_check = np.concatenate([ep0[valid0], ep1[valid1]])
        conn_var = np.concatenate([eidx[valid0], eidx[valid1]])

        # Prior LLRs
        p = np.clip(edge_probs.astype(np.float64), 1e-15, 1.0 - 1e-15)
        prior_llr = np.log((1.0 - p) / p)

        return cls(
            prior_llr=prior_llr,
            num_detectors=num_detectors,
            num_edges=U,
            conn_check=conn_check,
            conn_var=conn_var,
            num_connections=len(conn_check),
        )


def _scatter_add(
    keys: np.ndarray,
    values: np.ndarray,
    size: int,
) -> np.ndarray:
    """Sum *values* by integer *keys* into an array of length *size*.

    Handles both 1-D values ``(K,)`` and 2-D ``(K, B)`` for batch mode.
    Uses :func:`numpy.bincount` for the 1-D path and a flat-index trick
    (single ``bincount`` call on ``key + sample * size``) for 2-D —
    avoiding any Python-level per-sample loop.
    """
    if values.ndim == 1:
        out = np.bincount(keys, weights=values, minlength=size)
        return out.astype(np.float64, copy=False)

    # 2-D: (K, B) -> (size, B)
    _K, B = values.shape
    offsets = np.arange(B, dtype=np.int64) * size
    flat_keys = (keys[:, None] + offsets[None, :]).ravel()
    flat_out = np.bincount(flat_keys, weights=values.ravel(), minlength=size * B)
    return flat_out.reshape(B, size).T.astype(np.float64, copy=False)


def _product_excl_self(
    th: np.ndarray,
    conn_check: np.ndarray,
    num_detectors: int,
) -> np.ndarray:
    """Product-excluding-self of *th* within each check group.

    For every connection *k* belonging to check *d*, computes::

        prod_{k' in group(d), k' != k}  th[k']

    Uses a log-domain decomposition that is fully vectorised (no Python
    loop over checks) and numerically robust for near-zero values.

    Three cases per connection *k* in check *d*:

    A) *k* is non-tiny **and** *d* has zero tiny entries —
       ``log|excl| = sum_log[d] - log|th[k]|``;
       sign is the group sign product divided by ``sign(th[k])``.
    B) *k* is the sole tiny entry in *d* —
       ``log|excl| = sum_log[d]`` (the tiny was masked out of the sum);
       sign is the group sign product of non-tiny entries.
    C) *d* has >= 2 tiny entries, **or** *d* has 1 tiny and *k* is
       not it — product is zero.

    Parameters
    ----------
    th : ndarray, shape ``(K,)`` or ``(K, B)``
        ``tanh(v2c / 2)`` per connection (and per sample in batch mode).
    conn_check : ndarray (K,) int32
        Check index for each connection.
    num_detectors : int

    Returns
    -------
    ndarray, same shape as *th*
        Clipped to ``(-1+eps, 1-eps)`` for safe ``arctanh``.
    """
    D = num_detectors
    abs_th = np.abs(th)
    is_tiny = abs_th < _TINY

    # Log-magnitude of each factor (zeros masked out)
    safe_abs = np.where(is_tiny, 1.0, abs_th)
    log_abs = np.log(safe_abs)
    log_abs_nontiny = np.where(is_tiny, 0.0, log_abs)

    # Aggregate per check
    sum_log = _scatter_add(conn_check, log_abs_nontiny, D)
    tiny_cnt = _scatter_add(conn_check, is_tiny.astype(np.float64), D)

    # Sign tracking among non-tiny entries
    signs = np.sign(th)
    is_neg_nontiny = (~is_tiny) & (signs < 0)
    neg_cnt = _scatter_add(conn_check, is_neg_nontiny.astype(np.float64), D)
    grp_sign = np.where(neg_cnt.astype(np.intp) % 2 == 0, 1.0, -1.0)

    # Per-connection lookups
    tc = tiny_cnt[conn_check]
    sl = sum_log[conn_check]
    gs = grp_sign[conn_check]

    case_a = (~is_tiny) & (tc < 0.5)
    case_b = is_tiny & (tc > 0.5) & (tc < 1.5)

    log_excl = np.where(
        case_a,
        sl - log_abs,
        np.where(case_b, sl, _LOG_FLOOR),
    )
    sign_excl = np.where(
        case_a,
        gs * signs,
        np.where(case_b, gs, 0.0),
    )

    magnitude = np.exp(np.clip(log_excl, _LOG_FLOOR, 50.0))
    return np.clip(sign_excl * magnitude, -_CLIP, _CLIP)


def compute_bp_marginals(
    und_pairs: np.ndarray,
    edge_probs: np.ndarray,
    num_detectors: int,
    syndrome: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    _factor_graph: BPFactorGraph | None = None,
) -> np.ndarray:
    """Compute posterior edge-error marginals for a single syndrome.

    Runs loopy belief propagation on the detector-error factor graph.
    Each undirected edge is a variable node; each detector is a check
    node.

    Parameters
    ----------
    und_pairs : ndarray (U, 2) int
        Undirected edge endpoints.  Boundary nodes (>= *num_detectors*)
        are silently ignored.
    edge_probs : ndarray (U,) float
        Prior error probability per undirected edge.
    num_detectors : int
        Number of detector nodes (indices ``0 .. num_detectors-1``).
    syndrome : ndarray (num_detectors,) uint8/int
        Observed syndrome bits (0 or 1).
    max_iter : int
        Maximum BP iterations.
    tol : float
        Convergence threshold on max absolute marginal change.
    _factor_graph : BPFactorGraph, optional
        Precomputed topology — avoids redundant setup when called in a
        loop over syndromes that share the same graph.

    Returns
    -------
    ndarray (U,) float32
        Posterior marginals ``P(e=1 | syndrome)`` per undirected edge.
    """
    fg = _factor_graph or BPFactorGraph.build(und_pairs, edge_probs, num_detectors)
    U, K, D = fg.num_edges, fg.num_connections, fg.num_detectors

    if U == 0:
        return np.empty(0, dtype=np.float32)
    if K == 0:
        # No check-variable connections: marginals equal priors.
        return (1.0 / (1.0 + np.exp(fg.prior_llr))).astype(np.float32)

    s_sign = 1.0 - 2.0 * syndrome[:D].astype(np.float64)

    c2v = np.zeros(K, dtype=np.float64)
    prev = np.full(U, 0.5, dtype=np.float64)
    delta = 0.0

    for it in range(max_iter):
        # Variable -> check:  v2c[k] = prior[var] + sum(c2v) - c2v[k]
        total_c2v = _scatter_add(fg.conn_var, c2v, U)
        v2c = fg.prior_llr[fg.conn_var] + total_c2v[fg.conn_var] - c2v

        # Check -> variable:  c2v = s_d * 2 * arctanh(prod-excl-self)
        th = np.tanh(np.clip(v2c * 0.5, -_TANH_CLIP, _TANH_CLIP))
        pe = _product_excl_self(th, fg.conn_check, D)
        c2v = s_sign[fg.conn_check] * 2.0 * np.arctanh(pe)

        # Posterior marginals
        total_new = _scatter_add(fg.conn_var, c2v, U)
        marginals = 1.0 / (1.0 + np.exp(fg.prior_llr + total_new))

        delta = float(np.max(np.abs(marginals - prev)))
        prev = marginals.copy()
        if delta < tol:
            logger.debug("BP converged at iteration %d (delta=%.2e)", it + 1, delta)
            break
    else:
        logger.debug(
            "BP did not converge after %d iterations (delta=%.2e)",
            max_iter,
            delta,
        )

    return marginals.astype(np.float32)


def compute_bp_marginals_batch(
    und_pairs: np.ndarray,
    edge_probs: np.ndarray,
    num_detectors: int,
    syndromes: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    _factor_graph: BPFactorGraph | None = None,
) -> np.ndarray:
    """Compute posterior marginals for a batch of syndromes in parallel.

    Same algorithm as :func:`compute_bp_marginals` but all per-connection
    arrays carry an extra sample dimension *B*, turning vector ops into
    matrix ops and amortising Python/NumPy dispatch overhead.

    Parameters
    ----------
    und_pairs : ndarray (U, 2) int
        Undirected edge endpoints.
    edge_probs : ndarray (U,) float
        Prior error probability per undirected edge.
    num_detectors : int
        Number of detector nodes.
    syndromes : ndarray (B, num_detectors) uint8/int
        Batch of syndrome vectors.
    max_iter : int
        Maximum BP iterations.
    tol : float
        Convergence threshold (global max delta across all samples).
    _factor_graph : BPFactorGraph, optional
        Precomputed topology for reuse.

    Returns
    -------
    ndarray (B, U) float32
        Posterior marginals per sample and undirected edge.
    """
    fg = _factor_graph or BPFactorGraph.build(und_pairs, edge_probs, num_detectors)
    U, K, D = fg.num_edges, fg.num_connections, fg.num_detectors
    B = syndromes.shape[0]

    if U == 0:
        return np.empty((B, 0), dtype=np.float32)
    if K == 0:
        row = (1.0 / (1.0 + np.exp(fg.prior_llr))).astype(np.float32)
        return np.tile(row, (B, 1))

    # (D, B) — syndrome sign per check per sample
    s_sign = (1.0 - 2.0 * syndromes[:, :D].astype(np.float64)).T

    prior_col = fg.prior_llr[:, np.newaxis]  # (U, 1) for broadcasting

    c2v = np.zeros((K, B), dtype=np.float64)
    prev = np.full((U, B), 0.5, dtype=np.float64)
    delta = 0.0

    for it in range(max_iter):
        total_c2v = _scatter_add(fg.conn_var, c2v, U)  # (U, B)
        v2c = (
            prior_col[fg.conn_var]  # (K, 1)
            + total_c2v[fg.conn_var]  # (K, B)
            - c2v  # (K, B)
        )

        th = np.tanh(np.clip(v2c * 0.5, -_TANH_CLIP, _TANH_CLIP))
        pe = _product_excl_self(th, fg.conn_check, D)
        c2v = s_sign[fg.conn_check] * 2.0 * np.arctanh(pe)  # (K, B)

        total_new = _scatter_add(fg.conn_var, c2v, U)  # (U, B)
        marginals = 1.0 / (1.0 + np.exp(prior_col + total_new))

        delta = float(np.max(np.abs(marginals - prev)))
        prev = marginals.copy()
        if delta < tol:
            logger.debug(
                "BP batch converged at iteration %d (delta=%.2e)", it + 1, delta
            )
            break
    else:
        logger.debug(
            "BP batch did not converge after %d iterations (delta=%.2e)",
            max_iter,
            delta,
        )

    return marginals.T.astype(np.float32)  # (B, U)
