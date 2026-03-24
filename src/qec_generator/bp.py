"""Sum-product belief propagation on detector factor graphs.

Computes posterior edge-error marginals P(e=1 | syndrome) using
LLR-domain message passing.  Pure NumPy — no GPU dependencies.
"""

from __future__ import annotations

import logging

import numpy as np


logger = logging.getLogger(__name__)

# Numerical guard for tanh / arctanh arguments.
_CLIP = 1.0 - 1e-15


def compute_bp_marginals(
    und_pairs: np.ndarray,
    edge_probs: np.ndarray,
    num_detectors: int,
    syndrome: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute posterior edge-error marginals via sum-product BP.

    Runs loopy belief propagation on the detector-error factor graph
    derived from the detector error model.  Each undirected edge is a
    variable node; each detector is a check node.

    Parameters
    ----------
    und_pairs : ndarray, shape ``(U, 2)``, int
        Undirected edge endpoints.  Entries ≥ *num_detectors* (boundary
        nodes) are silently ignored.
    edge_probs : ndarray, shape ``(U,)``, float
        Prior error probability per undirected edge.
    num_detectors : int
        Number of detector nodes (indices ``0 .. num_detectors-1``).
    syndrome : ndarray, shape ``(num_detectors,)``, uint8 or int
        Observed syndrome bits (0 or 1).
    max_iter : int
        Maximum BP iterations.
    tol : float
        Convergence threshold on max absolute marginal change.

    Returns
    -------
    ndarray, shape ``(U,)``, float32
        Posterior marginals ``P(e=1 | syndrome)`` per undirected edge.
    """
    U = und_pairs.shape[0]
    if U == 0:
        return np.empty(0, dtype=np.float32)

    # Build adjacency lists
    # check_to_vars[d] = list of variable (edge) indices touching detector d
    # var_to_checks[e] = list of check (detector) indices touching edge e
    check_to_vars: list[list[int]] = [[] for _ in range(num_detectors)]
    var_to_checks: list[list[int]] = [[] for _ in range(U)]

    for e in range(U):
        for endpoint in (int(und_pairs[e, 0]), int(und_pairs[e, 1])):
            if 0 <= endpoint < num_detectors:
                check_to_vars[endpoint].append(e)
                var_to_checks[e].append(endpoint)

    # Prior LLRs
    p = np.clip(edge_probs.astype(np.float64), 1e-15, 1.0 - 1e-15)
    prior_llr = np.log((1.0 - p) / p)  # (U,)

    # Syndrome sign: +1 for s=0, -1 for s=1.
    s_sign = 1.0 - 2.0 * syndrome[:num_detectors].astype(np.float64)

    # Message arrays (sparse: only stored for existing edges)
    # c2v[d][local_idx] = check=>variable message for the local_idx-th
    # variable in check_to_vars[d].
    c2v: list[np.ndarray] = [
        np.zeros(len(nbrs), dtype=np.float64) for nbrs in check_to_vars
    ]

    prev_marginals = np.full(U, 0.5, dtype=np.float64)

    for iteration in range(max_iter):
        # Variable => Check
        # v2c[e=>d] = prior_llr[e] + sum_{d' != d} c2v[d'=>e]
        #
        # First compute total incoming c2v per variable.
        total_c2v = np.zeros(U, dtype=np.float64)
        for d in range(num_detectors):
            for li, e in enumerate(check_to_vars[d]):
                total_c2v[e] += c2v[d][li]

        # Check => Variable
        for d in range(num_detectors):
            nbrs = check_to_vars[d]
            n_nbrs = len(nbrs)
            if n_nbrs == 0:
                continue

            # Collect v2c messages for this check.
            v2c_vals = np.empty(n_nbrs, dtype=np.float64)
            for li, e in enumerate(nbrs):
                # v2c[e=>d] = prior_llr[e] + total_c2v[e] - c2v[d=>e]
                v2c_vals[li] = prior_llr[e] + total_c2v[e] - c2v[d][li]

            # Product of tanh(v2c/2) excluding self.
            tanh_half = np.tanh(np.clip(v2c_vals * 0.5, -30.0, 30.0))
            prod_all = np.prod(tanh_half)

            for li in range(n_nbrs):
                # Exclude self: prod_all / tanh_half[li]
                if abs(tanh_half[li]) > 1e-30:
                    prod_excl = prod_all / tanh_half[li]
                else:
                    # Recompute without this element to avoid 0/0.
                    prod_excl = 1.0
                    for lj in range(n_nbrs):
                        if lj != li:
                            prod_excl *= tanh_half[lj]

                prod_excl = np.clip(prod_excl, -_CLIP, _CLIP)
                c2v[d][li] = s_sign[d] * 2.0 * np.arctanh(prod_excl)

        # Posterior marginals
        total_c2v_new = np.zeros(U, dtype=np.float64)
        for d in range(num_detectors):
            for li, e in enumerate(check_to_vars[d]):
                total_c2v_new[e] += c2v[d][li]

        posterior_llr = prior_llr + total_c2v_new
        marginals = 1.0 / (1.0 + np.exp(posterior_llr))  # σ(-posterior_llr)

        # Convergence check
        delta = float(np.max(np.abs(marginals - prev_marginals)))
        prev_marginals = marginals.copy()

        if delta < tol:
            logger.debug(
                "BP converged at iteration %d (delta=%.2e)", iteration + 1, delta
            )
            break
    else:
        logger.debug(
            "BP did not converge after %d iterations (delta=%.2e)", max_iter, delta
        )

    return marginals.astype(np.float32)
