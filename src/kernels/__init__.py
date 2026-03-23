"""Custom CUDA kernels for GNN-QEC decoding.

Provides fused implementations of compute-intensive operations:

- ``symmetric_edge_features``: gather + sum/abs-diff/concat in one launch
- ``fused_norm_residual_dropout``: LayerNorm + dropout + residual add
- ``graph_normalized_bce``: per-graph BCE with scatter reduction
"""

try:
    from kernels._C import (
        fused_norm_residual_dropout,
        fused_symmetric_edge_features,
        graph_normalized_bce,
    )

    AVAILABLE = True
except ImportError:
    AVAILABLE = False
