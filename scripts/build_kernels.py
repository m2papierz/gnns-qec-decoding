"""Build CUDA extensions for GNN-QEC decoding.

Usage (from project root):
    python scripts/build_kernels.py build_ext --inplace
"""

import os
import sys
from pathlib import Path


# Ensure we run from project root regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


CUDA_ARCH_FLAGS = [
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_80,code=sm_80",
]

cuda_src = PROJECT_ROOT / "src" / "kernels" / "cpp"

setup(
    name="qec-cuda-ops",
    ext_modules=[
        CUDAExtension(
            name="kernels._C",
            sources=[
                str(cuda_src / "fused_edge_features.cu"),
                str(cuda_src / "fused_norm_residual.cu"),
                str(cuda_src / "graph_norm_bce.cu"),
                str(cuda_src / "bindings.cpp"),
            ],
            include_dirs=[
                str(cuda_src),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-lineinfo",
                    "--ptxas-options=-v",
                ]
                + CUDA_ARCH_FLAGS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
