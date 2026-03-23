/*
 * Warp-level reduction primitives for sm_89 (Ada Lovelace).
 *
 * Used by LayerNorm and BCE kernels for parallel summation.
 * All threads in a warp must participate (no divergent branches).
 */

#pragma once
#include <cuda_runtime.h>

namespace qec {
    __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
        for (int offset { 16 }; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        }
        return val;
    }

    __device__ __forceinline__ float block_reduce_sum(float val) {
        __shared__ float shared[32];

        int lane    { threadIdx.x % 32 };
        int warp_id { threadIdx.x / 32 };

        val = warp_reduce_sum(val);

        if (lane == 0) {
            shared[warp_id] = val;
        }
        __syncthreads();

        int num_warps { static_cast<int>((blockDim.x + 31) / 32) };
        val = (lane < num_warps) ? shared[lane] : 0.0f;

        if (warp_id == 0) {
            val = warp_reduce_sum(val);
        }
        return val;
    }
}  // namespace qec
