/*
 * Fused LayerNorm + residual + dropout.
 *
 * Replaces 3 separate kernels: LayerNorm -> Dropout -> Add.
 * Two-pass mean/variance via warp-level reductions,
 * then fused normalize + scale + dropout + residual in pass 3.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include "warp_reduce.cuh"

namespace qec {
    __global__ void fused_layernorm_residual_dropout_kernel(
        const float* __restrict__ input,
        const float* __restrict__ residual,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ output,
        const int N,
        const int H,
        const float dropout_p,
        const bool training,
        const unsigned long long seed
    ) {
        int row { static_cast<int>(blockIdx.x) };
        if (row >= N) return;

        const float* inp { input + row * H };
        const float* res { residual + row * H };
        float* out       { output + row * H };

        // Pass 1: compute mean
        float local_sum { 0.0f };
        for (int h { static_cast<int>(threadIdx.x) }; h < H; h += static_cast<int>(blockDim.x)) {
            local_sum += inp[h];
        }
        float mean { block_reduce_sum(local_sum) / static_cast<float>(H) };

        __shared__ float s_mean;
        if (threadIdx.x == 0) {
            s_mean = mean;
        }
        __syncthreads();
        mean = s_mean;

        // Pass 2: compute variance
        float local_var { 0.0f };
        for (int h { static_cast<int>(threadIdx.x) }; h < H; h += static_cast<int>(blockDim.x)) {
            float diff { inp[h] - mean };
            local_var += diff * diff;
        }
        float var { block_reduce_sum(local_var) / static_cast<float>(H) };

        __shared__ float s_inv_std;
        if (threadIdx.x == 0) {
            s_inv_std = rsqrtf(var + 1e-5f);
        }
        __syncthreads();
        float inv_std { s_inv_std };

        // Pass 3: normalize, scale, dropout, residual add
        curandStatePhilox4_32_10_t rng_state;
        if (training && dropout_p > 0.0f) {
            curand_init(
                seed,
                static_cast<unsigned long long>(row) * H + threadIdx.x,
                0, &rng_state
            );
        }

        float scale { 1.0f / (1.0f - dropout_p) };
        for (int h { static_cast<int>(threadIdx.x) }; h < H; h += static_cast<int>(blockDim.x)) {
            float normalized { (inp[h] - mean) * inv_std };
            float scaled     { normalized * gamma[h] + beta[h] };

            if (training && dropout_p > 0.0f) {
                float r { curand_uniform(&rng_state) };
                scaled = (r >= dropout_p) ? scaled * scale : 0.0f;
            }

            out[h] = scaled + res[h];
        }
    }


    torch::Tensor fused_norm_residual_dropout(
        torch::Tensor input,
        torch::Tensor residual,
        torch::Tensor gamma,
        torch::Tensor beta,
        float dropout_p,
        bool training
    ) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                    "input must be a contiguous CUDA tensor");
        TORCH_CHECK(residual.is_cuda() && residual.is_contiguous(),
                    "residual must be a contiguous CUDA tensor");

        int N { static_cast<int>(input.size(0)) };
        int H { static_cast<int>(input.size(1)) };

        auto output { torch::empty_like(input) };
        if (N == 0) return output;

        int threads { std::max(32, std::min(256, H)) };

        // Seed from high-resolution clock for dropout randomness
        unsigned long long rng_seed { training
            ? static_cast<unsigned long long>(
                std::chrono::steady_clock::now().time_since_epoch().count())
            : 0ULL
        };

        fused_layernorm_residual_dropout_kernel<<<N, threads>>>(
            input.data_ptr<float>(),
            residual.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            N, H,
            dropout_p, training, rng_seed
        );

        return output;
    }
}  // namespace qec
