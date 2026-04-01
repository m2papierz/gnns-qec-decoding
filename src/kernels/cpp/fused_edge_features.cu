/*
 * Fused edge feature computation for GNN message passing.
 *
 * Computes [h_src + h_dst | |h_src - h_dst| | edge_h]
 * in a single kernel launch, replacing 5 separate PyTorch ops
 * (2x gather + add + sub+abs + concat).
 *
 * Target: sm_80+ (Ampere, Ada Lovelace)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace qec {
    namespace {
        constexpr int kMaxThreads{256};
        constexpr int kMinThreads{32};

        // ------------------------------------------------------------------
        // Kernel: templated on vectorisation mode
        // ------------------------------------------------------------------

        template <bool UseVec4>
        __global__ void __launch_bounds__(kMaxThreads)
        symmetric_edge_features_kernel(
            const float* __restrict__ x,
            const int64_t* __restrict__ src_idx,
            const int64_t* __restrict__ dst_idx,
            const float* __restrict__ edge_h,
            float* __restrict__ output,
            const int num_edges,
            const int hidden_dim
        ) {
            const int edge_id{static_cast<int>(blockIdx.x)};
            if (edge_id >= num_edges) return;

            const int tid{static_cast<int>(threadIdx.x)};
            const int src{static_cast<int>(src_idx[edge_id])};
            const int dst{static_cast<int>(dst_idx[edge_id])};

            const float* __restrict__ src_ptr{x + src * hidden_dim};
            const float* __restrict__ dst_ptr{x + dst * hidden_dim};
            const float* __restrict__ eh_ptr{edge_h + edge_id * hidden_dim};
            float* __restrict__ out_ptr{output + edge_id * 3 * hidden_dim};

            if constexpr (UseVec4) {
                // Vectorised path: hidden_dim divisible by 4, row pointers
                // are 16-byte aligned (PyTorch base alignment + 4-aligned stride).
                const int stride{static_cast<int>(blockDim.x) * 4};
                for (int h{tid * 4}; h < hidden_dim; h += stride) {
                    const float4 hs{*reinterpret_cast<const float4*>(src_ptr + h)};
                    const float4 hd{*reinterpret_cast<const float4*>(dst_ptr + h)};
                    const float4 eh{*reinterpret_cast<const float4*>(eh_ptr + h)};

                    const float4 sum_part{
                        hs.x + hd.x, hs.y + hd.y,
                        hs.z + hd.z, hs.w + hd.w
                    };
                    const float4 abs_part{
                        fabsf(hs.x - hd.x), fabsf(hs.y - hd.y),
                        fabsf(hs.z - hd.z), fabsf(hs.w - hd.w)
                    };

                    *reinterpret_cast<float4*>(out_ptr + h)                  = sum_part;
                    *reinterpret_cast<float4*>(out_ptr + hidden_dim + h)     = abs_part;
                    *reinterpret_cast<float4*>(out_ptr + 2 * hidden_dim + h) = eh;
                }
            } else {
                // Scalar path: safe for any hidden_dim.
                const int stride{static_cast<int>(blockDim.x)};
                for (int h{tid}; h < hidden_dim; h += stride) {
                    const float s{src_ptr[h]};
                    const float d{dst_ptr[h]};
                    out_ptr[h]                  = s + d;
                    out_ptr[hidden_dim + h]     = fabsf(s - d);
                    out_ptr[2 * hidden_dim + h] = eh_ptr[h];
                }
            }
        }

        // ------------------------------------------------------------------
        // Thread count selection
        // ------------------------------------------------------------------

        inline int select_threads(int hidden_dim, bool use_vec4) noexcept {
            const int elements_per_thread{use_vec4 ? 4 : 1};
            const int needed{(hidden_dim + elements_per_thread - 1) / elements_per_thread};
            return std::max(kMinThreads, std::min(kMaxThreads, needed));
        }
    }  // anonymous namespace

    // ------------------------------------------------------------------
    // Host entry point
    // ------------------------------------------------------------------

    torch::Tensor fused_symmetric_edge_features(
        torch::Tensor x,
        torch::Tensor edge_index,
        torch::Tensor edge_h
    ) {
        TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
        TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
        TORCH_CHECK(edge_h.is_contiguous(), "edge_h must be contiguous");
        TORCH_CHECK(edge_index.is_contiguous(), "edge_index must be contiguous");
        TORCH_CHECK(
            edge_index.size(0) == 2,
            "edge_index must have shape (2, E), got (", edge_index.size(0), ", ...)"
        );

        const int num_edges{static_cast<int>(edge_index.size(1))};
        const int hidden_dim{static_cast<int>(x.size(1))};

        auto output{torch::empty({num_edges, 3 * hidden_dim}, x.options())};
        if (num_edges == 0) return output;

        const bool use_vec4{hidden_dim % 4 == 0};
        const int threads{select_threads(hidden_dim, use_vec4)};
        const auto stream{at::cuda::getCurrentCUDAStream()};

        if (use_vec4) {
            symmetric_edge_features_kernel<true><<<num_edges, threads, 0, stream>>>(
                x.data_ptr<float>(),
                edge_index[0].data_ptr<int64_t>(),
                edge_index[1].data_ptr<int64_t>(),
                edge_h.data_ptr<float>(),
                output.data_ptr<float>(),
                num_edges,
                hidden_dim
            );
        } else {
            symmetric_edge_features_kernel<false><<<num_edges, threads, 0, stream>>>(
                x.data_ptr<float>(),
                edge_index[0].data_ptr<int64_t>(),
                edge_index[1].data_ptr<int64_t>(),
                edge_h.data_ptr<float>(),
                output.data_ptr<float>(),
                num_edges,
                hidden_dim
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return output;
    }
}  // namespace qec
