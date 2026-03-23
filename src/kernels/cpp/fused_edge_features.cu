/*
 * Fused edge feature computation for GNN message passing.
 *
 * Computes [h_src + h_dst | |h_src - h_dst| | edge_h]
 * in a single kernel launch, replacing 5 separate PyTorch ops
 * (2x gather + add + sub+abs + concat).
 *
 * Target: sm_89 (Ada Lovelace)
 * Uses: vectorized float4 loads when hidden_dim is 4-aligned,
 *       scalar fallback otherwise.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace qec {
    __global__ void fused_symmetric_edge_features_kernel(
        const float* __restrict__ x,
        const int64_t* __restrict__ src_idx,
        const int64_t* __restrict__ dst_idx,
        const float* __restrict__ edge_h,
        float* __restrict__ output,
        const int num_edges,
        const int hidden_dim,
        const bool use_vec4
    ) {
        int edge_id { static_cast<int>(blockIdx.x) };
        if (edge_id >= num_edges) return;

        int tid { static_cast<int>(threadIdx.x) };
        int src { static_cast<int>(src_idx[edge_id]) };
        int dst { static_cast<int>(dst_idx[edge_id]) };

        const float* src_ptr { x + src * hidden_dim };
        const float* dst_ptr { x + dst * hidden_dim };
        const float* eh_ptr  { edge_h + edge_id * hidden_dim };
        float* out_ptr       { output + edge_id * 3 * hidden_dim };

        if (use_vec4) {
            // Vectorized path: hidden_dim is divisible by 4 so all row
            // pointers are 16-byte aligned (assuming base tensor is aligned).
            for (int h { tid * 4 }; h < hidden_dim; h += static_cast<int>(blockDim.x) * 4) {
                float4 hs { *reinterpret_cast<const float4*>(src_ptr + h) };
                float4 hd { *reinterpret_cast<const float4*>(dst_ptr + h) };
                float4 eh { *reinterpret_cast<const float4*>(eh_ptr + h) };

                float4 sum_part {
                    hs.x + hd.x, hs.y + hd.y,
                    hs.z + hd.z, hs.w + hd.w
                };
                float4 abs_part {
                    fabsf(hs.x - hd.x), fabsf(hs.y - hd.y),
                    fabsf(hs.z - hd.z), fabsf(hs.w - hd.w)
                };

                *reinterpret_cast<float4*>(out_ptr + h) = sum_part;
                *reinterpret_cast<float4*>(out_ptr + hidden_dim + h) = abs_part;
                *reinterpret_cast<float4*>(out_ptr + 2 * hidden_dim + h) = eh;
            }
        } else {
            // Scalar path: safe for any hidden_dim
            for (int h { tid }; h < hidden_dim; h += static_cast<int>(blockDim.x)) {
                float s { src_ptr[h] };
                float d { dst_ptr[h] };
                out_ptr[h]                  = s + d;
                out_ptr[hidden_dim + h]     = fabsf(s - d);
                out_ptr[2 * hidden_dim + h] = eh_ptr[h];
            }
        }
    }


    torch::Tensor fused_symmetric_edge_features(
        torch::Tensor x,
        torch::Tensor edge_index,
        torch::Tensor edge_h
    ) {
        TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
        TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
        TORCH_CHECK(edge_h.is_contiguous(), "edge_h must be contiguous");

        int num_edges  { static_cast<int>(edge_index.size(1)) };
        int hidden_dim { static_cast<int>(x.size(1)) };

        auto output { torch::empty({num_edges, 3 * hidden_dim}, x.options()) };
        if (num_edges == 0) return output;

        // float4 requires 16-byte alignment: rows must start at 4-float boundaries
        bool use_vec4 { hidden_dim % 4 == 0 };
        int threads { use_vec4
            ? std::min(256, (hidden_dim + 3) / 4)
            : std::min(256, hidden_dim)
        };
        threads = std::max(threads, 32);

        fused_symmetric_edge_features_kernel<<<num_edges, threads>>>(
            x.data_ptr<float>(),
            edge_index[0].data_ptr<int64_t>(),
            edge_index[1].data_ptr<int64_t>(),
            edge_h.data_ptr<float>(),
            output.data_ptr<float>(),
            num_edges,
            hidden_dim,
            use_vec4
        );

        return output;
    }
}  // namespace qec
