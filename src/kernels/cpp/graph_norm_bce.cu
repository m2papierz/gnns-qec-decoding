/*
 * Per-graph normalized BCE with logits.
 *
 * Fuses: BCE computation + scatter_add per graph + normalization.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace qec {
    namespace {
        constexpr int kBlockSize{256};
        constexpr unsigned int kFullMask{0xFFFFFFFF};

        // ------------------------------------------------------------------
        // BCE per element (numerically stable)
        // ------------------------------------------------------------------

        __device__ __forceinline__ float stable_bce(
            float logit, float target, float pos_weight
        ) {
            const float max_val{fmaxf(logit, 0.0f)};
            float bce{max_val - logit * target + logf(1.0f + expf(-fabsf(logit)))};
            if (pos_weight != 1.0f) {
                bce *= 1.0f + (pos_weight - 1.0f) * target;
            }
            return bce;
        }

        // ------------------------------------------------------------------
        // Scatter kernel: warp-cooperative segmented reduction
        //
        // Exploits the fact that edge_graph is sorted (monotonic) in PyG
        // batches.  Within each warp, contiguous edges belonging to the
        // same graph are accumulated via an inclusive segmented prefix sum,
        // then only the last lane per segment does a single atomicAdd.
        // ------------------------------------------------------------------

        __global__ void __launch_bounds__(kBlockSize)
        graph_norm_bce_scatter_kernel(
            const float* __restrict__ logits,
            const float* __restrict__ target,
            const int64_t* __restrict__ edge_graph,
            float* __restrict__ graph_loss,
            const float pos_weight,
            const int num_edges
        ) {
            const int eid{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
            const int lane{threadIdx.x & 31};

            // Out-of-bounds threads: participate in shuffles but do not write.
            // Sentinel gid=-1 ensures invalid threads never merge with a valid
            // segment during the prefix sum or segment-end detection.
            const bool valid{eid < num_edges};

            float bce{0.0f};
            int gid{-1};
            if (valid) {
                bce = stable_bce(logits[eid], target[eid], pos_weight);
                gid = static_cast<int>(edge_graph[eid]);
            }

            // --- Warp-cooperative segmented inclusive prefix sum ---
            // After this loop, each thread holds the partial sum of all
            // preceding threads in its segment (same graph_id) within the warp.
            float accum{bce};

            #pragma unroll
            for (int offset{1}; offset < 32; offset <<= 1) {
                const float val{__shfl_up_sync(kFullMask, accum, offset)};
                const int src_gid{__shfl_up_sync(kFullMask, gid, offset)};
                if (lane >= offset && src_gid == gid) {
                    accum += val;
                }
            }

            // The last thread in each contiguous segment writes the
            // accumulated sum.  A thread is a segment end if:
            //   (a) it is the last active lane in the warp, or
            //   (b) the next lane has a different graph_id.
            const int next_gid{__shfl_down_sync(kFullMask, gid, 1)};
            const bool is_segment_end{
                (lane == 31) || !valid || (gid != next_gid)
            };

            if (valid && is_segment_end) {
                atomicAdd(&graph_loss[gid], accum);
            }
        }

        // ------------------------------------------------------------------
        // Reduce kernel: mean-of-means across graphs (single block)
        // ------------------------------------------------------------------

        __device__ __forceinline__ float warp_reduce_sum(float val) {
            #pragma unroll
            for (int offset{16}; offset > 0; offset >>= 1) {
                val += __shfl_xor_sync(kFullMask, val, offset);
            }
            return val;
        }

        __device__ __forceinline__ float block_reduce_sum(float val) {
            __shared__ float shared[32];

            const int lane{threadIdx.x & 31};
            const int warp_id{static_cast<int>(threadIdx.x / 32)};
            const int num_warps{static_cast<int>((blockDim.x + 31) / 32)};

            val = warp_reduce_sum(val);
            if (lane == 0) shared[warp_id] = val;
            __syncthreads();

            val = (lane < num_warps) ? shared[lane] : 0.0f;
            if (warp_id == 0) val = warp_reduce_sum(val);
            return val;
        }

        __global__ void graph_norm_reduce_kernel(
            const float* __restrict__ graph_loss,
            const float* __restrict__ graph_count,
            float* __restrict__ result,
            const int n_graphs
        ) {
            float local_sum{0.0f};
            for (int g{static_cast<int>(threadIdx.x)}; g < n_graphs;
                g += static_cast<int>(blockDim.x))
            {
                const float count{fmaxf(graph_count[g], 1.0f)};
                local_sum += graph_loss[g] / count;
            }

            const float total{block_reduce_sum(local_sum)};
            if (threadIdx.x == 0) {
                result[0] = total / static_cast<float>(n_graphs);
            }
        }
    }  // anonymous namespace

    // ------------------------------------------------------------------
    // Host entry point
    // ------------------------------------------------------------------

    torch::Tensor graph_normalized_bce(
        torch::Tensor logits,
        torch::Tensor target,
        torch::Tensor edge_graph,
        int n_graphs,
        c10::optional<torch::Tensor> pos_weight
    ) {
        TORCH_CHECK(
            logits.is_cuda() && logits.is_contiguous(),
            "logits must be a contiguous CUDA tensor"
        );
        TORCH_CHECK(
            target.is_cuda() && target.is_contiguous(),
            "target must be a contiguous CUDA tensor"
        );
        TORCH_CHECK(
            edge_graph.is_cuda() && edge_graph.is_contiguous(),
            "edge_graph must be a contiguous CUDA tensor"
        );

        const int num_edges{static_cast<int>(logits.size(0))};
        const float pw{pos_weight.has_value() ? pos_weight->item<float>() : 1.0f};

        const auto opts{logits.options()};
        auto graph_loss{torch::zeros({n_graphs}, opts)};

        // Precompute per-graph edge counts (replaces all count atomics).
        auto graph_count{torch::bincount(
            edge_graph, /*weights=*/{}, /*minlength=*/n_graphs
        ).to(opts)};

        const auto stream{at::cuda::getCurrentCUDAStream()};

        if (num_edges > 0) {
            const int blocks{(num_edges + kBlockSize - 1) / kBlockSize};

            graph_norm_bce_scatter_kernel<<<blocks, kBlockSize, 0, stream>>>(
                logits.data_ptr<float>(),
                target.data_ptr<float>(),
                edge_graph.data_ptr<int64_t>(),
                graph_loss.data_ptr<float>(),
                pw,
                num_edges
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        auto result{torch::zeros({1}, opts)};
        graph_norm_reduce_kernel<<<1, kBlockSize, 0, stream>>>(
            graph_loss.data_ptr<float>(),
            graph_count.data_ptr<float>(),
            result.data_ptr<float>(),
            n_graphs
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return result.squeeze(0);
    }
}  // namespace qec
