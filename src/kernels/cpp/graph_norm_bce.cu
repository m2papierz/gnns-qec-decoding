/*
 * Per-graph normalized BCE with logits.
 *
 * Fuses: BCE computation + scatter_add per graph + normalization.
 * Two-pass: (1) compute per-edge BCE + scatter into graph accumulators,
 *           (2) normalize per graph and reduce to scalar.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "warp_reduce.cuh"

namespace qec {
    __global__ void graph_norm_bce_scatter_kernel(
        const float* __restrict__ logits,
        const float* __restrict__ target,
        const int64_t* __restrict__ edge_graph,
        float* __restrict__ graph_loss,
        float* __restrict__ graph_count,
        const float pos_weight,
        const int num_edges
    ) {
        int eid { static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        if (eid >= num_edges) return;

        float l { logits[eid] };
        float t { target[eid] };

        // Numerically stable BCE: max(l,0) - l*t + log(1 + exp(-|l|))
        float max_val { fmaxf(l, 0.0f) };
        float bce     { max_val - l * t + logf(1.0f + expf(-fabsf(l))) };

        if (pos_weight != 1.0f) {
            bce = bce * (1.0f + (pos_weight - 1.0f) * t);
        }

        int g { static_cast<int>(edge_graph[eid]) };
        atomicAdd(&graph_loss[g], bce);
        atomicAdd(&graph_count[g], 1.0f);
    }

    __global__ void graph_norm_reduce_kernel(
        const float* __restrict__ graph_loss,
        const float* __restrict__ graph_count,
        float* __restrict__ result,
        const int n_graphs
    ) {
        // Single block — reduce mean-of-means across graphs
        float local_sum { 0.0f };
        for (int g { static_cast<int>(threadIdx.x) }; g < n_graphs;
            g += static_cast<int>(blockDim.x)) {
            float count { fmaxf(graph_count[g], 1.0f) };
            local_sum += graph_loss[g] / count;
        }

        float total { block_reduce_sum(local_sum) };
        if (threadIdx.x == 0) {
            result[0] = total / static_cast<float>(n_graphs);
        }
    }

    torch::Tensor graph_normalized_bce(
        torch::Tensor logits,
        torch::Tensor target,
        torch::Tensor edge_graph,
        int n_graphs,
        c10::optional<torch::Tensor> pos_weight
    ) {
        TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(),
                    "logits must be a contiguous CUDA tensor");

        int num_edges { static_cast<int>(logits.size(0)) };
        float pw      { pos_weight.has_value() ? pos_weight->item<float>() : 1.0f };

        auto opts        { logits.options() };
        auto graph_loss  { torch::zeros({n_graphs}, opts) };
        auto graph_count { torch::zeros({n_graphs}, opts) };

        if (num_edges > 0) {
            constexpr int threads { 256 };
            int blocks { (num_edges + threads - 1) / threads };

            graph_norm_bce_scatter_kernel<<<blocks, threads>>>(
                logits.data_ptr<float>(),
                target.data_ptr<float>(),
                edge_graph.data_ptr<int64_t>(),
                graph_loss.data_ptr<float>(),
                graph_count.data_ptr<float>(),
                pw, num_edges
            );
        }

        auto result { torch::zeros({1}, opts) };
        graph_norm_reduce_kernel<<<1, 256>>>(
            graph_loss.data_ptr<float>(),
            graph_count.data_ptr<float>(),
            result.data_ptr<float>(),
            n_graphs
        );

        return result.squeeze(0);
    }
}  // namespace qec
