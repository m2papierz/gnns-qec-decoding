#include <torch/extension.h>

namespace qec {
    torch::Tensor fused_symmetric_edge_features(
        torch::Tensor x, torch::Tensor edge_index, torch::Tensor edge_h);

    torch::Tensor fused_norm_residual_dropout(
        torch::Tensor input, torch::Tensor residual,
        torch::Tensor gamma, torch::Tensor beta,
        float dropout_p, bool training);

    torch::Tensor graph_normalized_bce(
        torch::Tensor logits, torch::Tensor target,
        torch::Tensor edge_graph, int n_graphs,
        c10::optional<torch::Tensor> pos_weight);
}  // namespace qec

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_symmetric_edge_features",
          &qec::fused_symmetric_edge_features,
          "Fused symmetric edge features (CUDA)");
    m.def("fused_norm_residual_dropout",
          &qec::fused_norm_residual_dropout,
          "Fused LayerNorm + residual + dropout (CUDA)");
    m.def("graph_normalized_bce",
          &qec::graph_normalized_bce,
          "Per-graph normalized BCE (CUDA)");
}
