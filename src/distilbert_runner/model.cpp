//===- model.cpp - DistilBERT encoder + SST-2 classifier head graph --------===//
//
// The encoder section is a fork of bert.cpp's bert_build_graph() (commit
// c791b305, src/bert.cpp:737-915). We keep the embedding lookups, layer
// norms, attention block, and FFN block byte-for-byte identical so that
// numerical parity with the upstream embedding model is preserved. Only
// the tail differs:
//
//   bert.cpp tail        : mean-pool -> optional L2 normalize -> output
//   distilbert tail      : take last_hidden_state[:, 0]
//                            -> Linear pre_classifier (768 x 768)
//                            -> tanh
//                            -> Linear classifier      (num_labels x 768)
//                            -> output (raw logits)
//
// All matmuls in the encoder section come from bert_model::layers (Q/K/V/O,
// FFN1, FFN2). The classifier matmuls come from ClassifierHead. Both are
// reachable to the future NPU dispatcher because they go through
// ggml_mul_mat in the same single graph.
//
//===----------------------------------------------------------------------===//

#include "model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace distilbert_runner {

namespace {

// bert.cpp uses 8 KiB; we exercise more graph nodes (classifier head),
// so keep the same headroom.
constexpr size_t kMaxNodes = 8192;

ggml_tensor* getTensorOrThrow(ggml_context* ctx_data, const char* name) {
  ggml_tensor* t = ggml_get_tensor(ctx_data, name);
  if (!t) {
    throw std::runtime_error(
        std::string("missing tensor '") + name + "' in GGUF "
        "(did you forget --with-classifier-head when running convert.py?)");
  }
  return t;
}

}  // namespace

ClassifierHead loadClassifierHead(bert_ctx* ctx) {
  ClassifierHead h;
  h.pre_w = getTensorOrThrow(ctx->ctx_data, "pre_classifier.weight");
  h.pre_b = getTensorOrThrow(ctx->ctx_data, "pre_classifier.bias");
  h.cls_w = getTensorOrThrow(ctx->ctx_data, "classifier.weight");
  h.cls_b = getTensorOrThrow(ctx->ctx_data, "classifier.bias");
  h.num_labels = static_cast<int>(h.cls_b->ne[0]);
  if (h.num_labels <= 0) {
    throw std::runtime_error("classifier.bias has zero labels");
  }
  return h;
}

void runForwardClassify(bert_ctx* ctx,
                        const ClassifierHead& head,
                        const bert_tokens& tokens,
                        int n_threads,
                        std::vector<float>& logits_out) {
  // ----- Mirror of bert_build_graph() up to last_hidden_state ---------
  const bert_vocab& vocab = ctx->vocab;
  const bert_token pad_id = vocab.pad_id;

  const bert_model& model = ctx->model;
  const bert_hparams& hparams = model.hparams;
  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_max_tokens_hp = hparams.n_max_tokens;
  const int n_head = hparams.n_head;
  const float layer_norm_eps = hparams.layer_norm_eps;
  const int d_head = n_embd / n_head;

  const int cur_max_len = static_cast<int>(tokens.size());
  if (cur_max_len > n_max_tokens_hp) {
    throw std::runtime_error("token count exceeds n_max_tokens");
  }
  const int n_batch_size = 1;

  // Reset bert.cpp's compute alloc so every invocation starts clean.
  ggml_allocr_reset(ctx->compute_alloc);

  ggml_init_params params = {
      /*.mem_size   =*/ ctx->buf_compute_meta.size(),
      /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
      /*.no_alloc   =*/ true,
  };
  ggml_context* ctx0 = ggml_init(params);
  ggml_cgraph* gf = ggml_new_graph_custom(ctx0, kMaxNodes, false);

  // Inputs.
  ggml_tensor* token_layer  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur_max_len * n_batch_size);
  ggml_tensor* token_types  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur_max_len * n_batch_size);
  ggml_tensor* pad_mask     = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, cur_max_len, 1, n_batch_size);
  ggml_tensor* positions    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur_max_len * n_batch_size);
  ggml_tensor* minus_one    = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
  ggml_allocr_alloc(ctx->compute_alloc, token_layer);
  ggml_allocr_alloc(ctx->compute_alloc, token_types);
  ggml_allocr_alloc(ctx->compute_alloc, pad_mask);
  ggml_allocr_alloc(ctx->compute_alloc, positions);
  ggml_allocr_alloc(ctx->compute_alloc, minus_one);

  if (!ggml_allocr_is_measure(ctx->compute_alloc)) {
    std::vector<int32_t> token_layer_data(cur_max_len * n_batch_size);
    std::vector<int32_t> token_types_data(cur_max_len * n_batch_size, 0);
    std::vector<float>   pad_mask_data(cur_max_len * n_batch_size);
    std::vector<int32_t> pos_data(cur_max_len * n_batch_size);
    const float m1 = -1.0f;

    const int cur_len = static_cast<int>(tokens.size());
    for (int i = 0; i < cur_max_len; ++i) {
      if (i < cur_len) {
        token_layer_data[i] = tokens[i];
        pad_mask_data[i] = 1.0f;
      } else {
        token_layer_data[i] = pad_id;
        pad_mask_data[i] = 0.0f;
      }
      pos_data[i] = i;
    }

    ggml_backend_tensor_set(token_layer, token_layer_data.data(), 0, ggml_nbytes(token_layer));
    ggml_backend_tensor_set(token_types, token_types_data.data(), 0, ggml_nbytes(token_types));
    ggml_backend_tensor_set(pad_mask,    pad_mask_data.data(),    0, ggml_nbytes(pad_mask));
    ggml_backend_tensor_set(positions,   pos_data.data(),         0, ggml_nbytes(positions));
    ggml_backend_tensor_set(minus_one,   &m1,                      0, sizeof(m1));
  }

  // Outer-product the padding mask.
  ggml_tensor* attn_mask = ggml_mul_mat(ctx0, pad_mask, pad_mask);  // [L, L, 1, B]
  attn_mask = ggml_add(ctx0, attn_mask, minus_one);
  attn_mask = ggml_scale_inplace(ctx0, attn_mask, 100000.0f);

  // Token embeddings + token-type (zero rows in DistilBERT GGUF) + positions.
  ggml_tensor* inpL = ggml_get_rows(ctx0, model.word_embeddings, token_layer);          // [E, L*B]
  inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.token_type_embeddings, token_types), inpL);
  inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.position_embeddings, positions), inpL);
  inpL = ggml_reshape_3d(ctx0, inpL, n_embd, cur_max_len, n_batch_size);                 // [E, L, B]

  // Embedding LayerNorm.
  inpL = ggml_norm_inplace(ctx0, inpL, layer_norm_eps);
  inpL = ggml_add(ctx0, ggml_mul(ctx0, inpL, model.ln_e_w), model.ln_e_b);

  // Encoder layers (identical to bert_build_graph()).
  for (int il = 0; il < n_layer; ++il) {
    ggml_tensor* cur = inpL;

    // Self-attention.
    {
      ggml_tensor* Q = cur;
      Q = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].q_w, Q), model.layers[il].q_b);
      Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, cur_max_len, n_batch_size);
      Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

      ggml_tensor* K = cur;
      K = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].k_w, K), model.layers[il].k_b);
      K = ggml_reshape_4d(ctx0, K, d_head, n_head, cur_max_len, n_batch_size);
      K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));

      ggml_tensor* V = cur;
      V = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].v_w, V), model.layers[il].v_b);
      V = ggml_reshape_4d(ctx0, V, d_head, n_head, cur_max_len, n_batch_size);
      V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));

      ggml_tensor* KQ = ggml_mul_mat(ctx0, K, Q);
      KQ = ggml_scale_inplace(ctx0, KQ, 1.0f / std::sqrt(static_cast<float>(d_head)));
      KQ = ggml_add(ctx0, KQ, attn_mask);
      KQ = ggml_soft_max(ctx0, KQ);

      ggml_tensor* KQV = ggml_mul_mat(ctx0, V, KQ);
      KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3));

      cur = ggml_reshape_3d(ctx0, KQV, n_embd, cur_max_len, n_batch_size);
    }

    // O projection + residual + LayerNorm.
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].o_w, cur), model.layers[il].o_b);
    cur = ggml_add(ctx0, cur, inpL);
    cur = ggml_norm_inplace(ctx0, cur, layer_norm_eps);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_att_w), model.layers[il].ln_att_b);
    ggml_tensor* att_output = cur;

    // FFN.
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur), model.layers[il].ff_i_b);
    cur = ggml_gelu(ctx0, cur);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur), model.layers[il].ff_o_b);
    cur = ggml_add(ctx0, att_output, cur);

    // Output LayerNorm.
    cur = ggml_norm_inplace(ctx0, cur, layer_norm_eps);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_out_w), model.layers[il].ln_out_b);
    inpL = cur;
  }
  // inpL : [E, L, B]  (last_hidden_state)

  // ----- DistilBERT classification head ------------------------------
  // [CLS] = inpL[:, 0, :]  -> shape [E, B] for our B=1 case.
  ggml_tensor* cls = ggml_view_2d(
      ctx0, inpL,
      n_embd, n_batch_size,
      inpL->nb[2],
      /*offset=*/0);
  cls = ggml_cont(ctx0, cls);

  // pre_classifier(cls) = pre_w * cls + pre_b   ; ggml_mul_mat takes (W, x).
  ggml_tensor* pooled =
      ggml_add(ctx0, ggml_mul_mat(ctx0, head.pre_w, cls), head.pre_b);
  pooled = ggml_tanh(ctx0, pooled);

  // classifier(pooled) = cls_w * pooled + cls_b
  ggml_tensor* logits =
      ggml_add(ctx0, ggml_mul_mat(ctx0, head.cls_w, pooled), head.cls_b);
  // logits : [num_labels, B]

  ggml_build_forward_expand(gf, logits);

  // Allocate compute buffer for the whole graph and run.
  ggml_allocr_alloc_graph(ctx->compute_alloc, gf);
  ggml_backend_graph_compute(ctx->backend, gf);

  // Read out.
  logits_out.assign(static_cast<size_t>(head.num_labels) * n_batch_size, 0.0f);
  ggml_backend_tensor_get(logits, logits_out.data(), 0, ggml_nbytes(logits));

  ggml_free(ctx0);
  (void)n_threads;  // CPU thread count is wired through bert.cpp's backend.
}

}  // namespace distilbert_runner
