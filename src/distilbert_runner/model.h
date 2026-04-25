//===- model.h - DistilBERT forward + SST-2 classifier head -----*- C++ -*-===//
//
// Forks bert.cpp's bert_build_graph() so we can:
//   1. drop the mean-pool tail (DistilBertForSequenceClassification reads
//      last_hidden_state[:, 0], not the pooled mean)
//   2. attach the SST-2 classifier head ourselves
//   3. own every ggml_mul_mat call site, which is what we will swap for
//      our XDNA2 NPU dispatch in Step 4-3-D.
//
// The classifier head tensors (pre_classifier.weight/bias and
// classifier.weight/bias) live in bert_ctx::ctx_data because we wrote
// them there from bert_cpp/convert.py with --with-classifier-head. We
// look them up by name, not by struct field, so vendor bert.h does not
// have to learn about the SST-2 head.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <vector>

#include "bert.h"   // bert_ctx, bert_tokens

namespace distilbert_runner {

/// Classifier head pulled out of the GGUF on first use. Lifetimes match
/// the bert_ctx that owns ctx_data.
struct ClassifierHead {
  struct ggml_tensor* pre_w = nullptr;   // [hidden_size, hidden_size]
  struct ggml_tensor* pre_b = nullptr;   // [hidden_size]
  struct ggml_tensor* cls_w = nullptr;   // [num_labels, hidden_size]
  struct ggml_tensor* cls_b = nullptr;   // [num_labels]
  int num_labels = 0;                    // pulled from cls_b->ne[0]
};

/// Look up the four SST-2 head tensors in bert_ctx->ctx_data. Throws
/// std::runtime_error if any of them is missing — this is the signal that
/// the GGUF was built without --with-classifier-head.
ClassifierHead loadClassifierHead(bert_ctx* ctx);

/// Run a single-sample forward pass: tokens -> logits.
///
/// `tokens` is the output of bert_tokenize (already [CLS]..[SEP]). On
/// return `logits_out` holds num_labels floats; softmax is the caller's
/// job (we expose raw logits to keep numerics auditable).
///
/// Internally builds the same encoder graph as bert.cpp without the
/// mean-pool tail, slices the [CLS] hidden state, then runs
/// pre_classifier -> tanh -> classifier on top.
void runForwardClassify(
    bert_ctx* ctx,
    const ClassifierHead& head,
    const bert_tokens& tokens,
    int n_threads,
    std::vector<float>& logits_out);

}  // namespace distilbert_runner
