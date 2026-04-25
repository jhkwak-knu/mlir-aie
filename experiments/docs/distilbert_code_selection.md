# External C++ DistilBERT Inference: Selection Record

Notion task #22 §Step 4-1.

We need a C++ DistilBERT (or BERT-base) inference codebase to:
- Load the Hugging Face `distilbert-base-uncased` checkpoint
- Run SST-2 classification on 100 samples within ±2%p of the PyTorch
  reference (Step 4-2 baseline gate)
- Replace every GEMM call with our XDNA2 NPU dispatch in Step 4-3
- Provide enough non-GEMM coverage (LayerNorm, GeLU, softmax, residual,
  embedding lookup) so that we don't have to rewrite the model from
  scratch.

## Candidates surveyed

| Repo | Last commit | License | DistilBERT | GEMM separation | Tokenizer | SST-2 head | Stars / activity | Verdict |
|---|---|---|---|---|---|---|---|---|
| **iamlemec/bert.cpp** | 2024-02-19 (stale) | MIT | implicit (BERT path; weight loader needs DistilBERT branch) | excellent — every layer issues 8 explicit `ggml_mul_mat` calls (Q/K/V proj, K^T·Q, attn·V, O proj, FFN1, FFN2) → exact 1:1 mapping to our 48 dispatches | built-in (BERT WordPiece, GGUF vocab) | none (mean-pool + L2 norm only; classifier must be appended) | 57 stars, semi-defunct README | **chosen** |
| skeskinen/bert.cpp | 2024-02-23 (stale) | MIT | n/a | identical to iamlemec (parent) | built-in, CJK weak | none | 500 stars, deprecated, README points to iamlemec / llama.cpp | superseded |
| EeyoreLee/bert.cpp | 2024-10-21 | MIT | n/a | parent ggml structure | tokenizers-cpp (Rust binding) | **yes** (`BertForSequenceClassification` implemented) | 3 stars | useful classifier reference but tiny audience and Rust dep |
| ggml-org/llama.cpp | 2026-04-25 (active) | MIT | not registered (`llama-arch.cpp` lists `LLM_ARCH_BERT / MODERN_BERT / NOMIC_BERT`; no `distilbert`; `convert_hf_to_gguf.py` covers `BertModel / RobertaModel`) | excellent overall, but the BERT graph is buried in `llama-model.cpp`; carving out a custom dispatch path is much more invasive | built-in | none for BERT path (classification head not in the BERT inference path) | 106k stars | rejected: unsupported architecture and high integration cost |
| microsoft/onnxruntime | 2026-04-25 (active) | MIT | full (HF publishes a fine-tuned DistilBERT-SST2 ONNX) | Custom EP dispatch is **subgraph-level**, not op-level; routing only `MatMul` ops claim-by-claim incurs an EP boundary copy on each, plus a multi-GB build dependency | external (separate BertTokenizer C++ to vendor) | yes (`ORTModelForSequenceClassification`) | 20k stars | strong fallback, but the EP boundary cost and build weight outweigh the zero-weight-conversion appeal |
| huggingface/text-embeddings-inference | active | Apache 2.0 | DistilBert MaskedLM listed | Candle (Rust) backend — not C++ | built-in | embedding-only (SPLADE pool) | active | rejected: language and head |
| rbitr/ferrite | 2023-11-17 | MIT | "DistilBERT only" | Fortran | partial | unknown | 17 stars, stale | rejected: language |

## Decision

**Vendor `iamlemec/bert.cpp` at commit `c791b305e92a37514a0f6add0804b0e08150e59d` (2024-02-19) into `external/bert.cpp/`.**

Submodule `ggml` is materialized at commit `6b14d738d9100c50c199a3b1aaa960f633904476` and the upstream `.git`, `.gitmodules`, `ggml/.git` directories are stripped so the tree is a flat MIT-licensed vendor copy.

### Why iamlemec/bert.cpp

The project's brief is to study **NPU GEMM dispatch behavior** under four configuration setters, not to ship a BERT runtime. The dominant integration cost we care about is "where does GEMM live, and how cheaply can I redirect it to XDNA2?". `iamlemec/bert.cpp` exposes every transformer matmul as a top-level `ggml_mul_mat` call inside `bert_build_graph()`, which lines up 1:1 with the 48 GEMM dispatches we extracted in Step 1. Replacing them is a single targeted change — see the dispatch hook plan below.

### Hook plan for Step 4-3

Two equivalent options; we will start with the second because it keeps the change footprint inside our own files:

1. **ggml custom backend (preferred long-term)** — register an XDNA2 backend via `ggml_backend_buffer_type_t` plus a `compute_forward` callback that handles `op->op == GGML_OP_MUL_MAT` and falls back to the CPU backend for everything else. `ggml_backend_sched_t` then partitions the graph automatically.
2. **Direct call-site replacement (Step 4-3 starting point)** — wrap `ggml_mul_mat` in a thin `npu_mul_mat()` shim defined in `src/distilbert_runner/` that calls our existing `Dispatcher::dispatch(LayerKernelEntry, A, B, C)` and emits the result tensor; pass the shim into the model code in place of `ggml_mul_mat`. Smaller blast radius, easier to verify.

### What we still owe

These are the items we knowingly bring on in exchange for the clean GEMM separation; they are explicit Step 4-2 / 4-3 deliverables, not scope creep:

1. **DistilBERT weight loader.** `bert_cpp/convert.py` walks BERT-style HuggingFace state dicts. DistilBERT is structurally a 6-layer BERT without `token_type_embeddings` and without the pooler — so the converter needs a small `model_type == "distilbert"` branch (drop those tensors, remap layer count).
2. **SST-2 classification head.** Append `pre_classifier (Linear 768→768) + tanh + classifier (Linear 768→2)` after the existing CLS-token slice. EeyoreLee/bert.cpp shows how to wire this into the ggml graph and we will mirror that. **Implication for our shape extraction:** the head adds two extra GEMM shapes (128×768×768 — already present — and 128×768×2 — new). The 128×768×2 shape will likely fail XDNA2 C5 (N%16==0); we will pad N from 2 → 16 and slice the first 2 logits host-side, exactly the pattern MLP fc3 used (10 → 16). This needs an extract_shapes.py update before Step 4-3 measurements close, and we will re-run Steps 1–3 to add the new kernel binary.
3. **Tokenizer plumbing.** bert.cpp's WordPiece is sufficient; SST-2 inputs are short English sentences.
4. **Stale upstream.** `iamlemec/bert.cpp` is self-described as "semi-defunct" and recommends `llama.cpp`. Because we strip `.git` and vendor a frozen snapshot we are not exposed to upstream rot, but we should not expect to merge fixes back.

### Risks accepted

- ggml itself ships a lot of backend stubs we will never compile (vulkan, sycl, kompute, cuda, metal). Step 4-3 CMakeLists will exclude them so the binary stays small.
- `--backend npu` cycle still needs the existing XRT dispatcher; we do not introduce a new dispatcher class — the bert.cpp graph just calls our existing `Dispatcher::dispatch`.
- If Step 4-2 baseline accuracy (PyTorch ±2%p) does not converge, the most likely cause is the classifier-head wiring, not the BERT graph — bert.cpp's BERT path has well-known parity with HF.
