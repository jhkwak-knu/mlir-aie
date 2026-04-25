import sys
import torch

from gguf import GGUFWriter, GGMLQuantizationType
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

KEY_PAD_ID = 'tokenizer.ggml.padding_token_id'
KEY_UNK_ID = 'tokenizer.ggml.unknown_token_id'
KEY_BOS_ID = 'tokenizer.ggml.bos_token_id'
KEY_EOS_ID = 'tokenizer.ggml.eos_token_id'
KEY_WORD_PREFIX = 'tokenizer.ggml.word_prefix'
KEY_SUBWORD_PREFIX = 'tokenizer.ggml.subword_prefix'

# DistilBERT-only: state_dict tensor names use a different vocabulary from
# BERT. src/bert.cpp expects BERT-style keys, so when the source model is a
# DistilBertModel we rename each tensor on the way into the GGUF file.
#
# In addition, DistilBERT has no `token_type_embeddings`, but src/bert.cpp
# always indexes that tensor (with token_type_id = 0 hard-coded at line
# 816 of src/bert.cpp). We synthesize a zero tensor so the lookup returns
# a zero vector, leaving the embedding sum unchanged — which matches what
# DistilBERT's forward actually does.

_DISTILBERT_PER_LAYER_RENAME = {
    "attention.q_lin.weight":      "attention.self.query.weight",
    "attention.q_lin.bias":        "attention.self.query.bias",
    "attention.k_lin.weight":      "attention.self.key.weight",
    "attention.k_lin.bias":        "attention.self.key.bias",
    "attention.v_lin.weight":      "attention.self.value.weight",
    "attention.v_lin.bias":        "attention.self.value.bias",
    "attention.out_lin.weight":    "attention.output.dense.weight",
    "attention.out_lin.bias":      "attention.output.dense.bias",
    "sa_layer_norm.weight":        "attention.output.LayerNorm.weight",
    "sa_layer_norm.bias":          "attention.output.LayerNorm.bias",
    "ffn.lin1.weight":             "intermediate.dense.weight",
    "ffn.lin1.bias":               "intermediate.dense.bias",
    "ffn.lin2.weight":             "output.dense.weight",
    "ffn.lin2.bias":               "output.dense.bias",
    "output_layer_norm.weight":    "output.LayerNorm.weight",
    "output_layer_norm.bias":      "output.LayerNorm.bias",
}

_DISTILBERT_EMBEDDING_PASSTHROUGH = {
    "embeddings.word_embeddings.weight",
    "embeddings.position_embeddings.weight",
    "embeddings.LayerNorm.weight",
    "embeddings.LayerNorm.bias",
}


_CLASSIFIER_HEAD_PASSTHROUGH = {
    "pre_classifier.weight",
    "pre_classifier.bias",
    "classifier.weight",
    "classifier.bias",
}


def distilbert_to_bert_name(name):
    """Translate a DistilBert state_dict key to the BERT-style key used by
    src/bert.cpp. Returns None for keys we deliberately drop.

    Accepts both DistilBertModel-style keys (no prefix) and
    DistilBertForSequenceClassification-style keys (`distilbert.` prefix
    on the body, plus a `pre_classifier` / `classifier` head). Classifier
    head tensors pass through unchanged so that distilbert_runner can
    `gguf_get_tensor()` them directly while leaving src/bert.cpp's
    BERT-shaped lookups undisturbed.
    """
    if name in _CLASSIFIER_HEAD_PASSTHROUGH:
        return name
    # Strip the `distilbert.` prefix that DistilBertForSequenceClassification
    # adds to every body tensor; vanilla DistilBertModel state_dicts don't
    # carry it so the no-prefix path still works.
    if name.startswith("distilbert."):
        name = name[len("distilbert."):]
    if name in _DISTILBERT_EMBEDDING_PASSTHROUGH:
        return name
    if name.startswith("transformer.layer."):
        rest = name[len("transformer.layer."):]
        sep = rest.find(".")
        if sep == -1:
            return None
        layer_idx, sub = rest[:sep], rest[sep + 1:]
        mapped = _DISTILBERT_PER_LAYER_RENAME.get(sub)
        if mapped is None:
            return None
        return f"encoder.layer.{layer_idx}.{mapped}"
    return None


def _hparams_from_config(config):
    """Resolve BERT-style hparam values from either a BertConfig or a
    DistilBertConfig. src/bert.cpp consumes the BERT names, so the
    DistilBert path translates dim / hidden_dim / n_heads / n_layers."""
    model_type = getattr(config, "model_type", "")
    if model_type == "distilbert":
        return {
            "vocab_size":              config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "hidden_size":             config.dim,
            "intermediate_size":       config.hidden_dim,
            "num_attention_heads":     config.n_heads,
            "num_hidden_layers":       config.n_layers,
            # DistilBertConfig has no layer_norm_eps; BERT default is 1e-12.
            "layer_norm_eps":          getattr(config, "layer_norm_eps", 1e-12),
        }
    return {
        "vocab_size":              config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size":             config.hidden_size,
        "intermediate_size":       config.intermediate_size,
        "num_attention_heads":     config.num_attention_heads,
        "num_hidden_layers":       config.num_hidden_layers,
        "layer_norm_eps":          config.layer_norm_eps,
    }


def convert_hf(repo_id, output_path, float_type='f16', with_classifier_head=False):
    # convert to ggml quantization type
    if float_type not in ['f16', 'f32']:
        print(f'Float type must be f16 or f32, got: {float_type}')
        sys.exit(1)
    else:
        qtype = GGMLQuantizationType[float_type.upper()]
        dtype0 = {'f16': torch.float16, 'f32': torch.float32}[float_type]

    # load tokenizer and model
    vocab = AutoTokenizer.from_pretrained(repo_id)
    if with_classifier_head:
        # SST-2 fine-tuned models live as DistilBertForSequenceClassification:
        # body + pre_classifier (Linear 768->768) + classifier (Linear 768->n).
        # AutoModel would silently strip the head, so we ask for the full
        # classifier here.
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    else:
        model = AutoModel.from_pretrained(repo_id)
    config = model.config
    model_type = getattr(config, "model_type", "")
    is_distilbert = (model_type == "distilbert")
    hparams = _hparams_from_config(config)

    # get token list
    token_list = vocab.convert_ids_to_tokens(range(vocab.vocab_size))

    # detect subword scheme
    if getattr(vocab.backend_tokenizer.pre_tokenizer, 'replacement', None) == '▁':
        word_prefix = '▁'
        subword_prefix = ''
    else:
        word_prefix = ''
        subword_prefix = '##'

    # print model
    print(f'PARAMS (model_type={model_type or "?"})')
    for k in (
        'vocab_size', 'max_position_embeddings', 'hidden_size',
        'intermediate_size', 'num_attention_heads', 'num_hidden_layers',
        'layer_norm_eps',
    ):
        print(f'{k:<24s} = {hparams[k]}')
    print()

    # print vocab
    vocab_keys = [
        'vocab_size', 'pad_token_id', 'unk_token_id', 'cls_token_id', 'sep_token_id'
    ]
    print('VOCAB')
    for k in vocab_keys:
        v = getattr(vocab, k)
        print(f'{k:24s} = {v}')
    print(f'{"word_prefix":24s} = {word_prefix}')
    print(f'{"subword_prefix":24s} = {subword_prefix}')
    print()

    # start to write GGUF file
    gguf_writer = GGUFWriter(output_path, 'bert')

    # write metadata
    gguf_writer.add_name('BERT')
    gguf_writer.add_description(
        'GGML BERT model' + (' (converted from DistilBERT)' if is_distilbert else '')
    )
    gguf_writer.add_file_type(qtype)

    # write model params
    gguf_writer.add_uint32('vocab_size', hparams['vocab_size'])
    gguf_writer.add_uint32('max_position_embedding', hparams['max_position_embeddings'])
    gguf_writer.add_uint32('hidden_size', hparams['hidden_size'])
    gguf_writer.add_uint32('intermediate_size', hparams['intermediate_size'])
    gguf_writer.add_uint32('num_attention_heads', hparams['num_attention_heads'])
    gguf_writer.add_uint32('num_hidden_layers', hparams['num_hidden_layers'])
    gguf_writer.add_float32('layer_norm_eps', hparams['layer_norm_eps'])

    # write vocab params
    gguf_writer.add_int32(KEY_PAD_ID, vocab.pad_token_id)
    gguf_writer.add_int32(KEY_UNK_ID, vocab.unk_token_id)
    gguf_writer.add_int32(KEY_BOS_ID, vocab.cls_token_id)
    gguf_writer.add_int32(KEY_EOS_ID, vocab.sep_token_id)
    gguf_writer.add_string(KEY_WORD_PREFIX, word_prefix)
    gguf_writer.add_string(KEY_SUBWORD_PREFIX, subword_prefix)
    gguf_writer.add_token_list(token_list)

    # write tensors
    print('TENSORS')
    n_dropped = 0
    for name, data in model.state_dict().items():
        if is_distilbert:
            out_name = distilbert_to_bert_name(name)
            if out_name is None:
                n_dropped += 1
                print(f'  drop  {name}')
                continue
        else:
            out_name = name

        # get correct dtype
        if 'LayerNorm' in out_name or 'bias' in out_name:
            dtype = torch.float32
        else:
            dtype = dtype0

        # print info
        shape_str = str(list(data.shape))
        print(f'{out_name:64s} = {shape_str:16s} {data.dtype} → {dtype}')

        # do conversion
        data = data.to(dtype)

        # add to gguf output
        gguf_writer.add_tensor(out_name, data.numpy())

    # DistilBERT has no token_type_embeddings, but src/bert.cpp always
    # gathers row 0 of that tensor (token_type_id is hard-coded to 0). A
    # zero tensor of shape [2, hidden_size] gives the same numerics as
    # DistilBERT's forward without breaking BERT-style consumers.
    if is_distilbert:
        zero_tte = torch.zeros((2, hparams['hidden_size']), dtype=dtype0)
        gguf_writer.add_tensor(
            'embeddings.token_type_embeddings.weight', zero_tte.numpy(),
        )
        print(f'{"embeddings.token_type_embeddings.weight (zero)":64s} = '
              f'{list(zero_tte.shape)} synthesized')
        print(f'  ({n_dropped} DistilBERT tensors dropped)')

    # execute and close writer
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    # print success
    print()
    print(f'GGML model written to {output_path}')

# script usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace BERT/DistilBERT checkpoint to GGUF."
    )
    parser.add_argument("repo_id",
                        help="HuggingFace repo id (e.g. distilbert-base-uncased)")
    parser.add_argument("output_path",
                        help="Path for the resulting .gguf file")
    parser.add_argument("float_type", nargs="?", default="f16",
                        choices=("f16", "f32"),
                        help="Tensor float type (default: f16)")
    parser.add_argument("--with-classifier-head", action="store_true",
                        help="Load via AutoModelForSequenceClassification and "
                             "store pre_classifier / classifier tensors for "
                             "downstream consumers (e.g. distilbert_runner).")
    args = parser.parse_args()

    convert_hf(
        args.repo_id, args.output_path,
        float_type=args.float_type,
        with_classifier_head=args.with_classifier_head,
    )
