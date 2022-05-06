save = """
## SAVE
## where the samples will be written
save_data: {}
"""
vocab = """
## Where the vocab(s) will be written
src_vocab: {}
tgt_vocab: {}
"""


data = """
# CORPUS opts:
data:
    corpus_1:
        path_src: {}
        path_tgt: {}
        transforms: {}
        weight: 1

    valid:
        path_src: {}
        path_tgt: {}
        transforms: {}
"""

reproducibility = """
#SEED
seed: {}
"""

pyonmttok = """
# Specific arguments for pyonmttok
src_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"
tgt_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"
src_seq_length: 512
tgt_seq_length: 512
"""

tokenizers = """
# TOKENIZATION options
src_subword_type: sentencepiece
src_subword_model: {}
tgt_subword_type: sentencepiece
tgt_subword_model: {}
"""

subword_regularization = """
# SUBWORD RRGULARIZATION options
src_subword_nbest: {}
src_subword_alpha: {}
tgt_subword_nbest: {}
tgt_subword_alpha: {}
"""

early_stopping = """
# Early stopping criteria
early_stopping: {}
early_stopping_criteria: {}
"""

general = """
# General opts
save_model: {}
save_checkpoint_steps: {}
valid_steps: {}
train_steps: {}
"""

batching = """
# Batching
queue_size: 8
bucket_size: 8
world_size: 1
gpu_ranks: [0]
batch_type: "sents"
batch_size: 32
valid_batch_size: 8
max_generator_batches: 2
accum_count: [4]
accum_steps: [0]
"""
optimization = """
# Optimization
model_dtype: "fp32"
optim: "adam"
learning_rate: 1
warmup_steps: 4000
decay_method: "noam"
adam_beta2: 0.998
max_grad_norm: 0
label_smoothing: 0.1
param_init: 0
param_init_glorot: true
normalization: "sents"
"""
model = """
# Model
encoder_type: transformer
decoder_type: transformer
position_encoding: true
enc_layers: {}
dec_layers: {}
heads: {}
rnn_size: {}
word_vec_size: {}
transformer_ff: {}
dropout_steps: [0]
dropout: [{}]
attention_dropout: [{}]
"""

logging = """
# Logging
log_file: {}
...
"""


if __name__ == '__main__':
    foo = "model!"

    print(foo)
