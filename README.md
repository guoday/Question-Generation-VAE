### Introduction

Implement of QG model in the paper "Question Generation from SQL  Queries Improves Neural Semantic Parsing". This code is based on the repo, you can refer to more detail about API: https://github.com/tensorflow/nmt  

### Requirements

- python3

- TensorFlow==1.4 (It seems that only 1.4 version can work)

  

### Preprocessing data

```shell
unzip data.zip
python data/preprocess.py
```



### Train QG model

```shell
rm -r model
mkdir model
export CUDA_VISIBLE_DEVICES=0
python3 -u -m nmt.nmt \
--src=in --tgt=out \
--vocab_prefix=data/vocab \
--train_prefix=data/train \
--share_vocab=true \
--dev_prefix=data/dev \
--test_prefix=data/test \
--out_dir=model \
--attention=scaled_luong \
--attention_architecture=standard \
--batch_size=32 \
--learning_rate=0.001 \
--optimizer=adam \
--src_max_len=100 \
--tgt_max_len=100 \
--eos=_eos_ \
--sos=_sos_ \
--num_train_steps=6000 \
--steps_per_stats=200 \
--num_layers=2 \
--dropout=0.5 \
--metrics=bleu \
--beam_width=5 \
--num_units=300 \
--override_loaded_hparams=false \
--encoder_type=bi \
--src_vocab_size=50000 \
--tgt_vocab_size=50000 \
--num_keep_ckpts=100 \
--z_hidden_size=64 \ #if z_hidden_size=0, then not VAE 
--kl_steps=1000 \
--steps_per_external_eval=2000 \
--max_kl_weight=1

```



### Inference

Because we incorporate VAE in the QG model, you can inference multiple times to generate questions with diversity.

```shell
export CUDA_VISIBLE_DEVICES=0
python3 -u -m nmt.nmt \
--out_dir=model \
--inference_input_file=data/sql_gen.in \
--inference_output_file=infer.out
```


