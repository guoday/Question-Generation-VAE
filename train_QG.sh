#!/bin/bash
rm -r QG/model
mkdir QG/model
export CUDA_VISIBLE_DEVICES=7
python3 -u -m QG.nmt.nmt \
    --src=in --tgt=out \
    --vocab_prefix=QG/data/vocab \
    --train_prefix=QG/data/train \
    --share_vocab=true \
    --dev_prefix=QG/data/dev \
    --test_prefix=QG/data/test \
    --out_dir=QG/model\
    --attention=scaled_luong \
    --attention_architecture=standard \
    --batch_size=64 \
    --learning_rate=0.001 \
    --optimizer=adam \
    --src_max_len=250 \
    --tgt_max_len=100 \
    --eos=_eos_ \
    --sos=_sos_ \
    --num_train_steps=6000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --dropout=0.5 \
    --metrics=bleu,accuracy \
    --beam_width=5 \
    --num_units=300 \
    --override_loaded_hparams=false \
    --encoder_type=bi \
    --src_vocab_size=50000 \
    --tgt_vocab_size=50000 \
    --num_keep_ckpts=100 \
    --z_hidden_size=0 \
    --kl_steps=1000 \
    --steps_per_external_eval=1000 \
    --max_kl_weight=1
