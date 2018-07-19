#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 -u -m QG.nmt.nmt \
    --out_dir=QG/model \
    --inference_input_file=QG/data/test.in \
    --inference_output_file=infer.out


 
