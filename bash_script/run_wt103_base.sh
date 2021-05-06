#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

WORKDIR="/home/baihe/project/Dynasparse-transforme/wiki103/$2"
if [[ $3 == 'train' ]]; then
    echo 'Run training...'
    python -u train.py \
        --cuda \
        --data /home/baihe/data/LM_data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --droh_size 64 \
        --fp16 \
        --dpout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000 \
        --tgt_len 25 \
        --mem_len 25 \
        --attn_type 0 \
        --eval_tgt_len 25 \
        --batcynamic-loss-scale \
        --multi_gpu \
        --work_dir ${WORKDIR}
elif [[ $3 == 'eval' ]]; then
    echo 'Run evaluation...'
    python -u eval.py \
        --cuda \
        --data /home/baihe/data/LM_data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 64 \
        --mem_len 640 \
        --clamp_len 300 \
        --batch_size 10 \
        --split test \
        --sega \
        --work_dir ${WORKDIR}
else
    echo 'unknown argment 1'
fi
