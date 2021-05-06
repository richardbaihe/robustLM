#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

WORKDIR="/home/baihe/project/Dynasparse-transformer/enwik8/$2"
if [[ $3 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data /home/baihe/data/LM_data/enwik8/ \
        --dataset enwik8 \
        --n_layer 12 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 400000 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 128 \
        --fp16 \
        --dynamic-loss-scale \
        --batch_size 128 \
        --multi_gpu \
        --sega \
        --gpu0_bsz 10 \
        --work_dir ${WORKDIR}
elif [[ $3 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data /home/baihe/data/LM_data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --sega \
        --clamp_len 820 \
        --same_length \
        --split test \
        --work_dir ${WORKDIR}
else
    echo 'unknown argment 1'
fi