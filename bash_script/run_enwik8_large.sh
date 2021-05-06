#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

WORKDIR="/home/baihe/project/Dynasparse-transformer/enwik8/$2"
if [[ $3 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data /home/baihe/data/LM_data/enwik8/ \
        --dataset enwik8 \
        --n_layer 24 \
        --d_model 1024 \
        --n_head 8 \
        --d_head 128 \
        --d_inner 3072 \
        --dropout 0.15 \
        --dropatt 0.15 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 4000 \
        --max_step 400000 \
        --tgt_len 768 \
        --mem_len 768 \
        --eval_tgt_len 128 \
        --fp16 \
        --dynamic-loss-scale \
        --batch_size 64 \
        --multi_gpu \
        --sega \
        --gpu0_bsz 4 \
        --work_dir ${WORKDIR}
elif [[ $3 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data /home/baihe/data/LM_data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 128 \
        --mem_len 3800 \
        --sega \
        --clamp_len 1000 \
        --same_length \
        --split test \
        --work_dir ${WORKDIR}
else
    echo 'unknown argment 1'
fi
