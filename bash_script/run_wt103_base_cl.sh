#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#        --multi_gpu \
#        --gpu0_bsz 32 \
WORKDIR="/home/baihe/project/Dynasparse-transformer/wiki103/$2"
if [[ $3 == 'train' ]]; then
    echo 'Run training...'
    python -u train.py \
        --cuda \
        --data /home/baihe/datasets/LM_data/CL/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 100000 \
        --tgt_len 150 \
        --mem_len 0 \
        --attn_type 0 \
        --eval_tgt_len 150 \
        --batch_size 64 \
        --fp16 \
        --dynamic-loss-scale \
        --multi_gpu \
        --work_dir ${WORKDIR}
elif [[ $3 == 'eval' ]]; then
    echo 'Run evaluation...'
    python -u eval.py \
        --cuda \
        --data /home/baihe/datasets/LM_data/CL/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 400 \
        --mem_len 0 \
        --batch_size 10 \
        --split test \
        --work_dir ${WORKDIR}
else
    echo 'unknown argment 3'
fi
