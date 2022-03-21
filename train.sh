#!/bin/bash
WORKDIR=''

# wikitext103
JOB_NAME=small_transformer_hcp_wt103
python ${WORKDIR}/src/train.py \
    --data ${WORKDIR}/data/wikitext-103/ \
    --dataset wt103 \
    --work_dir ${WORKDIR}/output/${JOB_NAME}\
    --job_name ${JOB_NAME} \
    --batch_size 256 \
    --pacing_function step \
    --pacing_unit step \
    --a 0.12 \
    --b 0.8 \
    --model_size wt_small \
    --max_step 100000 \
    --mix_vocab \
    --adaptive \
    --fp16 \
    --dynamic-loss-scale \
    --multi_gpu \
    --input_root \
    --cuda \
    --do_train \
    --do_eval \
    --do_test \
    --lr 0.00025 \
    --eval_interval 4000 \
    --wandb_offline



JOB_NAME=base_segaxl_hcp_wt103
python ${WORKDIR}/src/train.py \
    --data ${WORKDIR}/data/wikitext-103/ \
    --dataset wt103 \
    --work_dir ${WORKDIR}/output/${JOB_NAME}\
    --job_name ${JOB_NAME} \
    --batch_size 64 \
    --sega \
    --mem_len 150 \
    --pacing_function step \
    --pacing_unit step \
    --a 0.12 \
    --b 0.8 \
    --model_size wt_base \
    --max_step 200000 \
    --mix_vocab \
    --adaptive \
    --fp16 \
    --dynamic-loss-scale \
    --multi_gpu \
    --input_root \
    --cuda \
    --do_train \
    --do_eval \
    --do_test \
    --lr 0.00025 \
    --eval_interval 4000 \
    --wandb_offline



JOB_NAME=large_segaxl_hcp_wt103
python ${WORKDIR}/src/train.py \
    --data ${WORKDIR}/data/wikitext-103/ \
    --dataset wt103 \
    --accumulation_steps 2 \
    --work_dir ${WORKDIR}/output/${JOB_NAME}\
    --job_name ${JOB_NAME} \
    --batch_size 128 \
    --sega \
    --mem_len 384 \
    --pacing_function step \
    --pacing_unit step \
    --a 0.12 \
    --b 0.8 \
    --model_size wt_large \
    --max_step 500000 \
    --break_step 350000 \
    --mix_vocab \
    --adaptive \
    --fp16 \
    --dynamic-loss-scale \
    --multi_gpu \
    --input_root \
    --cuda \
    --do_train \
    --do_eval \
    --do_test \
    --div_val 4 \
    --restart \
    --restart_dir ${WORKDIR}/output/${JOB_NAME} \
    --further_warmup \
    --warmup_step 16000 \
    --lr 0.00025 \
    --eval_interval 4000 \
    --wandb_offline

# arxiv dataset
JOB_NAME=base_segaxl_hcp_ax
python -u ${WORKDIR}/src/train_old.py \
    --data ${WORKDIR}/data/arxiv/timed/ \
    --dataset arxiv \
    --work_dir ${WORKDIR}/output/${JOB_NAME}\
    --job_name ${JOB_NAME} \
    --batch_size 64 \
    --sega \
    --mem_len 384 \
    --pacing_function step \
    --pacing_unit step \
    --a 0.12 \
    --b 0.8 \
    --model_size arxiv_base \
    --max_step 200000 \
    --mix_vocab \
    --ignore_freqency_threshold 1000 \
    --fp16 \
    --dynamic-loss-scale \
    --multi_gpu \
    --input_root \
    --cuda \
    --do_train \
    --do_eval \
    --do_test \
    --lr 0.00025 \
    --eval_interval 2000 \
    --wandb_offline



JOB_NAME=large_segaxl_hcp_ax
python -u ${WORKDIR}/src/train_old.py \
    --data ${WORKDIR}/data/arxiv/timed/ \
    --dataset arxiv \
    --work_dir ${WORKDIR}/output/${JOB_NAME}\
    --job_name ${JOB_NAME} \
    --batch_size 128 \
    --sega \
    --mem_len 384 \
    --pacing_function step \
    --pacing_unit step \
    --a 0.12 \
    --b 0.8 \
    --model_size arxiv_large \
    --max_step 200000 \
    --break_step 80000 \
    --mix_vocab \
    --ignore_freqency_threshold 1000 \
    --fp16 \
    --dynamic-loss-scale \
    --multi_gpu \
    --input_root \
    --cuda \
    --do_train \
    --do_eval \
    --do_test \
    --warmup_step 8000 \
    --accumulation_steps 2 \
    --lr 0.00025 \
    --eval_interval 2000 \
    --wandb_offline




