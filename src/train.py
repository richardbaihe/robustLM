# coding: utf-8
import time
import math
import os, sys
import json
import itertools
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import wandb
import mpu
import random
from utils.arguments import get_args
from data_utils import get_lm_corpus
from utils.exp_utils import create_exp_dir, setup_model_and_optimizer, \
    print_rank_0, save_checkpoint, load_checkpoint
from apex import amp, optimizers

def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)
    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

def wandb_init(args):
    wandb_config = json.load(open('wandb_config.json','r'))
    if args.wandb_offline:
        os.environ['WANDB_MODE'] ="offline"
    os.environ['WANDB_API_KEY'] = wandb_config['WANDB_API_KEY']
    os.environ['WANDB_ENTITY'] = wandb_config['WANDB_ENTITY']
    os.environ['WANDB_PROJECT'] = wandb_config['WANDB_PROJECT']
    wandb.init(config=args,name=args.job_name)

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

def evaluate(data_iterator, model, args):
    model.eval()
    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.module.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model.module.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    # Evaluation
    total_lm_len, total_lm_loss = 0, 0.
    total_cl_len, total_cl_loss = 0, 0.
    total_non_cl_len, total_non_cl_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for iteration in range(data_iterator.n_batch):
            if args.max_eval_steps > 0 and iteration >= args.max_eval_steps:
                total_len = args.max_eval_steps*args.eval_tgt_len
                break
            lm_loss, cl_loss, non_cl_loss, new_mems = forward_step(data_iterator, model, mems, iteration, args)
            
            total_lm_len += torch.numel(lm_loss)
            total_lm_loss += lm_loss.sum().float().item()
            total_cl_len += torch.numel(cl_loss)
            total_cl_loss += cl_loss.sum().float().item()
            total_non_cl_len += torch.numel(non_cl_loss)
            total_non_cl_loss += non_cl_loss.sum().float().item()
            
    # Switch back to the training mode
    model.module.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_lm_loss / total_lm_len, total_cl_loss / total_cl_len, total_non_cl_loss / total_non_cl_len

def evaluate_and_print_results(data_iterator, model, args, iteration, mode='valid'):

    lm_loss, cl_lm_loss, non_cl_lm_loss = evaluate(data_iterator, model, args)
    val_lm_ppl, val_cl_ppl, val_non_cl_ppl = math.exp(lm_loss), math.exp(cl_lm_loss), math.exp(non_cl_lm_loss)
    if args.rank == 0:
        wandb.log({mode+"/ppl": val_lm_ppl, mode+"/cl_ppl": val_cl_ppl, mode+"/non_cl_ppl": val_non_cl_ppl}, step=iteration)
    return val_lm_ppl


def get_batch(data_iterator, iteration):
    keys = ['input', 'target', 'cl_input', 'cl_target']
    datatype = torch.int64
    if data_iterator is not None:
        data = data_iterator.get_batch(iteration)
    else:
        data = None
    
    data_b = mpu.broadcast_data(keys, data, datatype)
    input_data = data_b['input']
    target = data_b['target']
    cl_input_data = data_b['cl_input']
    cl_target = data_b['cl_target']
    return input_data, target, cl_input_data, cl_target

def forward_step(data_iterator, model, mems, iteration, args):
    predict_root = False
    if model.training:
        if args.cl_steps!= 0:
            if iteration<args.cl_steps:
                predict_root = True
        elif args.cl_annealing>0:
            cl_portion = max(0, args.cl_annealing - iteration/args.max_step)
            random_number = np.random.randint(1,101)
            predict_root =  random_number<=cl_portion
    data, target, cl_data, cl_target = get_batch(data_iterator,iteration)
    root_mask = cl_target!=target
    if predict_root:
        ret = model(cl_data, cl_target, mems, predict_root)
    else:
        ret = model(data, target, mems, predict_root)
    # data, target = get_batch(data_iterator, iteration)
    # ret = model(data, target, mems)
    loss, new_mems = ret[0], ret[1:]
    cl_loss = loss[root_mask]
    non_cl_loss = loss[~root_mask]
    return loss, cl_loss, non_cl_loss, new_mems

def get_reduced_loss(loss, world_size):
    reduced_losses = loss.view(1)
    torch.distributed.all_reduce(reduced_losses.data)
    reduced_losses.data = reduced_losses.data / world_size
    loss_reduced = reduced_losses[0]
    return loss_reduced

def backward_step(optimizer, model, loss, args):
    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    
    lm_loss_reduced = get_reduced_loss(loss, args.world_size)
    if args.fp16:
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

    return lm_loss_reduced

def train_step(data_iterator, mems, model, optimizer, lr_scheduler, iteration, args):

    lm_loss, cl_loss, non_cl_loss, new_mems = forward_step(data_iterator, model, mems, iteration, args)
    lm_loss = lm_loss.float().mean().type_as(lm_loss)
    lm_loss_reduced = backward_step(optimizer, model, lm_loss, args)
    optimizer.step()

    if args.scheduler in ['cosine', 'constant', 'dev_perf']:
        # linear warmup stage
        if iteration < args.warmup_step:
            curr_lr = args.lr * iteration / args.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            if args.scheduler == 'cosine':
                lr_scheduler.step()
    elif args.scheduler == 'inv_sqrt':
        lr_scheduler.step()

    cl_loss = cl_loss.float().mean().type_as(cl_loss)
    non_cl_loss = non_cl_loss.float().mean().type_as(non_cl_loss)

    cl_lm_loss_reduced = get_reduced_loss(cl_loss,args.world_size)
    non_cl_lm_loss_reduced = get_reduced_loss(non_cl_loss,args.world_size)

    return lm_loss_reduced, cl_lm_loss_reduced, non_cl_lm_loss_reduced, new_mems

def train(model, optimizer, lr_scheduler, train_data_iterator, val_data_iterator, args):
    # Turn on training mode which enables dropout.
    # global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    total_lm_loss = 0.0
    total_cl_lm_loss = 0.0
    total_non_cl_lm_loss = 0.0
    best_val_ppl = 0
    mems = tuple()
    iteration = args.iteration
    while iteration < args.max_step:
        lm_loss, cl_loss, non_cl_loss, mems = train_step(train_data_iterator, mems, model, 
        optimizer, lr_scheduler, iteration, args)
        iteration += 1
        current_lm_loss = lm_loss.data.detach().float().item()
        total_lm_loss += current_lm_loss
        current_cl_lm_loss = cl_loss.data.detach().float().item()
        total_cl_lm_loss += current_cl_lm_loss
        current_non_cl_lm_loss = non_cl_loss.data.detach().float().item()
        total_non_cl_lm_loss += current_non_cl_lm_loss
        # logging
        learning_rate = optimizer.param_groups[0]['lr']
        if args.rank == 0:
            log_dict = {"lr": learning_rate, "lm_loss": current_lm_loss, 
            "cl_lm_loss": current_cl_lm_loss, "non_cl_lm_loss": current_non_cl_lm_loss}
            wandb.log(log_dict,step=iteration)
        if iteration % args.log_interval == 0:
            avg_lm_loss = total_lm_loss / args.log_interval
            avg_cl_lm_loss = total_cl_lm_loss / args.log_interval
            avg_non_cl_lm_loss = total_non_cl_lm_loss / args.log_interval
            if args.rank == 0:
                wandb.log({"train_avg_loss": avg_lm_loss, "train_avg_cl_loss":avg_cl_lm_loss,
                "train_avg_non_cl_loss":avg_non_cl_lm_loss}, step=iteration)
            total_lm_loss = 0
        if iteration % args.eval_interval == 0:
            val_lm_ppl = evaluate_and_print_results(val_data_iterator, model,args, iteration)
            if best_val_ppl==0 or val_lm_ppl < best_val_ppl:
                save_checkpoint(iteration, model, optimizer, lr_scheduler, args, best=True)
                best_val_ppl = val_lm_ppl
        if iteration % args.save_interval == 0 :
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args, best=False)

    return iteration


def main():
    args = get_args()
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        wandb_init(args)
    # writer = SummaryWriter(args.work_dir, flush_secs=30)

    torch.distributed.barrier()
    # Set the random seed manually for reproducibility.
    set_random_seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    corpus = get_lm_corpus(args)

    torch.distributed.barrier()

    model, optimizer, scheduler = setup_model_and_optimizer(args)
    args.iteration = 0
    if args.load_lastest:
        args.iteration = load_checkpoint(model, optimizer, scheduler, args, best=False)

    eval_batch_size = 10
    tr_iter, va_iter, te_iter = None, None, None
    if corpus:
        tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
            device=device, ext_len=args.ext_len,rank=args.rank, world_size=args.world_size)
        va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
            device=device, ext_len=args.ext_len,rank=args.rank, world_size=args.world_size)
        te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
            device=device, ext_len=args.ext_len,rank=args.rank, world_size=args.world_size)

    if args.do_train:
        train(model, optimizer, scheduler, tr_iter, va_iter, args)
    if args.do_test:
        load_checkpoint(model, optimizer, scheduler, args, best=True)
        evaluate_and_print_results(te_iter, model, args, iteration=0, mode='test')
    # vocab parallel
if __name__ == "__main__":
    main()
