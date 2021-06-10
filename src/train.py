# coding: utf-8
import time
import math
import os, sys
import json
import random
import glob
import itertools
import logging
import numpy as np
import wandb

import mpu
import torch
import torch.nn as nn
import torch.optim as optim
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from apex.parallel import DistributedDataParallel as apexDDP
from apex.optimizers import FusedAdam as Adam
from apex import amp, optimizers
from apex import amp, optimizers
from utils.arguments import get_args
from data_utils import get_lm_corpus
from utils.exp_utils import create_exp_dir, \
    print_rank_0, save_checkpoint, load_checkpoint, get_params_for_weight_decay_optimization
from mem_transformer import MemTransformerLM, SegaMemTransformerLM


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
    # # Set the model-parallel / data-parallel communicators.
    # mpu.initialize_model_parallel(args.model_parallel_size)

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
        # mpu.model_parallel_cuda_manual_seed(seed)

def get_model(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [20000, 40000, 200000]
            tie_projs += [True] * len(cutoffs)
        elif args.dataset == 'lm1b':
            cutoffs = [60000, 100000, 640000]
            tie_projs += [False] * len(cutoffs)
    args.tied_projs = tie_projs[0]
    def init_weight(weight):
        if args.init == 'uniform':
            nn.init.uniform_(weight, -args.init_range, args.init_range)
        elif args.init == 'normal':
            nn.init.normal_(weight, 0.0, args.init_std)

    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, args.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)

    def update_dropout(m):
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            if hasattr(m, 'p'):
                m.p = args.dropout

    def update_dropatt(m):
        if hasattr(m, 'dropatt'):
            m.dropatt.p = args.dropatt

    if args.restart:
        with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)
        if not args.fp16:
            model = model.float()
        model.apply(update_dropout)
        model.apply(update_dropatt)
    else:
        if args.sega:
            model = SegaMemTransformerLM(args.n_token, args.n_layer, args.n_head, args.d_model,
            args.d_head, args.d_inner, args.dropout, args.dropatt,
            tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
            tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
            ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
            same_length=args.same_length, attn_type=args.attn_type,
            clamp_len=args.clamp_len, sample_softmax=args.sample_softmax,sparse_mode=args.sparse_mode)
        else:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head, args.d_model,
                args.d_head, args.d_inner, args.dropout, args.dropatt,
                tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
                tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
                same_length=args.same_length, attn_type=args.attn_type,
                clamp_len=args.clamp_len, sample_softmax=args.sample_softmax, 
                cl_all_root_index=args.cl_all_root_index, cl_all_leaf_index=args.cl_all_leaf_index)
        model.apply(weights_init)
        model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing

    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])
    model = model.to(device)
    return model

def get_optimizer(model, args):
    while isinstance(model, (torchDDP, apexDDP, FP16_Module)):
        model = model.module
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer

def get_learning_rate_scheduler(optimizer, args):
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                    else step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    elif args.scheduler == 'constant':
        pass
    return scheduler

def setup_model_and_optimizer(args):
    #### get model
    model = get_model(args)
    #### optimizer
    optimizer = get_optimizer(model, args)
    #### scheduler
    scheduler = get_learning_rate_scheduler(optimizer, args)
    args.iteration = 0
    if args.load_lastest:
        args.iteration = load_checkpoint(model, optimizer, scheduler, args, best=False)
    
    if args.fp16:
        # model = model.half()
        model, optimizer = amp.initialize(model, optimizer,
                                      opt_level="O2",
                                      loss_scale='dynamic'
                                      )
    model = apexDDP(model)
    return model, optimizer, scheduler

def get_batch(data_iterator, iteration):
    data = data_iterator.get_batch(iteration)
    input_data = data['input']
    target = data['target']
    cl_input_data = data['cl_input']
    cl_target = data['cl_target']
    return input_data, target, cl_input_data, cl_target

def get_reduced_loss(loss, world_size):
    reduced_losses = loss.view(1)
    torch.distributed.all_reduce(reduced_losses.data)
    reduced_losses.data = reduced_losses.data / world_size
    loss_reduced = reduced_losses[0]
    return loss_reduced

def forward_step(data_iterator, model, mems, iteration, args):
    class_prediction = False
    eval_cl_loss = (args.cl_steps!= 0 or args.cl_annealing>0) and not model.training
    if model.training:
        if args.cl_steps!= 0:
            if iteration<args.cl_steps:
                class_prediction = True
        elif args.cl_annealing>0:
            cl_portion = max(0, args.cl_annealing - iteration/args.max_step)
            random_number = np.random.randint(1,101)
            class_prediction =  random_number<=(cl_portion*100)
    if args.multi_obj and args.cl_steps== 0 and args.cl_annealing==0:
        class_prediction = True
    if args.multi_obj and not model.training:
        class_prediction = True
    data, target, cl_data, cl_target = get_batch(data_iterator,iteration)
    root_mask = cl_target!=target
    input_root = args.input_root and class_prediction
    assert not (args.mix_vocab and args.multi_obj), "if predicted with multi-objective, can't set mix_vocab=True"
    if class_prediction:
        if input_root:
            ret = model(cl_data, target, cl_target, mems, class_prediction=True, multi_obj=args.multi_obj, mix_vocab=args.mix_vocab)
        else:
            ret = model(data, target, cl_target, mems, class_prediction=True, multi_obj=args.multi_obj, mix_vocab=args.mix_vocab)
    else:
        ret = model(data, target, cl_target, mems, mix_vocab=args.mix_vocab)
    # ret = model(data, target, cl_data, cl_target, mems, class_prediction, input_root)
    lm_loss, auxilary_loss, new_mems = ret[0], ret[1], ret[2:]
    if eval_cl_loss:
        if args.input_root:
            ret = model(cl_data, target, cl_target, mems, class_prediction=True, multi_obj=args.multi_obj, mix_vocab=args.mix_vocab)
        else:
            ret = model(data, target, cl_target, mems, class_prediction=True, multi_obj=args.multi_obj, mix_vocab=args.mix_vocab)
        auxilary_loss = ret[1]
    cl_loss = lm_loss[root_mask]
    non_cl_loss = lm_loss[~root_mask]
    return lm_loss, auxilary_loss, cl_loss, non_cl_loss, new_mems

def backward_step(optimizer, model, lm_loss, auxilary_loss, args):
    if args.multi_obj:
        loss = 0.8*lm_loss + 0.2*auxilary_loss
    else:
        loss = lm_loss
    optimizer.zero_grad()
    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()
    
    lm_loss_reduced = get_reduced_loss(lm_loss, args.world_size)
    auxilary_loss_reduced = get_reduced_loss(auxilary_loss, args.world_size)

    if args.fp16:
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

    return lm_loss_reduced, auxilary_loss_reduced

def train_step(data_iterator, mems, model, optimizer, lr_scheduler, iteration, args):

    lm_loss, auxilary_loss, cl_loss, non_cl_loss, new_mems = forward_step(data_iterator, model, mems, iteration, args)
    lm_loss = lm_loss.float().mean().type_as(lm_loss)
    auxilary_loss = auxilary_loss.float().mean().type_as(auxilary_loss)
    lm_loss_reduced, auxilary_loss_reduced = backward_step(optimizer, model, lm_loss, auxilary_loss, args)
    optimizer.step()

    if iteration < args.warmup_step:
        curr_lr = args.lr * iteration / args.warmup_step
        optimizer.param_groups[0]['lr'] = curr_lr
    else:
        lr_scheduler.step(iteration)
    cl_loss = cl_loss.float().mean().type_as(cl_loss)
    non_cl_loss = non_cl_loss.float().mean().type_as(non_cl_loss)

    cl_lm_loss_reduced = get_reduced_loss(cl_loss,args.world_size)
    non_cl_lm_loss_reduced = get_reduced_loss(non_cl_loss,args.world_size)

    return lm_loss_reduced, auxilary_loss_reduced, cl_lm_loss_reduced, non_cl_lm_loss_reduced, new_mems

def train(model, optimizer, lr_scheduler, train_data_iterator, val_data_iterator, args):
    # Turn on training mode which enables dropout.
    # global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    total_lm_loss = 0.0
    total_cl_lm_loss = 0.0
    total_non_cl_lm_loss = 0.0
    total_auxilary_loss = 0.0
    best_val_ppl = math.inf
    mems = tuple()
    iteration = args.iteration
    log_start_time = time.time()
    while iteration < args.max_step:
        lm_loss, auxilary_loss, cl_loss, non_cl_loss, mems = train_step(train_data_iterator, mems, model, optimizer, lr_scheduler, iteration, args)
        iteration += 1
        current_lm_loss = lm_loss.data.detach().float().item()
        total_lm_loss += current_lm_loss
        current_auxilary_loss = auxilary_loss.data.detach().float().item()
        total_auxilary_loss += current_auxilary_loss
        current_cl_lm_loss = cl_loss.data.detach().float().item()
        total_cl_lm_loss += current_cl_lm_loss
        current_non_cl_lm_loss = non_cl_loss.data.detach().float().item()
        total_non_cl_lm_loss += current_non_cl_lm_loss
        # logging
        learning_rate = optimizer.param_groups[0]['lr']
        if args.rank == 0:
            log_dict = {"train/lr": learning_rate, "train/lm_loss": current_lm_loss, 
            "train/cl_lm_loss": current_cl_lm_loss, "train/non_cl_lm_loss": current_non_cl_lm_loss, 
            "train/auxilary_loss":current_auxilary_loss}
            wandb.log(log_dict,step=iteration)
        if iteration % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            avg_lm_loss = total_lm_loss / args.log_interval
            avg_cl_lm_loss = total_cl_lm_loss / args.log_interval
            avg_non_cl_lm_loss = total_non_cl_lm_loss / args.log_interval
            avg_auxilary_loss = total_auxilary_loss / args.log_interval
            if args.rank == 0:
                epoch = iteration//train_data_iterator.n_batch
                log_str = '| epoch {:3d} step {:>8d} | ms/batch {:5.2f} |  lr {:.3g} | loss {:5.2f}'.format(
                    epoch, iteration, elapsed*1000/args.log_interval, learning_rate, avg_lm_loss)
                log_start_time = time.time()
                print_rank_0(log_str)
                wandb.log({"train_avg/lm_loss": avg_lm_loss, "train_avg/cl_lm_loss":avg_cl_lm_loss,
                "train_avg/non_cl_lm_loss":avg_non_cl_lm_loss, "train_avg/auxilary_loss":avg_auxilary_loss}, step=iteration)
            total_lm_loss = 0
            total_cl_lm_loss = 0
            total_non_cl_lm_loss = 0
            total_auxilary_loss = 0
        if iteration % args.save_interval == 0 :
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args, best=False)
        if iteration % args.eval_interval == 0:
            val_lm_ppl = None
            if args.rank == 0:
                val_lm_ppl = evaluate_and_print_results(val_data_iterator, model,args, iteration)
                if val_lm_ppl < best_val_ppl:
                    save_checkpoint(iteration, model, optimizer, lr_scheduler, args, best=True)
                    best_val_ppl = val_lm_ppl
            torch.distributed.barrier()

    return iteration

def evaluate(data_iterator, model, args):
    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    while isinstance(model, (torchDDP, apexDDP, FP16_Module)):
        model = model.module
    model.eval()
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    # Evaluation
    total_lm_len, total_lm_loss = 0, 0.
    total_cl_len, total_cl_loss = 0, 0.
    total_non_cl_len, total_non_cl_loss = 0, 0.
    total_auxilary_len, total_auxilary_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for iteration in range(data_iterator.n_batch):
            if args.max_eval_steps > 0 and iteration >= args.max_eval_steps:
                total_len = args.max_eval_steps*args.eval_tgt_len
                break
            lm_loss, auxilary_loss, cl_loss, non_cl_loss, new_mems = forward_step(data_iterator, model, mems, iteration, args)

            seq_len = lm_loss.size()[0]
            total_lm_len += seq_len
            total_lm_loss += lm_loss.mean().float().item()*seq_len
            cl_seq_len = torch.numel(cl_loss)/data_iterator.bsz
            total_cl_len += cl_seq_len
            total_cl_loss += cl_loss.mean().float().item()*cl_seq_len
            total_auxilary_loss += auxilary_loss.mean().float().item()*cl_seq_len
            non_cl_seq_len = seq_len-cl_seq_len
            total_non_cl_len += non_cl_seq_len
            total_non_cl_loss += non_cl_loss.mean().float().item()*non_cl_seq_len

            
    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_lm_loss / total_lm_len, total_auxilary_loss / total_cl_len,\
         total_cl_loss / total_cl_len, total_non_cl_loss / total_non_cl_len

def evaluate_and_print_results(data_iterator, model, args, iteration):
    model.eval()
    eval_start_time = time.time()
    lm_loss, auxilary_loss, cl_lm_loss, non_cl_lm_loss = evaluate(data_iterator, model, args)
    val_lm_ppl, val_cl_ppl, val_non_cl_ppl = math.exp(lm_loss), math.exp(cl_lm_loss), math.exp(non_cl_lm_loss)
    if args.rank == 0:
        log_str = '| Eval {:3d} at step {:>8d} |  time: {:5.2f}s | valid loss {:5.2f} | auxiliary loss {:5.2f} | valid ppl {:5.2f}'.format(
                iteration // args.eval_interval, iteration, (time.time() - eval_start_time),
                 lm_loss, auxilary_loss, val_lm_ppl)
        print_rank_0(log_str)
        wandb.log({"valid/ppl": val_lm_ppl, "valid/cl_ppl": val_cl_ppl, "valid/non_cl_ppl": val_non_cl_ppl, "valid/auxilary_loss":auxilary_loss}, step=iteration)
    model.train()
    return val_lm_ppl

def test(data_iterator, model, args, iteration):
    model.eval()
    eval_start_time = time.time()
    lm_loss, auxilary_loss, cl_lm_loss, non_cl_lm_loss = evaluate(data_iterator, model, args)
    val_lm_ppl, val_cl_ppl, val_non_cl_ppl = math.exp(lm_loss), math.exp(cl_lm_loss), math.exp(non_cl_lm_loss)
    if args.rank == 0:
        log_str = '| Test at step {:>8d} |  time: {:5.2f}s | test loss {:5.2f} | auxiliary loss {:5.2f} | test ppl {:5.2f}'.format(
                iteration, (time.time() - eval_start_time),
                 lm_loss, auxilary_loss, val_lm_ppl)
        print_rank_0(log_str)
        wandb.log({"test/ppl": val_lm_ppl, "test/cl_ppl": val_cl_ppl, "test/non_cl_ppl": val_non_cl_ppl, "test/auxilary_loss":auxilary_loss})
    model.train()
    return val_lm_ppl

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

    eval_batch_size = 10
    tr_iter, va_iter, te_iter = None, None, None
    if corpus:
        tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
            device=device, ext_len=args.ext_len,rank=args.rank, world_size=args.world_size)
        va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
            device=device, ext_len=args.ext_len,rank=0, world_size=1)
        te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
            device=device, ext_len=args.ext_len,rank=0, world_size=1)

    if args.do_train:
        train(model, optimizer, scheduler, tr_iter, va_iter, args)
    if args.do_eval and args.checkpoint_dir:
        for checkpoint_name in glob.glob(os.path.join(args.checkpoint_dir,'iter_*/model_optim_rng.pt')):
            if args.rank == 0:
                iteration = load_checkpoint(model, optimizer, scheduler, args, checkpoint_name=checkpoint_name)
                evaluate_and_print_results(va_iter, model, args, iteration=iteration)
            torch.distributed.barrier()
    if args.do_test:
        checkpoint_name = ""
        if args.checkpoint_dir:
            checkpoint_name = os.path.join(args.checkpoint_dir,'best/model_optim_rng.pt')
        if args.rank == 0:
            iteration = load_checkpoint(model, optimizer, scheduler, args, best=True, checkpoint_name=checkpoint_name)
            test(te_iter, model, args, iteration=iteration)
        torch.distributed.barrier()
    # vocab parallel

if __name__ == "__main__":
    main()
