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
from collections import defaultdict
import scipy.stats as ss
from tqdm import tqdm

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
from data_utils import MixCorpus
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

def get_lm_corpus(args):
    corpus = None
    kwargs = {}
    if args.dataset in ['wt103']:
        kwargs['special'] = ['<eos>', '<sent_eos>']
        kwargs['lower_case'] = False
    else:
        raise NotImplementedError
    fn = os.path.join(args.data, 'cache.pt')
    if not args.pt and os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset {} with rank {}'.format(args.dataset, args.rank))
        corpus = MixCorpus(args, **kwargs)
        torch.save(corpus, fn)
    #torch.save(corpus, fn)
    ntokens = torch.cuda.LongTensor([len(corpus.vocab)])
    # cl_root_tokens = torch.cuda.LongTensor(corpus.vocab.cl_root_tokens)
    # cl_leaf_tokens = torch.cuda.LongTensor(corpus.vocab.cl_leaf_tokens)
    args.n_token = ntokens.item()
    args.cl_all_root_index = corpus.vocab.cl_root_tokens
    args.cl_all_leaf_index = corpus.vocab.cl_leaf_tokens
    word2class_id = {}
    for k,v in corpus.vocab.word2class.items():
        word2class_id[corpus.vocab.sym2idx[k]] = corpus.vocab.sym2idx[v]
    args.cl_root_leaf_dict = corpus.vocab.class2words
    args.word2class_id = word2class_id if args.learn_offset else None
    return corpus

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
                                     cl_all_root_index=args.cl_all_root_index, cl_all_leaf_index=args.cl_all_leaf_index, adaptive_class_softmax=args.adaptive_class_softmax, cl_root_leaf_dict=args.cl_root_leaf_dict, word2class_id=args.word2class_id)
        model.apply(weights_init)
        model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing
        # if args.learn_offset:
        #     model.hypernym_emb.apply(weights_init)
        #     model.hypernym_emb.weight.data[0,:] = 0
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

def pacing_function(current_step, nbatch_per_epoch, training, args):

    # return true for hypernym prediction, false for token prediction
    # args.a: fraction of training steps/epochs for hypernym prediction
    # args.b: probability of hypernym prediction at step 0
    # function: step: y = b - b*min(1, x//a*total_steps)
    #           linear: y = b - b/(a*total_steps)*x
    #           exponential 
    #           logarithmic 
    class_prediction = False
    total_step = args.max_step
    if args.pacing_unit=='epoch':
        current_step = current_step//nbatch_per_epoch
        total_step = args.max_step//nbatch_per_epoch
    
    if training:
        if args.pacing_function=='step':
            y = args.b - args.b * min(1, current_step//int(args.a*total_step))
        elif args.pacing_function == 'linear':
            y = max(0, args.b - args.b/int(args.a*total_step)*current_step)
        else:
            y = 0
        random_number = np.random.randint(1, 101)
        class_prediction = random_number <= (y*100)
    if args.multi_obj and args.pacing_unit != 'none':
        class_prediction = True
    if args.multi_obj and not training:
        class_prediction = True
    return class_prediction
    
def forward_step(data_iterator, model, mems, iteration, args):
    eval_cl_loss = (args.pacing_unit!='none') and not model.training
    class_prediction = pacing_function(iteration, data_iterator.n_batch, model.training, args)
    data, target, cl_data, cl_target = get_batch(data_iterator,iteration)
    root_mask = cl_target!=target
    input_root = args.input_root and class_prediction

    hypernym_input = cl_data*(cl_data != data)
    if class_prediction:
        if input_root:
            ret = model(cl_data, target, cl_target, mems, args, class_prediction=True)
        else:
            ret = model(data, target, cl_target, mems, args,
                        class_prediction=True, hypernym_input=hypernym_input)
    else:
        ret = model(data, target, cl_target, mems,
                    args, hypernym_input=hypernym_input)
    # ret = model(data, target, cl_data, cl_target, mems, class_prediction, input_root)
    lm_loss, auxiliary_loss, new_mems = ret[0], ret[1], ret[2:]
    if eval_cl_loss:
        if args.input_root:
            ret = model(cl_data, target, cl_target, mems, args, class_prediction=True)
        else:
            ret = model(data, target, cl_target, mems, args,
                        class_prediction=True, hypernym_input=hypernym_input)
        auxiliary_loss = ret[1]
    cl_loss = lm_loss[root_mask]
    non_cl_loss = lm_loss[~root_mask]
    return lm_loss, auxiliary_loss, cl_loss, non_cl_loss, new_mems

def backward_step(optimizer, model, lm_loss, auxiliary_loss, args):
    if args.multi_obj:
        loss = 0.8*lm_loss + 0.2*auxiliary_loss
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
    auxiliary_loss_reduced = get_reduced_loss(auxiliary_loss, args.world_size)

    if args.fp16:
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

    return lm_loss_reduced, auxiliary_loss_reduced

def train_step(data_iterator, mems, model, optimizer, lr_scheduler, iteration, args):

    lm_loss, auxiliary_loss, cl_loss, non_cl_loss, new_mems = forward_step(data_iterator, model, mems, iteration, args)
    lm_loss = lm_loss.float().mean().type_as(lm_loss)
    auxiliary_loss = auxiliary_loss.float().mean().type_as(auxiliary_loss)
    lm_loss_reduced, auxiliary_loss_reduced = backward_step(optimizer, model, lm_loss, auxiliary_loss, args)
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

    return lm_loss_reduced, auxiliary_loss_reduced, cl_lm_loss_reduced, non_cl_lm_loss_reduced, new_mems

def train(model, optimizer, lr_scheduler, corpus, train_data_iterator, val_data_iterator, args):
    # Turn on training mode which enables dropout.
    # global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    total_lm_loss = 0.0
    total_cl_lm_loss = 0.0
    total_non_cl_lm_loss = 0.0
    total_auxiliary_loss = 0.0
    best_val_ppl = math.inf
    mems = tuple()
    iteration = args.iteration
    log_start_time = time.time()
    hypernym_grad = True
    cl_batch_size = False
    if args.pacing_unit == 'step':
        cl_steps = int(args.max_step*args.a)
    elif args.pacing_unit == 'epoch':
        cl_steps = int(
            args.max_step//train_data_iterator.n_batch*args.a)*train_data_iterator.n_batch
    else:
        cl_steps = 0
    while iteration < args.max_step:
        if args.dynamic_wn_layer_start_from >0:
            if iteration>=cl_steps:
                continue
            wn_layer_list = sorted(list(corpus.vocab.word2class_dict.keys()))
            update_interval = int(cl_steps//len(wn_layer_list))
            if iteration % update_interval == 0:
                wn_layer_index = round(iteration/update_interval)
                wn_layer = wn_layer_list[
                    wn_layer_index]
                corpus.rebuild_data_with_wn_layer_n(wn_layer)
                device = torch.device('cuda' if args.cuda else 'cpu')
                tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                            device=device, ext_len=args.ext_len, rank=args.rank, world_size=args.world_size)
                va_iter = corpus.get_iterator('valid', args.eval_batch_size, args.eval_tgt_len,
                                            device=device, ext_len=args.ext_len, rank=0, world_size=1)

        if args.cl_batch_size>0:
            if iteration < cl_steps*2 and not cl_batch_size:
                device = torch.device('cuda' if args.cuda else 'cpu')
                train_data_iterator = corpus.get_iterator('train', args.cl_batch_size, args.tgt_len,
                                            device=device, ext_len=args.ext_len, rank=args.rank, world_size=args.world_size)
                cl_batch_size = True
            elif iteration>=cl_steps*2 and cl_batch_size:
                cl_batch_size=False
                device = torch.device('cuda' if args.cuda else 'cpu')
                train_data_iterator = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                            device=device, ext_len=args.ext_len, rank=args.rank, world_size=args.world_size)
        # if hypernym_grad and args.learn_offset and iteration>args.cl_steps:
        #     model.module.change()
        #     hypernym_grad = False
        lm_loss, auxiliary_loss, cl_loss, non_cl_loss, mems = train_step(train_data_iterator, mems, model, optimizer, lr_scheduler, iteration, args)
        iteration += 1
        current_lm_loss = lm_loss.data.detach().float().item()
        total_lm_loss += current_lm_loss
        current_auxiliary_loss = auxiliary_loss.data.detach().float().item()
        total_auxiliary_loss += current_auxiliary_loss
        current_cl_lm_loss = cl_loss.data.detach().float().item()
        total_cl_lm_loss += current_cl_lm_loss
        current_non_cl_lm_loss = non_cl_loss.data.detach().float().item()
        total_non_cl_lm_loss += current_non_cl_lm_loss
        # logging
        learning_rate = optimizer.param_groups[0]['lr']
        if args.rank == 0:
            log_dict = {"train/lr": learning_rate, "train/lm_loss": current_lm_loss, 
            "train/cl_lm_loss": current_cl_lm_loss, "train/non_cl_lm_loss": current_non_cl_lm_loss, 
            "train/auxiliary_loss":current_auxiliary_loss}
            wandb.log(log_dict,step=iteration)
        if iteration % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            avg_lm_loss = total_lm_loss / args.log_interval
            avg_cl_lm_loss = total_cl_lm_loss / args.log_interval
            avg_non_cl_lm_loss = total_non_cl_lm_loss / args.log_interval
            avg_auxiliary_loss = total_auxiliary_loss / args.log_interval
            if args.rank == 0:
                epoch = iteration//train_data_iterator.n_batch
                log_str = '| epoch {:3d} step {:>8d} | ms/batch {:5.2f} |  lr {:.3g} | loss {:5.2f}'.format(
                    epoch, iteration, elapsed*1000/args.log_interval, learning_rate, avg_lm_loss)
                log_start_time = time.time()
                print_rank_0(log_str)
                wandb.log({"train_avg/lm_loss": avg_lm_loss, "train_avg/cl_lm_loss":avg_cl_lm_loss,
                "train_avg/non_cl_lm_loss":avg_non_cl_lm_loss, "train_avg/auxiliary_loss":avg_auxiliary_loss}, step=iteration)
            total_lm_loss = 0
            total_cl_lm_loss = 0
            total_non_cl_lm_loss = 0
            total_auxiliary_loss = 0
        if iteration % args.save_interval == 0 :
            save_checkpoint(iteration, model, None, None, args, best=False)
        if iteration % args.eval_interval == 0:
            val_lm_ppl = None
            if args.rank == 0:
                val_lm_ppl = evaluate_and_print_results(val_data_iterator, model,args, iteration)
                if val_lm_ppl < best_val_ppl:
                    save_checkpoint(iteration, model, None, None, args, best=True)
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
    total_auxiliary_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for iteration in range(data_iterator.n_batch):
            if args.max_eval_steps > 0 and iteration >= args.max_eval_steps:
                # total_len = args.max_eval_steps*args.eval_tgt_len
                break
            lm_loss, auxiliary_loss, cl_loss, non_cl_loss, mems = forward_step(data_iterator, model, mems, iteration, args)

            seq_len = lm_loss.size()[0]
            total_lm_len += seq_len
            total_lm_loss += lm_loss.mean().float().item()*seq_len
            cl_seq_len = torch.numel(cl_loss)/data_iterator.bsz
            total_cl_len += cl_seq_len
            total_cl_loss += cl_loss.mean().float().item()*cl_seq_len
            total_auxiliary_loss += auxiliary_loss.mean().float().item()*cl_seq_len
            non_cl_seq_len = seq_len-cl_seq_len
            total_non_cl_len += non_cl_seq_len
            total_non_cl_loss += non_cl_loss.mean().float().item()*non_cl_seq_len

            
    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_lm_loss / total_lm_len, total_auxiliary_loss / total_cl_len,\
         total_cl_loss / total_cl_len, total_non_cl_loss / total_non_cl_len

def evaluate_and_print_results(data_iterator, model, args, iteration):
    model.eval()
    eval_start_time = time.time()
    lm_loss, auxiliary_loss, cl_lm_loss, non_cl_lm_loss = evaluate(data_iterator, model, args)
    val_lm_ppl, val_cl_ppl, val_non_cl_ppl = math.exp(lm_loss), math.exp(cl_lm_loss), math.exp(non_cl_lm_loss)
    if args.rank == 0:
        log_str = '| Eval {:3d} at step {:>8d} |  time: {:5.2f}s | valid loss {:5.2f} | auxiliary loss {:5.2f} | valid ppl {:5.2f}'.format(
                iteration // args.eval_interval, iteration, (time.time() - eval_start_time),
                 lm_loss, auxiliary_loss, val_lm_ppl)
        print_rank_0(log_str)
        wandb.log({"valid/ppl": val_lm_ppl, "valid/cl_ppl": val_cl_ppl, "valid/non_cl_ppl": val_non_cl_ppl, "valid/auxiliary_loss":auxiliary_loss}, step=iteration)
    model.train()
    return val_lm_ppl

def test(data_iterator, model, args, iteration):
    model.eval()
    eval_start_time = time.time()
    lm_loss, auxiliary_loss, cl_lm_loss, non_cl_lm_loss = evaluate(data_iterator, model, args)
    val_lm_ppl, val_cl_ppl, val_non_cl_ppl = math.exp(lm_loss), math.exp(cl_lm_loss), math.exp(non_cl_lm_loss)
    if args.rank == 0:
        log_str = '| Test at step {:>8d} |  time: {:5.2f}s | test loss {:5.2f} | auxiliary loss {:5.2f} | test ppl {:5.2f}'.format(
                iteration, (time.time() - eval_start_time),
                 lm_loss, auxiliary_loss, val_lm_ppl)
        print_rank_0(log_str)
        wandb.log({"test/ppl": val_lm_ppl, "test/cl_ppl": val_cl_ppl, "test/non_cl_ppl": val_non_cl_ppl, "test/auxiliary_loss":auxiliary_loss})
    model.train()
    return val_lm_ppl

def get_topk_pred(model, va_iter,top_k=1000):
    model.eval()
    while isinstance(model, (torchDDP, apexDDP, FP16_Module)):
        model = model.module
    top_k_pred = defaultdict(list)
    mems=tuple()
    for iteration in tqdm(range(va_iter.n_batch)):
        if not mems:
            mems = model.init_mems()
        data, target, cl_data, cl_target = get_batch(va_iter,iteration)
        tgt_len = target.size(0)
        root_mask = cl_target!=target
        hidden, hiddens, new_mems = model._forward(data, mems=mems)
        pred_hid = hidden[-tgt_len:]
        pred_hid = torch.reshape(
                pred_hid, (-1, pred_hid.size(-1)))

        topk_words,topk_probs = model.crit.get_top_k_words_and_props(pred_hid,torch.reshape(target, (-1,)), top_k=top_k)   
        for k,v in zip(target.view(-1).tolist(), topk_words.tolist()):
            top_k_pred[k].append(v)
    return top_k_pred

def get_gt_rank_and_prob(model, va_iter,cl_all_leaf_sym, class2words, word2class):

    model.eval()
    while isinstance(model, (torchDDP, apexDDP, FP16_Module)):
        model = model.module
    results_rank = defaultdict(list)
    results_prob = defaultdict(list)
    results_sib_rank = defaultdict(list)
    results_sib_prob = defaultdict(list)
    results_sib_rank_among_classes = defaultdict(list)
    mems=tuple()
    for iteration in tqdm(range(va_iter.n_batch)):
        if not mems:
            mems = model.init_mems()
        data, target, cl_data, cl_target = get_batch(va_iter,iteration)
        tgt_len = target.size(0)
        root_mask = cl_target!=target
        hidden, hiddens, new_mems = model._forward(data, mems=mems)
        pred_hid = hidden[-tgt_len:]
        pred_hid = torch.reshape(
                pred_hid, (-1, pred_hid.size(-1)))
        target = torch.reshape(target, (-1,))
        mask = []
        for i,k in enumerate(target.view(-1).tolist()):
            if k not in cl_all_leaf_sym:
                mask.append(False)
            else:
                mask.append(True)
        mask = torch.tensor(mask)
        pred_hid = pred_hid[mask]
        target = target[mask]
        probs = model.crit.get_top_k_words_and_props(pred_hid, target, top_k=-1)
        for i,k in enumerate(target.view(-1).tolist()):
            if k not in cl_all_leaf_sym:
                continue
            else:
                siblings = class2words[word2class[k]]
                v = probs[i].tolist()
                sib_prob = -math.log(sum([math.exp(-v[idx]) for idx in siblings]))
                
                # ranked_v = ss.rankdata(v)
                # sib_rank = sum([1/(len(v) - ranked_v[idx]) for idx in siblings])/len(siblings)
                # normalized_sib_rank = 0
                # for cl, sibs in class2words.items():
                #     if cl == word2class[k]:
                #         continue
                #     if sib_rank > sum([1/(len(v) - ranked_v[idx]) for idx in sibs])/len(sibs):
                #         normalized_sib_rank+=1
                # rank = len(v) - ranked_v[k]
                sib_rank=0
                rank=0
                normalized_sib_rank=0

                results_rank[k].append(rank)
                results_prob[k].append(v[k])
                results_sib_prob[k].append(sib_prob)
                results_sib_rank[k].append(sib_rank)
                results_sib_rank_among_classes[k].append(normalized_sib_rank)
    return results_rank, results_prob, results_sib_rank, results_sib_prob,results_sib_rank_among_classes

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

    args.eval_batch_size = 10
    tr_iter, va_iter, te_iter = None, None, None
    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
        device=device, ext_len=args.ext_len,rank=args.rank, world_size=args.world_size)
    va_iter = corpus.get_iterator('valid', args.eval_batch_size, args.eval_tgt_len,
        device=device, ext_len=args.ext_len,rank=0, world_size=1)
    te_iter = corpus.get_iterator('test', args.eval_batch_size, args.eval_tgt_len,
        device=device, ext_len=args.ext_len,rank=0, world_size=1)

    if args.do_train:
        train(model, optimizer, scheduler, corpus, tr_iter, va_iter, args)
    if args.do_eval and args.checkpoint_dir:
        for checkpoint_name in glob.glob(os.path.join(args.checkpoint_dir,'iter_*/model_optim_rng.pt')):
            iteration = load_checkpoint(model, optimizer, scheduler, args, checkpoint_name=checkpoint_name)
            if args.rank == 0:
                evaluate_and_print_results(va_iter, model, args, iteration=iteration)
            torch.distributed.barrier()
    if args.do_test:
        checkpoint_name = ""
        if args.checkpoint_dir:
            checkpoint_name = os.path.join(args.checkpoint_dir,'best/model_optim_rng.pt')
        iteration = load_checkpoint(model, optimizer, scheduler, args, best=True, checkpoint_name=checkpoint_name)
        if args.rank == 0:
            test(te_iter, model, args, iteration=iteration)

if __name__ == "__main__":
    main()
