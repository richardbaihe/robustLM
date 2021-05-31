import functools
import os, shutil

import numpy as np
import mpu
import random
import torch
import torch.nn as nn
import torch.optim as optim
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from apex.parallel import DistributedDataParallel as apexDDP
from apex.multi_tensor_apply import multi_tensor_applier
from mem_transformer import MemTransformerLM, SegaMemTransformerLM
from utils.data_parallel import BalancedDataParallel

def get_checkpoint_name(checkpoints_path, iteration, best=False, mp_rank=None):
    if best:
        d = 'best'
    else:
        d = 'iter_{:07d}'.format(iteration)
    return os.path.join(checkpoints_path, d,
                        'mp_rank_{:02d}'.format(mpu.get_model_parallel_rank() if mp_rank is None else mp_rank),
                        'model_optim_rng.pt')

def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')

def save_checkpoint(iteration, model, optimizer, lr_scheduler, args, best):
    """Save a model checkpoint."""
    # Only rank zer0 of the data parallel writes to the disk.
    if isinstance(model, torchDDP):
        model = model.module
    if mpu.get_data_parallel_rank() == 0:
        checkpoint_name = get_checkpoint_name(args.work_dir, iteration, best)

        print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
              format(torch.distributed.get_rank(), iteration, checkpoint_name))

        sd = {}
        sd['iteration'] = iteration
        sd['model'] = model.state_dict()

        # Optimizer stuff.
        if optimizer is not None:
            sd['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            sd['lr_scheduler'] = lr_scheduler.state_dict()

        # rng states.
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

        ensure_directory_exists(checkpoint_name)
        torch.save(sd, checkpoint_name)
        print('  successfully saved {}'.format(checkpoint_name))
    if not best:
        # Wait so everyone is done (necessary)
        torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0 and not best:
        tracker_filename = get_checkpoint_tracker_filename(args.work_dir)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    if not best:
        # Wait so everyone is done (not necessary)
        torch.distributed.barrier()

def load_checkpoint(model, optimizer, lr_scheduler, args, best=False):
    """Load a model checkpoint."""
    if isinstance(model, torchDDP):
        model = model.module
    iteration = 0
    if not best:
        # Read the tracker file and set the iteration.
        tracker_filename = get_checkpoint_tracker_filename(args.work_dir)
        if not os.path.isfile(tracker_filename):
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from '
                        'random')
            return 0
        with open(tracker_filename, 'r') as f:
            metastring = f.read().strip()
            iteration = int(metastring)
        assert iteration>0 , 'error parsing metadata file {}'.format(
            tracker_filename)
    # Checkpoint.
    checkpoint_name = get_checkpoint_name(args.work_dir, iteration, best)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location='cpu')

    # Iterations.
    iteration = sd['iteration']

    # Model.
    model.load_state_dict(sd['model'])

    # Optimizer.
    if optimizer is not None:
        optimizer.load_state_dict(sd['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(sd['lr_scheduler'])

    # rng states.
    random.setstate(sd['random_rng_state'])
    np.random.set_state(sd['np_rng_state'])
    torch.set_rng_state(sd['torch_rng_state'])
    torch.cuda.set_rng_state(sd['cuda_rng_state'])
    mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])


    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
        
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
    if args.optim.lower() == 'sgd':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            # optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
            optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                momentum=args.mom)
    elif args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            # optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
            optimizer = optim.Adam(dense_params, lr=args.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    
    return optimizer

def get_learning_rate_scheduler(optimizer, args):
    if args.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
        # if args.sample_softmax > 0:
        #     scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
        #         args.max_step, eta_min=args.eta_min) # should use eta_min arg
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
        # if args.sample_softmax > 0:
        #     scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
        #         factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
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


    if args.fp16:
        # model = model.half()
        model, optimizer = amp.initialize(model, optimizer,
                                      opt_level="O2",
                                      loss_scale='dynamic'
                                      )
        i = torch.cuda.current_device()

        # model = torchDDP(model, device_ids=[i], output_device=i,
        #                   process_group=mpu.get_data_parallel_group())
        model = apexDDP(model)
    else:
        if args.gpu0_bsz >= 0:
            model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                            model, dim=1).to(model.device)
        else:
            model = nn.DataParallel(model, dim=1).to(model.device)
    # if args.multi_gpu:
    #     model = model.to(device)
    #     if args.gpu0_bsz >= 0:
    #         para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
    #                                         model, dim=1).to(device)
    #     else:
    #         para_model = nn.DataParallel(model, dim=1).to(device)
    # else:
    #     para_model = model.to(device)
    
        
    # if args.cuda and args.fp16:
    #     # If args.dynamic_loss_scale is False, static_loss_scale will be used.
    #     # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
    #     optimizer = FP16_Optimizer(optimizer,
    #                             static_loss_scale = args.static_loss_scale,
    #                             dynamic_loss_scale = args.dynamic_loss_scale,
    #                             dynamic_loss_args = {'init_scale': 2 ** 16})
    args.iteration = 0
    if args.restart:
        if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
            with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
                opt_state_dict = torch.load(f)
                if 'loss_scaler' in opt_state_dict.keys() and not args.fp16:
                    opt_state_dict = opt_state_dict['optimizer_state_dictb']
                optimizer.load_state_dict(opt_state_dict)
        else:
            print('Optimizer was not saved. Start from scratch.')

    return model, optimizer, scheduler

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))

# def save_checkpoint(model, optimizer, path, epoch):
#     torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
#     torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))
