import functools
import os, shutil, time

import numpy as np
import random
import torch
import torch.nn as nn
# from apex import amp
from utils.data_parallel import BalancedDataParallel
# from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
# from apex.parallel import DistributedDataParallel as apexDDP
from mpu.random import get_cuda_rng_tracker
# import mpu
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, args, last_epoch=-1, verbose=False):
        self.T_max = args.max_step
        self.min_lr = args.lr_min
        self.max_lr = args.lr_max
        self.t_mult = args.t_mult
        self.period = args.max_step- args.warmup_step
        self.lr_shrink = args.lr_shrink
        warmup_end_lr = args.lr_max 
        if args.warmup_step > 0:
            # linearly warmup for the first args.warmup_step
            self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_step
        else:
            self.lr_step = 1

        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch<self.args.warmup_step:
            return  self.args.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            curr_updates = self.last_epoch - self.args.warmup_step
            if self.t_mult != 1:
                i = math.floor(math.log(1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult))
                t_i = self.t_mult ** i * self.period
                t_curr = curr_updates - (1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
            else:
                i = math.floor(curr_updates / self.period)
                t_i = self.period
                t_curr = curr_updates - (self.period * i)
            lr_shrink = self.lr_shrink ** i
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))
            return lr

    def _get_closed_form_lr(self):
        if self.last_epoch<self.args.warmup_step:
            return self.args.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            curr_updates = self.last_epoch - self.args.warmup_step
            if self.t_mult != 1:
                i = math.floor(math.log(1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult))
                t_i = self.t_mult ** i * self.period
                t_curr = curr_updates - (1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
            else:
                i = math.floor(curr_updates / self.period)
                t_i = self.period
                t_curr = curr_updates - (self.period * i)
            lr_shrink = self.lr_shrink ** i
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))
            return lr


def upload_model(local_path, remote_path):
    """Saves the model to Google Cloud Storage
    Args:
      args: contains name for saved model.
    """
    if remote_path=='':
        return
    from google.cloud import storage
    scheme = 'gs://'
    bucket_name = "richardbaihe"
    remote_path = '{}/{}/{}/{}'.format(remote_path.split('/')[-2], remote_path.split('/')[-1], local_path.split('/')[-2], local_path.split('/')[-1])
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)
    # blob = bucket.blob(remote_path.replace("model_optim_rng.pt","wandb.id"))
    # blob.upload_from_filename(local_path.replace("model_optim_rng.pt","wandb.id"))

def download_model(local_path, remote_path):
    if remote_path=='':
        return
    from google.cloud import storage
    scheme = 'gs://'
    bucket_name = "richardbaihe"
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(remote_path)
    os.makedirs(os.path.dirname(local_path),exist_ok=True)
    blob.download_to_filename(local_path)
    print('Blob {} downloaded to {}.'.format(
         remote_path,
         local_path))

def get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, torch.nn.LayerNorm):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params

def get_checkpoint_name(checkpoints_path, iteration, ckpt_folder_name=None, mp_rank=None):
    if ckpt_folder_name is None:
        ckpt_folder_name = 'iter_{:07d}'.format(iteration)
    return os.path.join(checkpoints_path, ckpt_folder_name,
                        'model_optim_rng.pt')

def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_checkpoint(iteration, model, optimizer, lr_scheduler, work_dir, ckpt_folder_name):
    """Save a model checkpoint."""
    # Only rank zer0 of the data parallel writes to the disk.
    while isinstance(model, (nn.DataParallel, BalancedDataParallel)):
        model = model.module
    # if isinstance(model, torchDDP):
    #     model = model.module
    if True:
        checkpoint_name = get_checkpoint_name(work_dir, iteration, ckpt_folder_name)

        print('saving checkpoint at iteration {:7d} to {}'.
              format(iteration, checkpoint_name))

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

        ensure_directory_exists(checkpoint_name)
        torch.save(sd, checkpoint_name)
        print('  successfully saved {}'.format(checkpoint_name))
    # if not best:
    #     # Wait so everyone is done (necessary)
    #     torch.distributed.barrier()
    # And update the latest iteration
    if ckpt_folder_name!='best':
        tracker_filename = get_checkpoint_tracker_filename(work_dir)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # if not best:
    #     # Wait so everyone is done (not necessary)
    #     torch.distributed.barrier()
    return checkpoint_name


def load_checkpoint(model, optimizer, lr_scheduler, work_dir, ckpt_folder_name=None, checkpoint_name=""):
    """Load a model checkpoint."""
    while isinstance(model, (nn.DataParallel, BalancedDataParallel)):
        model = model.module
    iteration = 0
    if not ckpt_folder_name and not checkpoint_name:
        # Read the tracker file and set the iteration.
        tracker_filename = get_checkpoint_tracker_filename(work_dir)
        if not os.path.isfile(tracker_filename):
            print('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print('    will not load any checkpoints and will start from '
                        'random')
            return 0
        with open(tracker_filename, 'r') as f:
            metastring = f.read().strip()
            iteration = int(metastring)
        assert iteration>0 , 'error parsing metadata file {}'.format(
            tracker_filename)
    if not checkpoint_name:
        # Checkpoint.
        checkpoint_name = get_checkpoint_name(work_dir, iteration, ckpt_folder_name)
        if not os.path.isfile(checkpoint_name):
            print('WARNING: could not find the metadata file {} '.format(
                checkpoint_name))
            print('    will not load any checkpoints and will start from '
                        'random')
            return 0
    print(' is loading checkpoint {}'.format(
            checkpoint_name))

    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location='cpu')

    # Iterations.
    iteration = sd['iteration']

    # Model.
    if model is not None and 'model' in sd.keys():
        sd['model'] = {key.replace("module.", ""): value for key, value in sd['model'].items()}
        model.load_state_dict(sd['model'])

    # Optimizer.
    if optimizer is not None and 'optimizer' in sd.keys():
        if 'optimizer_state_dict' in sd['optimizer']:
            optimizer.load_state_dict(sd['optimizer']['optimizer_state_dict'])
        else:
            optimizer.load_state_dict(sd['optimizer'])
    if lr_scheduler is not None and 'lr_scheduler' in sd.keys():
        lr_scheduler.load_state_dict(sd['lr_scheduler'])
    # rng states.
    if 'random_rng_state' in sd.keys() and 'np_rng_state' in sd.keys() and 'torch_rng_state' in sd.keys():
        random.setstate(sd['random_rng_state'])
        np.random.set_state(sd['np_rng_state'])
        torch.set_rng_state(sd['torch_rng_state'])
        torch.cuda.set_rng_state(sd['cuda_rng_state'])
    if 'rng_tracker_states' in sd.keys():
        get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])


    # torch.distributed.barrier()
    # if args.rank == 0:
    print('  successfully loaded {}'.format(checkpoint_name))

    return iteration

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
        
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


class Timers:
    """Group of timers."""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '_time', value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0/ normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print_rank_0(string)
