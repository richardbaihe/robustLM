import functools
import os, shutil, time

import numpy as np
import mpu
import random
import torch
import torch.nn as nn
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from apex.parallel import DistributedDataParallel as apexDDP

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

def get_checkpoint_name(checkpoints_path, iteration, best=False, mp_rank=None):
    if best:
        d = 'best'
    else:
        d = 'iter_{:07d}'.format(iteration)
    return os.path.join(checkpoints_path, d,
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
    while isinstance(model, (torchDDP, apexDDP, FP16_Module)):
        model = model.module
    # if isinstance(model, torchDDP):
    #     model = model.module
    if args.rank == 0:
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

def load_checkpoint(model, optimizer, lr_scheduler, args, best=False, checkpoint_name=""):
    """Load a model checkpoint."""
    while isinstance(model, (torchDDP, apexDDP, FP16_Module)):
        model = model.module
    iteration = 0
    if not best and not checkpoint_name:
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
    if not checkpoint_name:
        # Checkpoint.
        checkpoint_name = get_checkpoint_name(args.work_dir, iteration, best)
    if args.rank == 0:
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


    torch.distributed.barrier()
    if args.rank == 0:
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
