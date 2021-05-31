import argparse
import os
from tensorboard.compat import tf
import time

def add_model_config_args(parser):
    group = parser.add_argument_group('model', 'model configurations')
    group.add_argument('--n_layer', type=int, default=12,
                        help='number of total layers')
    group.add_argument('--n_head', type=int, default=10,
                        help='number of heads')
    group.add_argument('--d_head', type=int, default=50,
                        help='head dimension')
    group.add_argument('--d_embed', type=int, default=-1,
                        help='embedding dimension')
    group.add_argument('--d_model', type=int, default=500,
                        help='model dimension')
    group.add_argument('--d_inner', type=int, default=1000,
                        help='inner dimension in FF')
    group.add_argument('--dropout', type=float, default=0.1,
                        help='global dropout rate')
    group.add_argument('--dropatt', type=float, default=0.0,
                        help='attention probability dropout rate')
    group.add_argument('--init', default='normal', type=str,
                        help='parameter initializer to use.')
    group.add_argument('--emb_init', default='normal', type=str,
                        help='parameter initializer to use.')
    group.add_argument('--init_range', type=float, default=0.1,
                        help='parameters initialized by U(-init_range, init_range)')
    group.add_argument('--emb_init_range', type=float, default=0.01,
                        help='parameters initialized by U(-init_range, init_range)')
    group.add_argument('--init_std', type=float, default=0.02,
                        help='parameters initialized by N(0, init_std)')
    group.add_argument('--proj_init_std', type=float, default=0.01,
                        help='parameters initialized by N(0, init_std)')
    group.add_argument('--not_tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')
    group.add_argument('--adaptive', action='store_true',
                        help='use adaptive softmax')
    group.add_argument('--div_val', type=int, default=1,
                        help='divident value for adapative input and softmax')
    group.add_argument('--pre_lnorm', action='store_true',
                        help='apply LayerNorm to the input instead of the output')
    return parser

def add_training_config_args(parser):
    group = parser.add_argument_group('training', 'training configurations')
    group.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    group.add_argument('--varlen', action='store_true',
                        help='use variable length')
    group.add_argument('--same_length', action='store_true',
                        help='use the same attn length for all tokens')
    group.add_argument('--log_interval', type=int, default=200,
                        help='report interval')
    group.add_argument('--eval_interval', type=int, default=4000,
                        help='evaluation interval')
    group.add_argument('--save_interval', type=int, default=5000,
                        help='evaluation interval')
    group.add_argument('--load_lastest', action='store_true',
                        help='continue training from lastest checkpoint')                
    group.add_argument('--optim', default='adam', type=str,
                        choices=['adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')
    group.add_argument('--lr', type=float, default=0.00025,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    group.add_argument('--mom', type=float, default=0.0,
                        help='momentum for sgd')
    group.add_argument('--scheduler', default='cosine', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                        help='lr scheduler to use.')
    group.add_argument('--warmup_step', type=int, default=0,
                        help='upper epoch limit')
    group.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay factor when ReduceLROnPlateau is used')
    group.add_argument('--lr_min', type=float, default=0.0,
                        help='minimum learning rate during annealing')
    group.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    group.add_argument('--clip_nonemb', action='store_true',
                        help='only clip the gradient of non-embedding params')
    group.add_argument('--max_step', type=int, default=100000,
                        help='upper epoch limit')
    group.add_argument('--batch_size', type=int, default=60,
                        help='batch size')
    group.add_argument('--gpu0_bsz', type=int, default=-1,
                        help='batch size on gpu 0')
    group.add_argument('--batch_chunk', type=int, default=1,
                        help='split batch into chunks to save memory')
    group.add_argument('--tgt_len', type=int, default=70,
                        help='number of tokens to predict')
    group.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    group.add_argument('--mem_len', type=int, default=0,
                        help='length of the retained previous heads')
    group.add_argument('--attn_type', type=int, default=0,
                        help='attention type. 0 for ours, 1 for Shaw et al,'
                        '2 for Vaswani et al, 3 for Al Rfou et al.')
    group.add_argument('--clamp_len', type=int, default=-1,
                        help='use the same pos embeddings after clamp_len')
    group.add_argument('--eta_min', type=float, default=0.0,
                        help='min learning rate for cosine scheduler')
    group.add_argument('--restart', action='store_true',
                        help='restart training from the saved checkpoint')
    group.add_argument('--restart_dir', type=str, default='',
                        help='restart dir')
    group.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    group.add_argument('--patience', type=int, default=0,
                        help='patience')
    group.add_argument('--finetune_v2', action='store_true',
                        help='finetune v2')
    group.add_argument('--finetune_v3', action='store_true',
                        help='finetune v3')
    parser.add_argument('--sega', action='store_true',
                        help='sega or not')
    
    # distributed training args
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                       'training. One of [gloo, nccl]')
    group.add_argument('--DDP-impl', default='local',
                       help='which DistributedDataParallel implementation '
                       'to use. One of [local, torch]')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='size of the model parallel.')
    # autoresume
    group.add_argument('--adlr-autoresume', action='store_true',
                       help='enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='intervals over which check for autoresume'
                       'termination signal')

    return parser

def add_evaluation_config_args(parser):
    group = parser.add_argument_group('evaluation', 'evaluation configurations')
    group.add_argument('--eval_tgt_len', type=int, default=50,
                        help='number of tokens to predict for evaluation')
    group.add_argument('--max_eval_steps', type=int, default=-1,
                        help='max eval steps')
    group.add_argument('--sample_softmax', type=int, default=-1,
                        help='number of samples in sampled softmax')
    return parser

def add_data_config_args(parser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument('--data', type=str, default=os.getenv('PT_DATA_DIR', 'data'),
                        help='location of the data corpus')
    group.add_argument('--dataset', type=str, default='wt103',
                        choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    group.add_argument('--do_train', action='store_true',
                        help='train model')
    group.add_argument('--do_test', action='store_true',
                        help='test model')
    return parser

def add_device_config_args(parser):
    group = parser.add_argument_group('device', 'device configurations')
    group.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    group.add_argument('--multi_gpu', action='store_true',
                        help='use multiple GPU')
    group.add_argument('--fp16', action='store_true',
                        help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')                        
    group.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can '
                        'improve fp16 convergence.')
    group.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument'
                        ' supersedes --static-loss-scale.')
    group.add_argument('--pt', action='store_true',
                        help='phillytool or local')
    group.add_argument('--work_dir', default='LM-TFM', type=str,
                        help='experiment directory.')
    group.add_argument('--job_name', default='example', type=str,
                        help='experimetn name')
    group.add_argument('--wandb_offline', action='store_true',
                        help='debugging offline')
    return parser

def add_classLM_config_args(parser):
    group = parser.add_argument_group('classLM', 'class-based LM configurations')
    group.add_argument('--cl_steps', type=int, default=0,
                        help='initial steps for classLM training')
    group.add_argument('--cl_annealing', type=float, default=0,
                        help='initial cl portion for mix training')
    group.add_argument('--mix_corpus', action='store_true',
                        help='process two corpus and mix them for class LM training')
    return parser

def get_args():
    # parser.add_argument('--window_size', type=int, default=0,
    #                     help='local attention window size')
    # parser.add_argument('--sparse_mode', type=str, default='none',
    #                     help='spare mode for longformer')

    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

    parser = add_data_config_args(parser)
    parser = add_device_config_args(parser)
    parser = add_model_config_args(parser)
    parser = add_training_config_args(parser)
    parser = add_evaluation_config_args(parser)
    parser = add_classLM_config_args(parser)

    args = parser.parse_args()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    args.tied = not args.not_tied
    args.sent_eos=False
    # if 'eos' in args.sparse_mode:
    #     args.sent_eos=True
    if args.d_embed < 0:
        args.d_embed = args.d_model

    assert args.ext_len >= 0, 'extended context length must be non-negative'
    assert args.batch_size % args.batch_chunk == 0

    if args.pt:
        # this hack is required to enable `pt monitor` *while the job is running*.
        delattr(tf.io.gfile.LocalFileSystem, 'append')
        args.work_dir = os.environ.get('PT_OUTPUT_DIR', '.')
    else:
        # args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
        args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d'))

    return args
