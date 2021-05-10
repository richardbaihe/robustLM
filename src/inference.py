# coding: utf-8
import argparse
import time
import math
import os, sys

import torch
from nltk.corpus import wordnet as wn
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM, SegaMemTransformerLM,\
    Sega_wo_p_MemTransformerLM,Sega_wo_s_MemTransformerLM
from utils.exp_utils import get_logger

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=5,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
parser.add_argument('--sega', action='store_true',
                    help='sega or not')
parser.add_argument('--sparse_mode', type=str, default='none',
                    help='spare mode for longformer')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'
args.sent_eos=False
if 'eos' in args.sparse_mode:
    args.sent_eos=True
args.compressed_mem = False
if 'compress' in args.sparse_mode:
    args.compressed_mem=True
device = torch.device("cuda" if args.cuda else "cpu")

args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset, sega=args.sega, sent_eos=args.sent_eos)
ntokens = len(corpus.vocab)

va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
model.backward_compatible()
model = model.to(device)

logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
       args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True

###############################################################################
# Evaluation code
###############################################################################
def get_top_50_words_and_props(model, data, target, *mems):
    if not mems: mems = model.init_mems()

    tgt_len = target.size(0)
    hidden, new_mems = model._forward(data, mems=mems)

    pred_hid = hidden[-tgt_len:]
    words, probs = model.crit.get_top_50_words_and_props(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))

    if new_mems is None:
        return [words, probs]
    else:
        return [words, probs] + new_mems

def get_all_props(model, data, target, *mems):
    if not mems: mems = model.init_mems()

    tgt_len = target.size(0)
    hidden, new_mems = model._forward(data, mems=mems)

    pred_hid = hidden[-tgt_len:]
    probs = model.crit.get_all_props(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))

    if new_mems is None:
        return [probs]
    else:
        return [probs] + new_mems
def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    all_words, all_probs, all_targets = [],[],[]
    start_time = time.time()
    with torch.no_grad():
        mems = tuple()
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            ret = get_all_props(model,data, target, *mems)
            probs = ret[0]#, ret[1]
            # all_words.extend(words.tolist())
            all_probs.extend(probs.tolist())
            all_targets.extend(target.view(-1).tolist())
            mems = ret[1:]
        total_time = time.time() - start_time
    logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))
    return all_probs, all_targets#all_words, 

# def sega_evaluate(eval_iter):
#     # Turn on evaluation mode which disables dropout.
#     model.eval()
#     total_len, total_loss = 0, 0.
#     start_time = time.time()
#     with torch.no_grad():
#         mems = tuple()
#         mems_pst = tuple()
#         for idx, (data, target, seq_len,pst) in enumerate(eval_iter):
#             ret = model(data, target, mems, pst, mems_pst)
#             loss, mems = ret[0], ret[1:]
#             new_pst = []
#             if not mems_pst:
#                 mems_pst = pst
#             else:
#                 for t, m_t in zip(pst, mems_pst):
#                     cat = torch.cat([m_t, t], dim=0)
#                     new_pst.append(cat[max(0,len(cat)-model.mem_len):])
#                 mems_pst = tuple(new_pst)
#             if args.sent_eos:
#                 sent_eos_pos = target==corpus.vocab.sent_eos_idx
#                 loss = loss * (1-sent_eos_pos.contiguous().type(loss.type()))
#                 total_loss += loss.sum().float().item()/target.shape[-1]
#                 total_len += (seq_len*target.shape[-1]-torch.sum(sent_eos_pos)).float()/target.shape[-1]
#             else:
#                 loss = loss.mean()
#                 total_loss += seq_len * loss.float().item()
#                 total_len += seq_len

#         total_time = time.time() - start_time
#     logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
#             total_time, 1000 * total_time / (idx+1)))
#     return total_loss / total_len


all_probs, all_targets = evaluate(va_iter)
print('done')

