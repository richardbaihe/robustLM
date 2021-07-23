import os
import sys
import glob
import mpu
from collections import Counter, OrderedDict, defaultdict
import numpy as np
import torch

from utils.vocabulary import Vocab, SegaVocab


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.sega = isinstance(data, tuple)
        self.device = device
        if self.sega:
            data, p, s, t = data
        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)
        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)
        if self.sega:
            p = p.narrow(0, 0, self.n_step * bsz)
            s = s.narrow(0, 0, self.n_step * bsz)
            t = t.narrow(0, 0, self.n_step * bsz)
            self.p = p.view(bsz, -1).t().contiguous().to(device)
            self.s = s.view(bsz, -1).t().contiguous().to(device)
            self.t = t.view(bsz, -1).t().contiguous().to(device)
        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]
        pst = tuple()
        if self.sega:
            pst = (self.p[beg_idx:end_idx],
                   self.s[beg_idx:end_idx], self.t[beg_idx:end_idx])
        return data, target, seq_len, pst

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            if self.sega:
                data, target, seq_len, pst = self.get_batch(i, bptt)
                i += seq_len
                yield data, target, seq_len, pst
            else:
                data, target, seq_len = self.get_batch(i, bptt)
                i += seq_len
                yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
            else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                            streams[i][:n_new]
                        target[n_filled:n_filled+n_new, i] = \
                            streams[i][1:n_new+1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None,
                 shuffle=False):

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


class Corpus(object):
    def __init__(self, path, dataset, sega, sent_eos, *args, **kwargs):
        self.dataset = dataset
        if sega:
            self.vocab = SegaVocab(*args, **kwargs)
        else:
            self.vocab = Vocab(*args, **kwargs)
        self.add_sent_eos = sent_eos

        if self.dataset in ['ptb', 'wt2', 'text8']:
            self.vocab.count_file(os.path.join(
                path, 'train.txt'), sega=sega, sent_eos=sent_eos)
            self.vocab.count_file(os.path.join(
                path, 'valid.txt'), sega=sega, sent_eos=sent_eos)
            self.vocab.count_file(os.path.join(
                path, 'test.txt'), sega=sega, sent_eos=sent_eos)
        elif self.dataset == 'enwik8':
            self.vocab.count_file(os.path.join(
                path, 'train.txt.raw'), sega=sega, sent_eos=sent_eos, char_level=True)
            self.vocab.count_file(os.path.join(
                path, 'valid.txt.raw'), sega=sega, sent_eos=sent_eos, char_level=True)
            self.vocab.count_file(os.path.join(
                path, 'test.txt.raw'), sega=sega, sent_eos=sent_eos, char_level=True)
        elif self.dataset == 'wt103':
            self.vocab.count_file(os.path.join(
                path, 'train.txt'), sega=sega, sent_eos=sent_eos)

        elif self.dataset == 'lm1b':
            train_path_pattern = os.path.join(
                path, '1-billion-word-language-modeling-benchmark-r13output',
                'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called

        self.vocab.build_vocab()

        if self.dataset in ['ptb', 'wt2', 'wt103']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
        elif self.dataset in ['enwik8', 'text8']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt.raw'), ordered=True, add_eos=False, char_level=True)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt.raw'), ordered=True, add_eos=False, char_level=True)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt.raw'), ordered=True, add_eos=False, char_level=True)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
            self.test = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == 'lm1b':
                kwargs['shuffle'] = True
                data_iter = LMMultiFileIterator(
                    self.train, self.vocab, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, *args, **kwargs)
        elif split == 'train_cl':
            data_iter = LMOrderedIterator(self.train_cl, *args, **kwargs)
        return data_iter


class MixLMOrderedIterator(LMOrderedIterator):
    def __init__(self, data, data_cl, bsz, bptt, device='cpu', ext_len=None, cl_portion=0, rank=-1, world_size=1):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        if rank == -1:
            assert False, 'should not be here'
            rank = torch.distributed.get_rank()
        self.rank = rank
        self.world_size = world_size
        self.bsz = bsz*self.world_size
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.sega = isinstance(data, tuple)
        self.device = device
        self.cl_portion = cl_portion*100
        if self.sega:
            data, p, s, t = data
        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // self.bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * self.bsz)
        # Evenly divide the data across the bsz batches.
        self.data = data.view(self.bsz, -1).t().contiguous().to(device)
        data_cl = data_cl.narrow(0, 0, self.n_step*self.bsz)
        self.data_cl = data_cl.view(self.bsz, -1).t().contiguous().to(device)
        # if self.sega:
        #     p = p.narrow(0, 0, self.n_step * bsz)
        #     s = s.narrow(0, 0, self.n_step * bsz)
        #     t = t.narrow(0, 0, self.n_step * bsz)
        #     self.p = p.view(bsz, -1).t().contiguous().to(device)
        #     self.s = s.view(bsz, -1).t().contiguous().to(device)
        #     self.t = t.view(bsz, -1).t().contiguous().to(device)
        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i):
        #if bptt is None: bptt = self.bptt
        i = i % self.n_batch
        seq_len = min(self.bptt, self.data.size(0) - 1 - i)
        # while i+seq_len >= self.data.size(0):
        #     i = max(0, i-self.data.size(0))
        i = i*self.bptt
        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)
        start = self.rank*self.bsz//self.world_size
        end = (self.rank+1)*self.bsz//self.world_size
        data = self.data[beg_idx:end_idx, start:end]
        target = self.data[i+1:i+1+seq_len, start:end]

        cl_data = self.data_cl[beg_idx:end_idx, start:end]
        cl_target = self.data_cl[i+1:i+1+seq_len, start:end]
        # pst = tuple()
        # if self.sega:
        #     pst = (self.p[beg_idx:end_idx], self.s[beg_idx:end_idx],self.t[beg_idx:end_idx])
        # return data, target, cl_data, cl_target
        return {'input': data, 'target': target, 'cl_input': cl_data, 'cl_target': cl_target}

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    # def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
    #     max_len = self.bptt + max_deviation * std
    #     i = start
    #     while True:
    #         bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
    #         bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
    #         if self.sega:
    #             data, target, seq_len, pst = self.get_batch(i, bptt)
    #             i += seq_len
    #             yield data, target, seq_len, pst
    #         else:
    #             data, target, seq_len = self.get_batch(i, bptt)
    #             i += seq_len
    #             yield data, target, seq_len
    #         if i >= self.data.size(0) - 2:
    #             break

    def __iter__(self):
        return self.get_fixlen_iter()

    def __len__(self):
        return self.n_batch


class MixCorpus(object):
    def __init__(self, args, *_args, **kwargs):
        path, dataset, sega, sent_eos = args.data, args.dataset, args.sega, args.sent_eos
        self.dataset = dataset
        self.vocab = Vocab(*_args, **kwargs)
        self.add_sent_eos = sent_eos
        self.path = path
        self.vocab.count_file(os.path.join(
            path, 'train.txt'), sega=sega, sent_eos=sent_eos)
        if args.dynamic_wn_layer_start_from>0:
            self.vocab.get_wn_replaced_dict_list(
                min_synset_layer=args.dynamic_wn_layer_start_from, max_synset_layer=args.wn_layer)
            self.vocab.word2class = self.vocab.word2class_dict[args.wn_layer]
        else:
            self.vocab.get_wn_replaced_dict(
                synset_layer=args.wn_layer, ignore_freqency_threshold=args.ignore_freqency_threshold,
                min_tokens_per_hypernym=args.min_tokens_per_hypernym)
        if args.adaptive_class_softmax:
            self.vocab.build_vocab_with_cl_order()
        # elif args.learn_offset or args.vocab_order_hypernym_last:
        #     self.vocab.build_vocab_hypernym_last()
        else:
            self.vocab.build_vocab()
        self.train, self.train_cl = self.vocab.encode_file_plus(
            os.path.join(path, 'train.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
        self.valid, self.valid_cl = self.vocab.encode_file_plus(
            os.path.join(path, 'valid.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
        self.test, self.test_cl = self.vocab.encode_file_plus(
            os.path.join(path, 'test.txt'), ordered=True, add_sent_eos=self.add_sent_eos)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            data = self.train
            data_cl = self.train_cl
        else:
            data = self.valid if split == 'valid' else self.test
            data_cl = self.valid_cl if split == 'valid' else self.test_cl
        data_iter = MixLMOrderedIterator(data, data_cl, *args, **kwargs)
        # if split == 'train':
        #     data_iter = MixLMOrderedIterator(self.train, self.train_cl, *args, **kwargs)
        # elif split in ['valid', 'test']:
        #     data = self.valid if split == 'valid' else self.test
        #     data_iter = LMOrderedIterator(data, *args, **kwargs)
        return data_iter
    
    def rebuild_data_with_wn_layer_n(self, wn_layer):
        self.vocab.word2class = self.vocab.word2class_dict[wn_layer]
        self.train, self.train_cl = self.vocab.encode_file_plus(
            os.path.join(self.path, 'train.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
        self.valid, self.valid_cl = self.vocab.encode_file_plus(
            os.path.join(self.path, 'valid.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
        self.test, self.test_cl = self.vocab.encode_file_plus(
            os.path.join(self.path, 'test.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
