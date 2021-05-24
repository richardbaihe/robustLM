import os, sys
import glob

from collections import Counter, OrderedDict
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
        self.sega =  isinstance(data, tuple)
        self.device = device
        if self.sega:
            data,p,s,t = data
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
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]
        pst = tuple()
        if self.sega:
            pst = (self.p[beg_idx:end_idx], self.s[beg_idx:end_idx],self.t[beg_idx:end_idx])
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
            self.vocab.count_file(os.path.join(path, 'train.txt'),sega=sega,sent_eos=sent_eos)
            self.vocab.count_file(os.path.join(path, 'valid.txt'),sega=sega,sent_eos=sent_eos)
            self.vocab.count_file(os.path.join(path, 'test.txt'),sega=sega,sent_eos=sent_eos)
        elif self.dataset == 'enwik8':
            self.vocab.count_file(os.path.join(path, 'train.txt.raw'),sega=sega,sent_eos=sent_eos,char_level=True)
            self.vocab.count_file(os.path.join(path, 'valid.txt.raw'), sega=sega, sent_eos=sent_eos, char_level=True)
            self.vocab.count_file(os.path.join(path, 'test.txt.raw'), sega=sega, sent_eos=sent_eos, char_level=True)
        elif self.dataset == 'wt103':
            self.vocab.count_file(os.path.join(path, 'train.txt'),sega=sega,sent_eos=sent_eos)

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
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt.raw'), ordered=True, add_eos=False, char_level=True)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == 'lm1b':
                kwargs['shuffle'] = True
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, *args, **kwargs)
        elif split == 'train_cl':
            data_iter = LMOrderedIterator(self.train_cl, *args, **kwargs)
        return data_iter

def get_lm_corpus(datadir, dataset,sega=False,sent_eos=False, mix_corpus=False):
    target_name = 'cache.pt'
    if sega:
        target_name = 'sega_'+target_name
    if sent_eos:
        target_name = 'eos_'+target_name
    fn = os.path.join(datadir, target_name)
    # if os.path.exists(fn):
    #     print('Loading cached dataset...')
    #     corpus = torch.load(fn)
    # else:
    print('Producing dataset {}...'.format(dataset))
    kwargs = {}
    if dataset in ['wt103', 'wt2']:
        kwargs['special'] = ['<eos>','<sent_eos>']
        kwargs['lower_case'] = False
    elif dataset == 'ptb':
        kwargs['special'] = ['<eos>']
        kwargs['lower_case'] = True
    elif dataset == 'lm1b':
        kwargs['special'] = []
        kwargs['lower_case'] = False
        kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
    elif dataset in ['enwik8', 'text8']:
        pass
    kwargs['sega'] = sega
    kwargs['sent_eos'] = sent_eos
    if mix_corpus:
        corpus = MixCorpus(datadir, dataset, **kwargs)
    else:
        corpus = Corpus(datadir, dataset, **kwargs)
        # torch.save(corpus, fn)

    return corpus


class MixLMOrderedIterator(LMOrderedIterator):
    def __init__(self, data, data_cl, bsz, bptt, device='cpu', ext_len=None, cl_portion=0):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.sega =  isinstance(data, tuple)
        self.device = device
        self.cl_portion = cl_portion*100
        if self.sega:
            data,p,s,t = data
        assert data.size(0) == data_cl.size(0)
        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)
        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)
        data_cl = data_cl.narrow(0,0,self.n_step*bsz)
        self.data_cl = data_cl.view(bsz, -1).t().contiguous().to(device)
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
        random_number = np.random.randint(1,101)
        if random_number<=self.cl_portion:
            cl = True
        else:
            cl = False
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)
        if cl:
            data = self.data_cl[beg_idx:end_idx]
            target = self.data_cl[i+1:i+1+seq_len]
        else:
            data = self.data[beg_idx:end_idx]
            target = self.data[i+1:i+1+seq_len]
        pst = tuple()
        if self.sega:
            pst = (self.p[beg_idx:end_idx], self.s[beg_idx:end_idx],self.t[beg_idx:end_idx])
        return data, target, seq_len, pst, cl

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


class MixCorpus(object):
    def __init__(self, path, dataset, sega, sent_eos, cl_vocab=False, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)
        self.add_sent_eos = sent_eos

        self.vocab.count_file(os.path.join(path, 'train.txt'),sega=sega,sent_eos=sent_eos)
        self.vocab.count_cl_file(os.path.join(path, 'cl-train.txt'),sega=sega,sent_eos=sent_eos)
        self.vocab.build_vocab()

        self.train_cl = self.vocab.encode_file( 
        os.path.join(path, 'cl-train.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
        self.train = self.vocab.encode_file( 
            os.path.join(path, 'train.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
        self.valid = self.vocab.encode_file(
            os.path.join(path, 'valid.txt'), ordered=True, add_sent_eos=self.add_sent_eos)
        self.test = self.vocab.encode_file(
            os.path.join(path, 'test.txt'), ordered=True, add_sent_eos=self.add_sent_eos)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            data_iter = MixLMOrderedIterator(self.train,self.train_cl, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            data_iter = LMOrderedIterator(data, *args, **kwargs)
        return data_iter

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
