import os
from collections import Counter, OrderedDict

import torch
import nltk

class Vocab(object):
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file

    def tokenize(self, line, add_eos=False, add_double_eos=False, add_sent_eos=False,char_level=False):
        line = line.strip()
        if char_level:
            line = ' '.join([str(ord(c)) for c in line])
        # convert to lower case
        if self.lower_case:
            line = line.lower()
        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)
        if add_sent_eos:
            symbols = symbols + ['<sent_eos>']
        if add_double_eos: # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False,sega=False,sent_eos=False,char_level=False):
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                if sega:
                    nltk_sents = nltk.tokenize.sent_tokenize(line)
                    for sent in nltk_sents:
                        symbols = self.tokenize(sent, add_eos=add_eos,add_sent_eos=sent_eos,char_level=char_level)
                        self.counter.update(symbols)
                else:
                    symbols = self.tokenize(line, add_eos=add_eos,add_sent_eos=sent_eos,char_level=char_level)
                    self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<UNK>']

    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq: break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
            add_double_eos=False,add_sent_eos=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos,
                    add_double_eos=add_double_eos, add_sent_eos=add_sent_eos)
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)


class SegaVocab(Vocab):
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
            add_double_eos=False, add_sent_eos=False, char_level=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        p = []
        s = []
        t = []
        index_p = 0
        index_s = 0
        index_t = 0
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                if line.strip() == '':
                    continue
                sents = nltk.tokenize.sent_tokenize(line)
                symbols = []
                para_pos = []
                sent_pos = []
                token_pos = []
                for i,sent in enumerate(sents):
                    if i == len(sents)-1:
                        sent_symbol = self.tokenize(sent,add_eos=add_eos,add_double_eos=add_double_eos,
                                                    add_sent_eos=add_sent_eos,char_level=char_level)
                    else:
                        sent_symbol = self.tokenize(sent,add_sent_eos=add_sent_eos,char_level=char_level)
                    symbols.extend(sent_symbol)
                    para_pos.extend([index_p]*len(sent_symbol))
                    sent_pos.extend([index_s]*len(sent_symbol))
                    token_pos.extend(range(index_t,index_t+len(sent_symbol)))
                    index_s+=1
                    index_t+=len(sent_symbol)
                index_p+=1
                # symbols = self.tokenize(line, add_eos=add_eos,
                #     add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))
                p.append(torch.LongTensor(para_pos))
                s.append(torch.LongTensor(sent_pos))
                t.append(torch.LongTensor(token_pos))

        if ordered:
            encoded = torch.cat(encoded)
            p = torch.cat(p)
            s = torch.cat(s)
            t = torch.cat(t)

        return (encoded,p,s,t)
