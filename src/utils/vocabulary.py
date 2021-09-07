import os
from collections import Counter, OrderedDict, defaultdict

import torch
import nltk
from nltk.corpus import wordnet as wn

class Vocab(object):
    def __init__(self, special=[], min_freq=1, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file
        self.cl_root_tokens = []
        self.cl_leaf_tokens = []
        self.word2class = {}
        self.class2words = defaultdict(list)
        self.word2class_dict = defaultdict(dict)

    def tokenize(self, line, add_eos=False, add_double_eos=False, add_sent_eos=False, char_level=False):
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
        if add_double_eos:  # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False, sega=False, sent_eos=False, char_level=False):
        if verbose:
            print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                if sega:
                    nltk_sents = nltk.tokenize.sent_tokenize(line)
                    for sent in nltk_sents:
                        symbols = self.tokenize(
                            sent, add_eos=add_eos, add_sent_eos=sent_eos, char_level=char_level)
                        self.counter.update(symbols)
                else:
                    symbols = self.tokenize(
                        line, add_eos=add_eos, add_sent_eos=sent_eos, char_level=char_level)
                    self.counter.update(symbols)

    def count_cl_file(self, path, verbose=False, add_eos=False, sega=False, sent_eos=False, char_level=False):
        if verbose:
            print('counting cl file {} ...'.format(path))
        if not os.path.exists(path):
            print("found no cl files to count")
            return
        temp_counter = Counter()
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(
                    line, add_eos=add_eos, add_sent_eos=sent_eos, char_level=char_level)
                temp_counter.update(symbols)
        self.cl_root_tokens = list((temp_counter-self.counter).keys())
        self.cl_leaf_tokens = list((self.counter-temp_counter).keys())
        self.counter = self.counter | temp_counter

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
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))
        if self.cl_root_tokens:
            self.cl_root_tokens = [self.get_idx(
                sym) for sym in self.cl_root_tokens]
            self.cl_leaf_tokens = [self.get_idx(
                sym) for sym in self.cl_leaf_tokens]

    def build_vocab_hypernym_last(self):
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
            hypernym_tokens = self.cl_root_tokens
            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                if sym in hypernym_tokens:
                    continue
                self.add_symbol(sym)
            for h_token in hypernym_tokens:
                self.add_symbol(h_token)
            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))
        if self.cl_root_tokens:
            self.cl_root_tokens = [self.get_idx(
                sym) for sym in self.cl_root_tokens]
            self.cl_leaf_tokens = [self.get_idx(
                sym) for sym in self.cl_leaf_tokens]


    def build_vocab_with_cl_order(self):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        for sym in self.cl_root_tokens:
            self.add_symbol(sym)

        for sym in self.special:
            self.add_special(sym)

        add_leaf_flag = True
        for sym, cnt in self.counter.most_common(self.max_size):
            if cnt < self.min_freq:
                break
            if add_leaf_flag and len(self.idx2sym) == 20000:
                for _sym in self.cl_leaf_tokens:
                    self.add_symbol(_sym)
                add_leaf_flag = False
            if add_leaf_flag and sym in self.cl_leaf_tokens:
                continue
            self.add_symbol(sym)

        print('final vocab size {} from {} unique tokens'.format(
            len(self), len(self.counter)))
        if self.cl_root_tokens:
            self.cl_root_tokens = [self.get_idx(
                sym) for sym in self.cl_root_tokens]
            self.cl_leaf_tokens = [self.get_idx(
                sym) for sym in self.cl_leaf_tokens]

    def get_wn_replaced_dict(self, synset_layer=5, ignore_freqency_threshold=6000, replaced_with_new_symbol=True,
                             min_tokens_per_hypernym=0):
        
        word2class = {}
        class2words = defaultdict(list)
        for k, cnt in self.counter.most_common(self.max_size):
            if cnt >= ignore_freqency_threshold:
                continue
            if cnt < self.min_freq:
                break
            continue_for_k = True
            for synset in wn.synsets(k):
                paths = synset.hypernym_paths()
                for path in paths:
                    if len(path) < synset_layer+1:
                        continue
                    else:
                        hypernym_name = path[synset_layer].name()
                        if '.n.' not in hypernym_name:
                            continue
                        if not replaced_with_new_symbol:
                            hypernym_name = hypernym_name.split('.')[0].split('')
                        class2words[path[synset_layer].name()].append(
                            k)
                        word2class[k] = path[synset_layer].name()
                        # self.counter.update([path[synset_layer].name()]*cnt)
                        self.counter.update([path[synset_layer].name()])
                        continue_for_k = False
                        break
                if not continue_for_k:
                    break
        for k, v in class2words.items():
            if len(v) >= min_tokens_per_hypernym:
                self.class2words[k].extend(v)
                for token in v:
                    self.word2class[token] = k
            else:
                self.counter[k]=0
        for k, v in self.class2words.items():
            self.cl_root_tokens.append(k)
            self.cl_leaf_tokens.extend(v)
        self.cl_leaf_tokens = list(set(self.cl_leaf_tokens))
        # self.vocab.cl_root_tokens = list(self.vocab.class2words.keys())
        # self.vocab.cl_leaf_tokens = list(self.vocab.word2class.keys())

    def get_wn_replaced_dict_list(self, min_synset_layer=4, max_synset_layer=5, ignore_freqency_threshold=6000, replaced_with_new_symbol=True):
        for k, cnt in self.counter.most_common(self.max_size):
            if cnt >= ignore_freqency_threshold:
                continue
            if cnt < self.min_freq:
                break
            continue_for_k = True
            for synset in wn.synsets(k):
                paths = synset.hypernym_paths()
                for path in paths:
                    if len(path) < max_synset_layer+1:
                        continue
                    else:
                        hypernym_name = path[max_synset_layer].name()
                        if '.n.' not in hypernym_name:
                            continue
                        for synset_layer in range(min_synset_layer, max_synset_layer+1):
                            hypernym_name = path[synset_layer].name()
                            self.class2words[hypernym_name].append(
                                k)
                            self.word2class_dict[synset_layer][k] = hypernym_name
                            self.counter.update([hypernym_name])
                        continue_for_k = False
                        break
                if not continue_for_k:
                    break
        for k, v in self.class2words.items():
            self.cl_root_tokens.append(k)
            self.cl_leaf_tokens.extend(v)
        self.cl_leaf_tokens = list(set(self.cl_leaf_tokens))
        # self.vocab.cl_root_tokens = list(self.vocab.class2words.keys())
        # self.vocab.cl_leaf_tokens = list(self.vocab.word2class.keys())


    def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
                    add_double_eos=False, add_sent_eos=False):
        if verbose:
            print('encoding file {} ...'.format(path))
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

    def encode_file_plus(self, path, ordered=False, verbose=False, add_eos=True,
                         add_double_eos=False, add_sent_eos=False):
        if verbose:
            print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        encoded_cl = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos,
                                        add_double_eos=add_double_eos, add_sent_eos=add_sent_eos)
                cl_symbols = [self.word2class[x]
                              if x in self.word2class else x for x in symbols]
                encoded.append(self.convert_to_tensor(symbols))
                encoded_cl.append(self.convert_to_tensor(cl_symbols))
        if ordered:
            encoded = torch.cat(encoded)
            encoded_cl = torch.cat(encoded_cl)

        return encoded, encoded_cl

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose:
            print('encoding {} sents ...'.format(len(sents)))
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
        super().__init__(special, min_freq, max_size, lower_case,
                 delimiter, vocab_file)

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
                    add_double_eos=False, add_sent_eos=False, char_level=False):
        if verbose:
            print('encoding file {} ...'.format(path))
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
                for i, sent in enumerate(sents):
                    if i == len(sents)-1:
                        sent_symbol = self.tokenize(sent, add_eos=add_eos, add_double_eos=add_double_eos,
                                                    add_sent_eos=add_sent_eos, char_level=char_level)
                    else:
                        sent_symbol = self.tokenize(
                            sent, add_sent_eos=add_sent_eos, char_level=char_level)
                    symbols.extend(sent_symbol)
                    para_pos.extend([index_p]*len(sent_symbol))
                    sent_pos.extend([index_s]*len(sent_symbol))
                    token_pos.extend(range(index_t, index_t+len(sent_symbol)))
                    index_s += 1
                    index_t += len(sent_symbol)
                index_p += 1
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

        return (encoded, p, s, t)

    def encode_file_plus(self, path, ordered=False, verbose=False, add_eos=True,
                    add_double_eos=False, add_sent_eos=False, char_level=False):
        if verbose:
            print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        encoded_cl = []
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
                for i, sent in enumerate(sents):
                    if i == len(sents)-1:
                        sent_symbol = self.tokenize(sent, add_eos=add_eos, add_double_eos=add_double_eos,
                                                    add_sent_eos=add_sent_eos, char_level=char_level)
                    else:
                        sent_symbol = self.tokenize(
                            sent, add_sent_eos=add_sent_eos, char_level=char_level)
                    symbols.extend(sent_symbol)

                    para_pos.extend([index_p]*len(sent_symbol))
                    sent_pos.extend([index_s]*len(sent_symbol))
                    token_pos.extend(range(index_t, index_t+len(sent_symbol)))
                    index_s += 1
                    index_t += len(sent_symbol)
                cl_symbols = [self.word2class[x]
                              if x in self.word2class else x for x in symbols]
                index_p += 1
                # symbols = self.tokenize(line, add_eos=add_eos,
                #     add_double_eos=add_double_eos)
                encoded_cl.append(self.convert_to_tensor(cl_symbols))
                encoded.append(self.convert_to_tensor(symbols))
                p.append(torch.LongTensor(para_pos))
                s.append(torch.LongTensor(sent_pos))
                t.append(torch.LongTensor(token_pos))

        if ordered:
            encoded = torch.cat(encoded)
            encoded_cl = torch.cat(encoded_cl)
            p = torch.cat(p)
            s = torch.cat(s)
            t = torch.cat(t)

        return (encoded, p, s, t), encoded_cl
