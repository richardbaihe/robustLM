from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA_MAJOR = int(torch.version.cuda.split('.')[0])
CUDA_MINOR = int(torch.version.cuda.split('.')[1])

class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(
                        nn.Parameter(torch.Tensor(d_proj, d_embed))
                    )
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                self.out_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i))
                )

                self.out_layers.append(nn.Linear(d_emb_i, r_idx-l_idx))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target, keep_order=False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            nll = -F.log_softmax(logit, dim=-1) \
                    .gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            #head_logit[:,0] = -float('inf')
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target,
                    dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:,None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] \
                              + tail_logprob_i.gather(1, target_i[:,None]).squeeze(1)

                if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll

    def get_top_50_words_and_props(self, hidden,target, keep_order=False):
        
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''
        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            all_probs = F.softmax(logit,dim=-1)
            probs, words = torch.topk(all_probs,k=50,dim=-1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            #head_logit[:,0] = -float('inf')
            # head softmax
            head_probs = F.softmax(head_logit, dim=1)
            head_argmax = torch.argmax(head_probs,dim=1)
            head_n_words = head_weight.size()[0]-self.n_clusters
            # cutoff_values=[0,1000,50000,100000]
            cutoff_values = [0] + self.cutoffs

            words = torch.zeros(size=(hidden.size()[0],50),dtype=torch.long, device=hidden.device)
            probs = torch.zeros(size=(hidden.size()[0],50), device=hidden.device)
            for i in range(len(cutoff_values) - 1):
                if i == 0 :
                    cluster_i_indices = torch.nonzero(head_argmax<head_n_words).squeeze()
                    
                    cluster_i_probs = head_probs.index_select(0, cluster_i_indices)[:,:-self.n_clusters]
                    cluster_i_probs, cluster_i_words = torch.topk(cluster_i_probs,k=50,dim=-1)
                    probs.index_copy_(0, cluster_i_indices, cluster_i_probs)
                    words.index_copy_(0, cluster_i_indices, cluster_i_words)
                else:
                    cluster_i_indices = torch.nonzero(head_argmax==(head_n_words+i-1)).squeeze()
                    hidden_i = hidden.index_select(0, cluster_i_indices)
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    cluster_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    cluster_i_probs = F.softmax(cluster_logit_i, dim=1)

                    cluster_i_probs, cluster_i_words = torch.topk(cluster_i_probs,k=50,dim=-1)
                    probs.index_copy_(0, cluster_i_indices, cluster_i_probs)
                    words.index_copy_(0, cluster_i_indices, cluster_i_words)

        return words, probs

    def get_all_props(self, hidden,target, keep_order=False):
        
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''
        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            all_probs = F.softmax(logit,dim=-1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            #head_logit[:,0] = -float('inf')
            # head softmax
            head_probs = F.softmax(head_logit, dim=1)
            cutoff_values = [0] + self.cutoffs

            all_probs = torch.zeros(size=(hidden.size()[0],self.out_layers[0].weight.size()[0]),dtype=torch.float, device=hidden.device)

            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                temp_prob = torch.zeros(size=(indices_i.size()[0],all_probs.size()[1]), dtype=torch.float, device=hidden.device)
                if i == 0 :
                    cluster_i_probs = head_probs.index_select(0, indices_i)[:,:-self.n_clusters]
                else:
                    hidden_i = hidden.index_select(0, indices_i)
                    weight_i, bias_i, proj_i = weights[i], biases[i],self.out_projs[i]
                    cluster_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    cluster_i_probs = F.softmax(cluster_logit_i, dim=1)
                temp_prob.index_copy_(1, torch.range(l_idx,r_idx-1, dtype=torch.long, device=hidden.device).squeeze(), cluster_i_probs)
                all_probs.index_copy_(0, indices_i, temp_prob)
        all_probs = all_probs.cpu()
        return all_probs

class ClassedProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 keep_order=False, cl_all_root_index = None,cl_all_leaf_index = None):
        super(ClassedProjectedAdaptiveLogSoftmax, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val


        self.cl_all_root_index = cl_all_root_index
        self.cl_all_leaf_index = cl_all_leaf_index

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
            self.cl_cutoffs_root_index = []
            self.cl_cutoffs_leaf_index = []
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                self.cl_cutoffs_root_index.append([index-l_idx for index in self.cl_all_root_index if index<r_idx and index>=l_idx])
                self.cl_cutoffs_leaf_index.append([index-l_idx for index in self.cl_all_leaf_index if index<r_idx and index>=l_idx])


        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(
                        nn.Parameter(torch.Tensor(d_proj, d_embed))
                    )
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                self.out_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i))
                )

                self.out_layers.append(nn.Linear(d_emb_i, r_idx-l_idx))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target, keep_order=True, predict_root=False, general_words_only=False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
            predict_root: True for class label + general words; False for normal LM
            separate_vocab: True for general words only
        '''

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            if predict_root:
                logit[:,self.cl_all_leaf_index] = -float('inf')
            else:
                logit[:,self.cl_all_root_index] = -float('inf')
            if general_words_only:
                logit[:,self.cl_all_leaf_index+self.cl_all_root_index] = -float('inf')

            nll = -F.log_softmax(logit, dim=-1) \
                    .gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            head_leaf_index, head_root_index = self.cl_cutoffs_leaf_index[0], self.cl_cutoffs_root_index[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            if predict_root:
                if head_leaf_index:
                    head_logit.index_fill_(1,torch.tensor(head_leaf_index,device=head_logit.device),float('-inf'))
            else:
                if head_root_index:
                    head_logit.index_fill_(1,torch.tensor(head_root_index,device=head_logit.device),float('-inf'))
            if general_words_only:
                if head_leaf_index+head_root_index:
                    head_logit.index_fill_(1,torch.tensor(head_leaf_index+head_root_index,device=head_logit.device),float('-inf'))
            #head_logit[:,0] = -float('inf')
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target,
                    dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:,None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    leaf_index_i, root_index_i = self.cl_cutoffs_leaf_index[i], self.cl_cutoffs_root_index[i]
                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    if predict_root:
                        if leaf_index_i:
                            tail_logit_i.index_fill_(1,torch.tensor(leaf_index_i,device=head_logit.device),float('-inf'))
                    else:
                        if root_index_i:
                            tail_logit_i.index_fill_(1,torch.tensor(root_index_i,device=head_logit.device),float('-inf'))
                    if general_words_only:
                        if leaf_index_i+root_index_i:
                            tail_logit_i.index_fill_(1,torch.tensor(leaf_index_i+root_index_i,device=head_logit.device),float('-inf'))
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] \
                              + tail_logprob_i.gather(1, target_i[:,None]).squeeze(1)

                if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll


class HeriarchicalClassedProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 keep_order=False, cl_root_leaf_dict=None):
        super(HeriarchicalClassedProjectedAdaptiveLogSoftmax, self).__init__()
        
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        # here we assume 1 token only map to 1 class
        # vocab : [c1] [c2] [t1] [t2] [t3] | [t_l1] [t_l2] | [t_l3] |
        # cutoff: [20000, 20002, 20003, ..., XXXXX, 200000, ]
        cl_all_root_index = []
        cl_all_leaf_index = []
        cutoffs = [20000]
        for k,v in cl_root_leaf_dict.items():
            cutoffs.append(cutoffs[-1]+len(v))
            cl_all_root_index.append(k)
            cl_all_leaf_index.extend(v)
        cutoffs.append(200000)

        self.cutoffs = cutoffs + [n_token]
        # cutoff: [20000, 20002, 20003, ..., XXXXX, 200000, n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.cl_all_root_index = cl_all_root_index
        self.cl_all_leaf_index = cl_all_leaf_index

        self.n_clusters = len(self.cutoffs) - 1 - len(cl_all_root_index)

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(
                torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
            # self.cl_cutoffs_root_index = []
            # self.cl_cutoffs_leaf_index = []
            # for i in range(len(self.cutoffs)):
            #     l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
            #     self.cl_cutoffs_root_index.append(
            #         [index-l_idx for index in self.cl_all_root_index if index < r_idx and index >= l_idx])
            #     self.cl_cutoffs_leaf_index.append(
            #         [index-l_idx for index in self.cl_all_leaf_index if index < r_idx and index >= l_idx])

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(
                        nn.Parameter(torch.Tensor(d_proj, d_embed))
                    )
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                self.out_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i))
                )

                self.out_layers.append(nn.Linear(d_emb_i, r_idx-l_idx))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target, keep_order=True, predict_root=False, general_words_only=False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
            predict_root: True for class label + general words; False for normal LM
            separate_vocab: True for general words only
        '''

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            if predict_root:
                logit[:, self.cl_all_leaf_index] = -float('inf')
            else:
                logit[:, self.cl_all_root_index] = -float('inf')
            if general_words_only:
                logit[:, self.cl_all_leaf_index +
                      self.cl_all_root_index] = -float('inf')

            nll = -F.log_softmax(logit, dim=-1) \
                    .gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # construct weights and biases
            
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            # head_leaf_index, head_root_index = self.cl_cutoffs_leaf_index[
            #     0], self.cl_cutoffs_root_index[0]
            head_logit = self._compute_logit(
                hidden, head_weight, head_bias, head_proj)

            #head_logit[:,0] = -float('inf')
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target,
                                   dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(
                        1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(
                        hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    j = i-1 if i <= len(self.cl_all_root_index) else - \
                        (i-len(self.cl_all_root_index))
                    logprob_i = head_logprob_i[:, j] \
                        + tail_logprob_i.gather(1,
                                                target_i[:, None]).squeeze(1)

                if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll
