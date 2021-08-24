import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax, ClassedProjectedAdaptiveLogSoftmax, HeriarchicalClassedProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits
# from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations


def parameters(m, recurse=True):
    def model_parameters(m):
        ps = m._former_parameters.values() \
            if hasattr(m, "_former_parameters") \
            else m.parameters(recurse=False)
        for p in ps:
            yield p

    for m in m.modules() if recurse else [m]:
        for p in model_parameters(m):
            yield p


class PositionalEmbedding(nn.Module):
    def __init__(self, demb, sega=False):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None, sega=False):
        if torch.is_autocast_enabled():
            self.inv_freq = self.inv_freq.half()
            pos_seq=pos_seq.half()
        if sega:
            sinusoid_inp = torch.bmm(pos_seq.unsqueeze(-1),
                                     self.inv_freq.unsqueeze(0).unsqueeze(0).expand(pos_seq.size(0), -1, -1))
        else:
            sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(
                    attn_mask[None, :, :, None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(
                    attn_mask[:, :, :, None].bool(), -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(
            self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, sega=False):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        # qlen x bsz x n_head x d_head
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if sega:
            r_head_k = r_head_k.view(rlen, bsz, self.n_head, self.d_head)
        else:
            # qlen x n_head x d_head
            r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        #### compute attention score
        # qlen x bsz x n_head x d_head
        rw_head_q = w_head_q + r_w_bias
        # qlen x klen x bsz x n_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_r_bias
        if sega:
            BD = torch.einsum('ibnd,jbnd->ijbn', (rr_head_q, r_head_k))
        else:
            # qlen x klen x bsz x n_head
            BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None].bool(), -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None].bool(), -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output, attn_score


'''
class LongformerRelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(LongformerRelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, sega=False,
                attention_window=1,attention_dilation=1):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if attn_mask is not None:
            # batch q_len k_len 1
            # attention_mask = attention_mask.squeeze(-1)
            key_padding_mask = attn_mask == 1
            #ones = key_padding_mask.new_ones(size=key_padding_mask.size())  # tensor of ones
            #key_padding_mask = diagonaled_mm_tvm(ones, key_padding_mask, attention_window, attention_dilation, False, 0, True)
            extra_attention_mask = attn_mask == 2
            remove_from_windowed_attention_mask = attn_mask != 1

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                 device=num_extra_indices_per_batch.device)
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None



        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)       # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        if sega:
            r_head_k = r_head_k.view(rlen, bsz, self.n_head, self.d_head)
        else:
            r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        # AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        AC =  diagonaled_mm_tvm(rw_head_q.transpose(0,1).contiguous(),
                                w_head_k.transpose(0,1).contiguous(),
                                attention_window, attention_dilation,False,0,True)            # batch qlen n_head w+1
        rr_head_q = w_head_q + r_r_bias
        if sega:
            BD = diagonaled_mm_tvm(rr_head_q.transpose(0,1).contiguous(),
                                   r_head_k.transpose(0,1).contiguous(),attention_window,
                                   attention_dilation,False,0,True)            # batch qlen n_head w+1
            # BD = torch.einsum('ibnd,jbnd->ijbn', (rr_head_q, r_head_k))
        else:
            r_head_k = r_head_k.transpose(0,1).unsqueeze().repeat(bsz,1,1,1).contiguous()     # bsz n_head qlen d_head
            BD = diagonaled_mm_tvm(rr_head_q.transpose(0, 1).contiguous(), r_head_k, attention_window,
                                   attention_dilation, False, 0, True)  # batch qlen n_head w+1
            # BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [bsz x qlen x n_head x w+1]
        attn_score = AC + BD
        mask_invalid_locations(attn_score, attention_window, attention_dilation, True)
        # TODO
        # if remove_from_windowed_attention_mask is not None:
        #     # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
        #     # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
        #     # remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
        #     # cast to float/half then replace 1's with -inf
        #     float_mask = remove_from_windowed_attention_mask.type_as(attn_score).masked_fill(remove_from_windowed_attention_mask, -10000.0)
        #     #repeat_size = 1 if isinstance(attention_dilation, int) else len(attention_dilation)
        #     #float_mask = float_mask.repeat(1, 1, repeat_size, 1)
        #     ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
        #     # diagonal mask with zeros everywhere and -inf inplace of padding
        #     d_mask = diagonaled_mm_tvm(ones, float_mask, attention_window, attention_dilation, False, 0, True)
        #     attn_score += d_mask
        if extra_attention_mask is not None:
            raise NotImplementedError

        attn_score.mul_(self.scale)

        attn_weights_float = F.softmax(attn_score, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        # TODO
        # if key_padding_mask is not None:
        #     # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        #     attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask,
        #                                            0.0)

        attn_score = attn_weights_float.type_as(attn_score)
        attn_prob = self.dropatt(attn_score)

        w_head_v = w_head_v.transpose(0,1).float().contiguous()
        attn_vec = diagonaled_mm_tvm(attn_prob,w_head_v,attention_window,attention_dilation,True,0,True)
        attn_vec = attn_vec.contiguous().transpose(0,1)
        attn_vec = attn_vec.view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)  # [qlen x bsz x n_head x d_head]

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output
'''


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        # qlen x bsz x n_head x d_head
        rw_head_q = w_head_q + r_w_bias[None]

        # qlen x klen x bsz x n_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))
        # 1    x klen x 1   x n_head
        D_ = r_bias[None, :, None]
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(
                    attn_mask[None, :, :, None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(
                    attn_mask[:, :, :, None].bool(), -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                                  **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None, sega=False):

        output, attn_score = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                           attn_mask=dec_attn_mask,
                                           mems=mems, sega=sega)
        output = self.pos_ff(output)

        return output, attn_score


'''
class LongformerRelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(LongformerRelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = LongformerRelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None, sega=False,attention_window=1):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems,sega=sega,attention_window=attention_window)
        output = self.pos_ff(output)

        return output
'''


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(
                    torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(
                    torch.Tensor(d_proj, d_emb_i)))
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.register_buffer("_half_tensor", torch.HalfTensor(1))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            inp_flat = inp.reshape(-1)
            if torch.is_autocast_enabled():
                emb_flat = self._half_tensor.new(inp_flat.shape + (self.d_proj,))
            else:
                emb_flat = self._float_tensor.new(inp_flat.shape + (self.d_proj,))

            for i in range(len(self.cutoffs)):
                mask = inp_flat.lt(self.cutoffs[i])
                if i > 0:
                    mask.mul_(inp_flat.ge(self.cutoffs[i - 1]))
                    chunk_input = inp_flat[mask] - self.cutoffs[i - 1]
                else:
                    chunk_input = inp_flat[mask]
                if mask.any():
                    emb_i = self.emb_layers[i](chunk_input)
                    emb_flat[mask]=F.linear(emb_i, self.emb_projs[i])
            embed = emb_flat.view(*inp.size(), self.d_proj)
        embed.mul_(self.emb_scale)
        return embed
    # def forward(self, inp):
    #     if self.div_val == 1:
    #         embed = self.emb_layers[0](inp)
    #         if self.d_proj != self.d_embed:
    #             embed = F.linear(embed, self.emb_projs[0])
    #     else:
    #         param = next(self.parameters())
    #         inp_flat = inp.reshape(-1)
    #         emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
    #                                dtype=torch.float16 if torch.is_autocast_enabled() else param.dtype, device=param.device)
    #         for i in range(len(self.cutoffs)):
    #             l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

    #             mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
    #             indices_i = mask_i.nonzero().squeeze()

    #             if indices_i.numel() == 0:
    #                 continue

    #             inp_i = inp_flat.index_select(0, indices_i) - l_idx
    #             emb_i = self.emb_layers[i](inp_i)
    #             emb_i = F.linear(emb_i, self.emb_projs[i])

    #             emb_flat.index_copy_(0, indices_i, emb_i)

    #         embed = emb_flat.view(*inp.size(), self.d_proj)

    #     embed.mul_(self.emb_scale)

    #     return embed


def get_ith_embed_layer_given_index(index, word_emb):
    for i in range(len(word_emb.cutoffs)):
        l_idx, r_idx = word_emb.cutoff_ends[i], word_emb.cutoff_ends[i + 1]
        if index >= l_idx and index < r_idx:
            new_index = index-l_idx
            return i, new_index
    return -1, -1


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1, cl_all_root_index=None, cl_all_leaf_index=None,
                 adaptive_class_softmax=False, cl_root_leaf_dict=None, word2class_id=None,
                 mix_vocab=True):
        super(MemTransformerLM, self).__init__()
        # if word2class_id:
        #     n_token = n_token-len(cl_root_leaf_dict)
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word2class = word2class_id
        # if word2class_id:
        #     self.word_emb = AdaptiveEmbedding(n_token-len(cl_root_leaf_dict), d_embed, d_model, cutoffs,
        #                         div_val=div_val)
        #     self.hypernym_emb = nn.Embedding(
        #         len(cl_all_root_index)+1, d_embed, padding_idx=0)
        # else:
        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                        div_val=div_val)
        if not mix_vocab:
            self.auxiliary_projection_layer = nn.Linear(d_model, d_model)
            self.auxiliary_output_layer = nn.Linear(
                d_model, len(cl_all_root_index))
        self.cl_root_vocab = {}
        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)
        # use adaptive softmax (including standard softmax)
        else:
            if not cl_all_root_index or not cl_all_leaf_index:

                self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                        cutoffs, div_val=div_val)
            elif adaptive_class_softmax:
                self.crit = HeriarchicalClassedProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                                           cutoffs, div_val=div_val,
                                                                           cl_root_leaf_dict=cl_root_leaf_dict)
            else:
                self.crit = ClassedProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                               cutoffs, div_val=div_val,
                                                               cl_all_root_index=cl_all_root_index,
                                                               cl_all_leaf_index=cl_all_leaf_index, word2class=self.word2class)
                self.cl_root_vocab = {
                    old_idx: new_idx for new_idx, old_idx in enumerate(cl_all_root_index)}
            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight
                # if self.word2class:
                #     self.crit.hypernym_emb.weight = self.hypernym_emb.weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.predict_root = False
        self.same_length = same_length
        self.clamp_len = clamp_len
        self._create_params()

    def change(self):
        for k, v in self.word2class.items():
            self.word_emb.emb_layers[0].weight.data[k] = torch.Tensor(
                (self.word_emb.emb_layers[0].weight[v] + self.word_emb.emb_layers[0].weight[k]).tolist())

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None, pst=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:, :, None]

        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out, _ = layer(core_out, pos_emb, self.r_w_bias,
                                    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, hids, new_mems

    def _forward_offset(self, dec_inp, hypernym_inp, mems=None, pst=None, mask=False):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)
        # if mask:
        #     word_emb[(hypernym_inp!=0)] = 0
        hypernym_emb = self.word_emb(hypernym_inp)
        hypernym_emb[(hypernym_inp==0)] = 0
        word_emb = word_emb+hypernym_emb

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:, :, None]

        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out, _ = layer(core_out, pos_emb, self.r_w_bias,
                                    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, hids, new_mems


    def auxiliary_forward(self, hiddens, indices, root_target, tgt_len, auxiliary_layer=-1):
        root_pred_hid = self.drop(hiddens[auxiliary_layer])[-tgt_len:]
        root_hidden_state = torch.reshape(
            root_pred_hid, (-1, root_pred_hid.size(-1))).index_select(0, indices)
        root_hidden_state = self.auxiliary_projection_layer(root_hidden_state)
        root_logits = self.crit._compute_logit(root_hidden_state,
                                               self.word_emb.emb_layers[0].weight.index_select(0, torch.tensor(
                                                   list(self.cl_root_vocab.keys()), device=root_pred_hid.device)),
                                               bias=None, proj=None)
        #root_logits = self.auxiliary_output_layer(root_hidden_state)
        selected_root_target = torch.reshape(
            root_target, (-1,)).index_select(0, indices)
        auxiliary_target = torch.tensor(
            [self.cl_root_vocab[x] for x in selected_root_target.tolist()], device=selected_root_target.device)
        auxiliary_loss = - \
            F.log_softmax(root_logits, dim=1).gather(
                1, auxiliary_target.unsqueeze(1)).squeeze(1)
        return auxiliary_loss

    def forward(self, data, target, root_target, mems, args, class_prediction=False,
    hypernym_input=None):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        multi_obj=args.multi_obj
        mix_vocab=args.mix_vocab
        auxiliary_layer = args.auxiliary_layer
        if not mems:
            mems = self.init_mems()

        tgt_len = target.size(0)
        if hypernym_input is not None:
            # mask=False
            # if data.max()>=self.word_emb.n_token:
            #     mask=True
            #     data = torch.clamp(data,max=self.word_emb.n_token-1)
            hidden, hiddens, new_mems = self._forward_offset(data, hypernym_input, mems=mems)
        else:
            hidden, hiddens, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        indices = torch.reshape((target != root_target),
                                (-1,)).nonzero().squeeze()
        if class_prediction:
            if multi_obj:
                # predict both the normal targets for all tokens and then, predict the root labels for the leaf nodes
                loss = self.crit(torch.reshape(
                    pred_hid, (-1, pred_hid.size(-1))), torch.reshape(target, (-1,)))
                if mix_vocab:
                    auxiliary_loss = self.crit(torch.reshape(
                        pred_hid, (-1, pred_hid.size(-1))), torch.reshape(root_target, (-1,)), predict_root=True)
                    auxiliary_loss = torch.reshape(
                        auxiliary_loss, (-1,)).index_select(0, indices)
                else:
                    auxiliary_loss = self.auxiliary_forward(
                        hiddens, indices, root_target, tgt_len, auxiliary_layer)
            else:
                # predict the normal targets only for non-leaf nodes and root labels for the leaf nodes
                if mix_vocab:
                    loss = self.crit(torch.reshape(pred_hid, (-1, pred_hid.size(-1))),
                                     torch.reshape(root_target, (-1,)), predict_root=True)
                    auxiliary_loss = torch.reshape(
                        loss, (-1,)).index_select(0, indices)
                else:
                    loss = self.crit(torch.reshape(pred_hid, (-1, pred_hid.size(-1))),
                                     torch.reshape(root_target, (-1,)), predict_root=False)
                    auxiliary_loss = self.auxiliary_forward(
                        hiddens, indices, root_target, tgt_len, auxiliary_layer)
                    loss.index_copy_(0, indices, auxiliary_loss)
        else:
            loss = self.crit(torch.reshape(
                pred_hid, (-1, pred_hid.size(-1))), torch.reshape(target, (-1,)))
            auxiliary_loss = torch.zeros_like(loss)

        loss = loss.view(tgt_len, -1)
        # if len(auxiliary_loss) == 0:
        #     auxiliary_loss = torch.zeros_like(loss)
        if new_mems is None:
            return [loss, auxiliary_loss]
        else:
            return [loss, auxiliary_loss] + new_mems


class SegaMemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1, sparse_mode='none'):
        super(SegaMemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.sparse_mode = sparse_mode

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            t_pos_size = self.d_model//3 + self.d_model//3 % 2
            s_pos_size = self.d_model//3 + self.d_model//3 % 2
            p_pos_size = self.d_model - t_pos_size - s_pos_size
            self.t_pos_emb = PositionalEmbedding(t_pos_size)
            self.s_pos_emb = PositionalEmbedding(s_pos_size)
            self.p_pos_emb = PositionalEmbedding(p_pos_size)
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(parameters(self))
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def init_mems_pst(self):
        if self.mem_len > 0:
            param = next(parameters(self))
            mems_pst = (torch.empty(0, dtype=torch.int64, device=param.device),
                        torch.empty(0, dtype=torch.int64, device=param.device),
                        torch.empty(0, dtype=torch.int64, device=param.device))
            return mems_pst
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def _forward(self, dec_inp, mems=None, pst=None, mems_pst=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:, :, None]

        def local_window_mask(input_mask, pst, window_length, global_pos=False):
            # q k 1
            qlen, klen, _ = input_mask.size()
            # klen bsz
            p, s, t = pst
            sentence_end = ~torch.cat([s[1:, :], -s.new_ones(1, bsz)]).eq(s)
            paragraph_end = ~torch.cat([p[1:, :], -p.new_ones(1, bsz)]).eq(p)
            all_ones = word_emb.new_ones(qlen, klen)
            window_mask = (torch.triu(
                all_ones, diagonal=1 + window_length+mlen) + torch.tril(all_ones, -window_length - 1+mlen)).byte()[:, :, None]

            if global_pos:
                window_mask = window_mask.repeat(1, 1, bsz)
                for i in range(bsz):
                    sentence_end_i = sentence_end[:,
                                                  i].nonzero().squeeze(-1).tolist()
                    paragraph_end_i = paragraph_end[:, i].nonzero(
                    ).squeeze(-1).tolist()
                    for end in sentence_end_i:
                        window_mask[:, end, i] = 0
                        if end in paragraph_end_i:
                            window_mask[:, end - 1, i] = 0
            input_mask = input_mask | window_mask
            return input_mask

        def sega_window_mask(input_mask, pst, sent_window=1, global_pos=False):
            # qlen klen bsz
            input_mask = input_mask.repeat(1, 1, bsz)
            empty_input_mask = input_mask.new_ones(input_mask.size())
            # klen bsz
            p, s, t = pst
            sentence_end = ~torch.cat(
                [s[mlen+1:, :], -s.new_ones(1, bsz)]).eq(s[mlen:, :])
            all_paragraph_end = ~torch.cat(
                [p[1:, :], -p.new_ones(1, bsz)]).eq(p)
            all_sentence_start = ~torch.cat(
                [-s.new_ones(1, bsz), s[:-1, :]]).eq(s)
            all_sentence_end = ~torch.cat(
                [s[1:, :], -s.new_ones(1, bsz)]).eq(s)

            for i in range(bsz):
                all_sentence_start_i = all_sentence_start[:, i].nonzero(
                ).squeeze(-1).tolist()
                sentence_end_i = sentence_end[:,
                                              i].nonzero().squeeze(-1).tolist()
                all_sentence_end_i = all_sentence_end[:, i].nonzero(
                ).squeeze(-1).tolist()
                all_paragraph_end_i = all_paragraph_end[:, i].nonzero(
                ).squeeze(-1).tolist()
                last_sentence_start_list = [0]*sent_window
                for start, end in zip(all_sentence_start_i, all_sentence_end_i):
                    if sent_window > 1:
                        last_sentence_start = last_sentence_start_list[-sent_window + 1]
                    else:
                        last_sentence_start = start
                    if end-mlen in sentence_end_i:
                        bq = 0 if start-mlen < 0 else 0
                        empty_input_mask[bq:end-mlen+1,
                                         last_sentence_start:end+1, i] = 0
                    last_sentence_start_list.pop(0)
                    last_sentence_start_list.append(start)
                    if global_pos:
                        empty_input_mask[:, end, i] = 0
                        if end in all_paragraph_end_i:
                            empty_input_mask[:, end-1, i] = 0
            output_mask = input_mask | empty_input_mask

            # q k bsz
            return output_mask

        def heuristic_mask(attn_score, input_mask):
            empty_input_mask = input_mask.new_ones(input_mask.size())

        hids = []
        if self.attn_type == 0:  # default
            p, s, t = pst
            if mlen > 0:
                m_p, m_s, m_t = mems_pst

                ab_t = torch.cat([m_t, t], 0)
                ab_p = torch.cat([m_p, p], 0)
                ab_s = torch.cat([m_s, s], 0)
            else:
                ab_p, ab_s, ab_t = p, s, t
            max_t, _ = ab_t.max(0)
            t_pos_seq = max_t.unsqueeze(0).expand(ab_t.size(0), -1) - ab_t
            max_p, _ = ab_p.max(0)
            p_pos_seq = max_p.unsqueeze(0).expand(ab_p.size(0), -1) - ab_p
            max_s, _ = ab_s.max(0)
            s_pos_seq = max_s.unsqueeze(0).expand(ab_s.size(0), -1) - ab_s
            # t_pos_seq = ab_t.max(0)[0]-ab_t
            # p_pos_seq = ab_p.max(0)[0]-ab_p
            # s_pos_seq = ab_s.max(0)[0]-ab_s

            if self.clamp_len > 0:
                t_pos_seq.clamp_(max=self.clamp_len)
                s_pos_seq.clamp_(max=self.clamp_len)
                p_pos_seq.clamp_(max=self.clamp_len)
            t_pos_emb = self.t_pos_emb(
                t_pos_seq.to(dtype=word_emb.dtype), sega=True)
            s_pos_emb = self.s_pos_emb(
                s_pos_seq.to(dtype=word_emb.dtype), sega=True)
            p_pos_emb = self.p_pos_emb(
                p_pos_seq.to(dtype=word_emb.dtype), sega=True)
            pos_emb = torch.cat([t_pos_emb, s_pos_emb, p_pos_emb], -1)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)

            if 'global' in self.sparse_mode:
                global_pos = True
            else:
                global_pos = False
            dec_attn_mask_origin = dec_attn_mask
            # fix window size
            if "static" in self.sparse_mode:
                window_length = int(self.sparse_mode.split('_')[0][6:])
                dec_attn_mask = local_window_mask(dec_attn_mask, [ab_p, ab_s, ab_t],
                                                  window_length=window_length, global_pos=global_pos)
            # dynamic window size
            elif "dynamic" in self.sparse_mode:
                window_length = int(self.sparse_mode.split('_')[0][7:])
                dec_attn_mask = sega_window_mask(dec_attn_mask, [ab_p, ab_s, ab_t], sent_window=window_length,
                                                 global_pos=global_pos)

            for i, layer in enumerate(self.layers):
                if 'layer' in self.sparse_mode:
                    n_layer = int(self.sparse_mode.split('_')[-1][5:])
                    if i < n_layer:
                        dec_attn_mask = dec_attn_mask_origin
                mems_i = None if mems is None else mems[i]
                core_out, attn_score = layer(core_out, pos_emb, self.r_w_bias,
                                             self.r_r_bias, dec_attn_mask=dec_attn_mask,
                                             mems=mems_i, sega=True)
                if 'heuristic' in self.sparse_mode:
                    dec_attn_mask = heuristic_mask(
                        attn_score, dec_attn_mask_origin)
                hids.append(core_out)
        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, mems, pst, mems_pst):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self.init_mems()
        if not mems_pst:
            mems_pst = self.init_mems_pst()
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(
            data, mems=mems, pst=pst, mems_pst=mems_pst)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                                  self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)),
                             target.contiguous().view(-1), keep_order=True)
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


class Sega_wo_p_MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1):
        super(Sega_wo_p_MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            t_pos_size = self.d_model//2 + self.d_model//2 % 2
            s_pos_size = self.d_model - t_pos_size
            self.t_pos_emb = PositionalEmbedding(t_pos_size)
            self.s_pos_emb = PositionalEmbedding(s_pos_size)
            # self.p_pos_emb = PositionalEmbedding(p_pos_size)
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def init_mems_pst(self):
        if self.mem_len > 0:
            param = next(self.parameters())
            mems_pst = (torch.empty(0, dtype=torch.int64, device=param.device),
                        torch.empty(0, dtype=torch.int64, device=param.device),
                        torch.empty(0, dtype=torch.int64, device=param.device))
            return mems_pst
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def _forward(self, dec_inp, mems=None, pst=None, mems_pst=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:, :, None]

        hids = []
        if self.attn_type == 0:  # default
            p, s, t = pst
            m_p, m_s, m_t = mems_pst

            ab_t = torch.cat([m_t, t], 0)
            #ab_p = torch.cat([m_p,p],0)
            ab_s = torch.cat([m_s, s], 0)
            max_t, _ = ab_t.max(0)
            t_pos_seq = max_t.unsqueeze(0).expand(ab_t.size(0), -1) - ab_t
            #max_p, _ = ab_p.max(0)
            #p_pos_seq = max_p.unsqueeze(0).expand(ab_p.size(0), -1) - ab_p
            max_s, _ = ab_s.max(0)
            s_pos_seq = max_s.unsqueeze(0).expand(ab_s.size(0), -1) - ab_s
            # t_pos_seq = ab_t.max(0)[0]-ab_t
            # p_pos_seq = ab_p.max(0)[0]-ab_p
            # s_pos_seq = ab_s.max(0)[0]-ab_s

            if self.clamp_len > 0:
                t_pos_seq.clamp_(max=self.clamp_len)
                s_pos_seq.clamp_(max=self.clamp_len)
                #p_pos_seq.clamp_(max=self.clamp_len)
            t_pos_emb = self.t_pos_emb(
                t_pos_seq.to(dtype=word_emb.dtype), sega=True)
            s_pos_emb = self.s_pos_emb(
                s_pos_seq.to(dtype=word_emb.dtype), sega=True)
            # p_pos_emb = self.p_pos_emb(p_pos_seq.to(dtype=word_emb.dtype),sega=True)
            pos_emb = torch.cat([t_pos_emb, s_pos_emb], -1)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                                 self.r_r_bias, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i, sega=True)
                hids.append(core_out)
        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, mems, pst, mems_pst):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self.init_mems()
        if not mems_pst:
            mems_pst = self.init_mems_pst()
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(
            data, mems=mems, pst=pst, mems_pst=mems_pst)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                                  self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(
                pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


class Sega_wo_s_MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1):
        super(Sega_wo_s_MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            t_pos_size = self.d_model//2 + self.d_model//2 % 2
            #s_pos_size = self.d_model//3 + self.d_model//3%2
            p_pos_size = self.d_model - t_pos_size
            self.t_pos_emb = PositionalEmbedding(t_pos_size)
            #self.s_pos_emb = PositionalEmbedding(s_pos_size)
            self.p_pos_emb = PositionalEmbedding(p_pos_size)
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def init_mems_pst(self):
        if self.mem_len > 0:
            param = next(self.parameters())
            mems_pst = (torch.empty(0, dtype=torch.int64, device=param.device),
                        torch.empty(0, dtype=torch.int64, device=param.device),
                        torch.empty(0, dtype=torch.int64, device=param.device))
            return mems_pst
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def _forward(self, dec_inp, mems=None, pst=None, mems_pst=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:, :, None]

        hids = []
        if self.attn_type == 0:  # default
            p, s, t = pst
            m_p, m_s, m_t = mems_pst

            ab_t = torch.cat([m_t, t], 0)
            ab_p = torch.cat([m_p, p], 0)
            #ab_s = torch.cat([m_s,s],0)
            max_t, _ = ab_t.max(0)
            t_pos_seq = max_t.unsqueeze(0).expand(ab_t.size(0), -1) - ab_t
            max_p, _ = ab_p.max(0)
            p_pos_seq = max_p.unsqueeze(0).expand(ab_p.size(0), -1) - ab_p
            #max_s, _ = ab_s.max(0)
            #s_pos_seq = max_s.unsqueeze(0).expand(ab_s.size(0), -1) - ab_s
            # t_pos_seq = ab_t.max(0)[0]-ab_t
            # p_pos_seq = ab_p.max(0)[0]-ab_p
            # s_pos_seq = ab_s.max(0)[0]-ab_s

            if self.clamp_len > 0:
                t_pos_seq.clamp_(max=self.clamp_len)
                #s_pos_seq.clamp_(max=self.clamp_len)
                p_pos_seq.clamp_(max=self.clamp_len)
            t_pos_emb = self.t_pos_emb(
                t_pos_seq.to(dtype=word_emb.dtype), sega=True)
            #s_pos_emb = self.s_pos_emb(s_pos_seq.to(dtype=word_emb.dtype),sega=True)
            p_pos_emb = self.p_pos_emb(
                p_pos_seq.to(dtype=word_emb.dtype), sega=True)
            pos_emb = torch.cat([t_pos_emb, p_pos_emb], -1)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                                 self.r_r_bias, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i, sega=True)
                hids.append(core_out)
        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, mems, pst, mems_pst):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self.init_mems()
        if not mems_pst:
            mems_pst = self.init_mems_pst()
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(
            data, mems=mems, pst=pst, mems_pst=mems_pst)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                                  self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(
                pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


'''
class LongTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1, window_size=5,dilation_size=1):
        super(LongTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.window_size = window_size
        self.dilation_size = dilation_size

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    LongformerRelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        else:
            raise NotImplementedError

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention
            t_pos_size = self.d_model//3 + self.d_model//3%2
            s_pos_size = self.d_model//3 + self.d_model//3%2
            p_pos_size = self.d_model - t_pos_size - s_pos_size
            self.t_pos_emb = PositionalEmbedding(t_pos_size)
            self.s_pos_emb = PositionalEmbedding(s_pos_size)
            self.p_pos_emb = PositionalEmbedding(p_pos_size)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def init_mems_pst(self):
        if self.mem_len > 0:
            param = next(self.parameters())
            mems_pst=(torch.empty(0, dtype=torch.int64, device=param.device),
                      torch.empty(0, dtype=torch.int64, device=param.device),
                      torch.empty(0, dtype=torch.int64, device=param.device))
            return mems_pst
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def _forward(self, dec_inp, mems=None, pst=None, mems_pst=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        all_ones = word_emb.new_ones(qlen, klen)
        if self.same_length:
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen).bool() |
                    torch.tril(all_ones, -mask_shift_len) | torch.tril(all_ones, -self.window_size-1)).byte()[:, :, None] # -1
        else:
            dec_attn_mask = (torch.triu(
                all_ones, diagonal=1+mlen)+torch.tril(all_ones, -self.window_size-1)).byte()[:,:,None]
        dec_attn_mask = dec_attn_mask.unsqueeze(0).repeat(bsz,1,1,1)
        hids = []
        if self.attn_type == 0: # default
            p, s, t = pst
            if mems_pst and mlen>0:
                m_p, m_s, m_t = mems_pst
                ab_t = torch.cat([m_t, t], 0)
                ab_p = torch.cat([m_p,p],0)
                ab_s = torch.cat([m_s,s],0)
            else:
                ab_t, ab_p, ab_s = t,p,s
            max_t,_ = ab_t.max(0)
            t_pos_seq = max_t.unsqueeze(0).expand(ab_t.size(0),-1) - ab_t
            max_p, _ = ab_p.max(0)
            p_pos_seq = max_p.unsqueeze(0).expand(ab_p.size(0), -1) - ab_p
            max_s, _ = ab_s.max(0)
            s_pos_seq = max_s.unsqueeze(0).expand(ab_s.size(0), -1) - ab_s
            # t_pos_seq = ab_t.max(0)[0]-ab_t
            # p_pos_seq = ab_p.max(0)[0]-ab_p
            # s_pos_seq = ab_s.max(0)[0]-ab_s

            if self.clamp_len > 0:
                t_pos_seq.clamp_(max=self.clamp_len)
                s_pos_seq.clamp_(max=self.clamp_len)
                p_pos_seq.clamp_(max=self.clamp_len)
            t_pos_emb = self.t_pos_emb(t_pos_seq.to(dtype=word_emb.dtype),sega=True)
            s_pos_emb = self.s_pos_emb(s_pos_seq.to(dtype=word_emb.dtype),sega=True)
            p_pos_emb = self.p_pos_emb(p_pos_seq.to(dtype=word_emb.dtype),sega=True)
            pos_emb = torch.cat([t_pos_emb,s_pos_emb,p_pos_emb],-1)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i,sega=True,attention_window=self.window_size)
                hids.append(core_out)
        elif self.attn_type == 1: # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                        r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, mems, pst, mems_pst):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()
        if not mems_pst: mems_pst = self.init_mems_pst()
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems, pst=pst, mems_pst=mems_pst)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems
'''

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(
        data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                                     args.d_model, args.d_head, args.d_inner, args.dropout,
                                     dropatt=args.dropout, tie_weight=True,
                                     d_embed=d_embed, div_val=div_val,
                                     tie_projs=tie_projs, pre_lnorm=True,
                                     tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                     cutoffs=cutoffs, attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
