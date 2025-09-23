import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from einops import rearrange,repeat

from typing import Optional, List
from layers.dyseformer.embedding import get_activation_fn


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False, scores_only=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.scores_only = scores_only

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        if not self.scores_only:
            # compute the new values given the attention weights
            output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

            return output, attn_weights, attn_scores
        else:
            return attn_weights, attn_scores


class BroadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, topk=1, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, scores_only=False):
        """Broadcast multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x channel x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()

        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.scores_only = scores_only
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        
        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
    
    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        bs = Q.size(0)
        channel = K.size(1)
        if K is None: K = Q
        if V is None: V = Q

        q_s = self.W_Q(Q)
        k_s = self.W_K(K)
        v_s = self.W_V(V)

        # Linear (+ split in multiple heads)
        q_s = q_s.view(bs, -1, self.n_heads, self.d_k).transpose(1,2) # q_s    : [bs x n_heads x max_q_len x d_k]
        q_s = q_s.unsqueeze(2).expand(-1, -1, channel, -1, -1)                       # q_s    : [bs x n_heads x channel x max_q_len x d_k]
        k_s = k_s.view(bs, channel, -1, self.n_heads, self.d_k).permute(0,3,1,4,2)     # k_s    : [bs x n_heads x channel x d_k x q_len]
        v_s = v_s.view(bs, channel, -1, self.n_heads, self.d_v).permute(0,3,1,2,4)       # v_s    : [bs x n_heads x channel x q_len x d_v]


        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q_s, k_s) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]
        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v_s)  # [bs x n_heads x channel x q_len x d_v]
        output = output.mean(dim=2)  # [bs x n_heads x max_q_len x d_v]
        
        # back to the original inputs dimensions
        output = output.permute(0,2,1,3).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)
        
        return output, attn_weights, attn_scores


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, scores_only=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        # shift 

        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.scores_only = scores_only
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa, scores_only=scores_only)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        q_s = self.W_Q(Q)
        k_s = self.W_K(K)
        v_s = self.W_V(V)
    
        # Linear (+ split in multiple heads)
        q_s = q_s.view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = k_s.view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] by transpose(1,2) + transpose(2,3)
        v_s = v_s.view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]
        
        if not self.scores_only:
            # Apply Scaled Dot-Product Attention (multiple heads)
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
            # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

            # back to the original inputs dimensions
            output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
            output = self.to_out(output)

            return output, attn_weights, attn_scores
        else:
            attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return attn_weights, attn_scores

class GlobalRelate(nn.Module):
    '''
        x shape: [bs, channel, patch_num, d_model]
        x_glb shape: [bs, channel, d_model]
    '''
    def __init__(self, d_model, nvars, d_ff=256, bias=True, activation="gelu", dropout=0.) -> None:
        super().__init__()

        self.d_model = d_model # (512)

        self.MLP = nn.Sequential(nn.Linear(2*d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, nvars, bias=bias))
    
    def forward(self, x, x_glb):
        B, C, patch_num, _ = x.shape
        self_glb  = x_glb.unsqueeze(2).expand(-1, -1, patch_num, -1)
        src = torch.cat((x, self_glb), dim=-1)
        scores = self.MLP(src)
        return scores







