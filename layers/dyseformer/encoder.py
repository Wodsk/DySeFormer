from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import numpy as np

from math import sqrt
from layers.dyseformer.embedding import *
from layers.dyseformer.layers import *
from layers.dyseformer.utils import *
from layers.RevIN import RevIN

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual: 
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2) 
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                x = self.flattens[i](x[:,i,:,:])          # x: [bs x d_model * patch_num]
                x = self.linears[i](x)                    # x: [bs x target_window]
                x = self.dropouts[i](x)
                x_out.append(x)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class Encoder(nn.Module):  
    def __init__(self, nvars:int, seq_len:int, pred_len:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 topk:int=4, use_timestamp:bool=False, n_blocks:int=1, n_layers:int=3, 
                 d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, padding_patch = None, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, individual=False, bias=True, **kwargs):
        
        
        super().__init__()
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(nvars, affine=affine, subtract_last=subtract_last)

        self.embedding_layer = Embedding_layer(padding_patch, seq_len, patch_len, stride, d_model, dropout=dropout, pe=pe, learn_pe=learn_pe)
        self.encode_block_list = nn.ModuleList()
        self.global_block_list = nn.ModuleList()
        
        self.n_blocks = n_blocks

        # patch embedding and proj head
        self.patch_len = patch_len
        self.stride = stride
        patch_num = int((seq_len - patch_len)/stride + 1)
        if padding_patch == 'end' and (seq_len - patch_len) % stride != 0:
            patch_num += 1
        head_nf = d_model * patch_num
        n_points = patch_num // 2 
        self.head = Flatten_Head(individual=individual, n_vars=nvars, nf=head_nf, target_window=pred_len, head_dropout=head_dropout)

        # glb tokens
        self.glb_tokens = nn.Parameter(torch.randn(1, nvars, d_model))
        self.global_proj = nn.Linear(d_model, pred_len)
        self.global_dropout = nn.Dropout(dropout)

        # Cross generate
        self.glb_relate = GlobalRelate(d_model, nvars, d_ff=d_ff, bias=bias, activation=act, dropout=dropout)

        for i in range(0, n_blocks):
            # set patch_len and patch
            self.encode_block_list.append(EncoderBlock(d_model, n_heads, use_timestamp=use_timestamp,
                                                    topk=topk, n_points=n_points, length=patch_num, bias=bias,
                                                    d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn))
            self.global_block_list.append(GlobalBlock(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout, activation=act))
            # pred heads
    
    
    def forward(self, x, x_mark) -> Tensor:                                      # x: [bs x nvars x patch_num x d_model]
        
        if self.revin: 
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)
        
        x_glb = self.glb_tokens.repeat((x.shape[0], 1, 1)) #bs x nvars x d_model
        x_pred = None
        x = self.embedding_layer(x) # x_i: bs x nvars x patch_num_i x d_model
        
        for i in range(self.n_blocks):
            # sample cross
            glb_scores = self.glb_relate(x, x_glb)
            # attn between patch
            x = self.encode_block_list[i](x, x_glb, glb_scores, timestamp=None)
            # attn for glb token
            x_glb = self.global_block_list[i](x, x_glb)
            
        x_pred = self.head(x)
        # get glb prediction
        x_glb_pred = self.global_proj(x_glb)
        x_glb_pred = self.global_dropout(x_glb_pred)
        
        x = x_pred  + x_glb_pred

        if self.revin: 
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0,2,1)
        
        return x
   

class GlobalBlock(nn.Module):
    '''
    x [bs x channel x length x d_model]
    x_glb [bs x channel x d_model]
    '''
    def __init__(self, d_model, n_heads, norm='LayerNorm', 
                 attn_dropout=0., dropout=0., d_ff=None, activation='gelu', bias=True):
        super().__init__()
        self.cross_attn = MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout, scores_only=False)
        self.self_attn = MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout, scores_only=False)
        # norm
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm_ff = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn1 = nn.LayerNorm(d_model)
            self.norm_attn2 = nn.LayerNorm(d_model)
            self.norm_ff = nn.LayerNorm(d_model)
        
        # dropout
        self.dropout_attn1 = nn.Dropout(dropout)
        self.dropout_attn2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

    def forward(self, x, x_glb):
        B, C, d_model = x_glb.shape
        x_attn = x
        # reshape for attn
        x_attn = rearrange(x_attn, 'b c l d -> (b c) l d', b=B)
        x_glb = rearrange(x_glb, 'b c d -> (b c) d', b=B)
        x_glb = x_glb.unsqueeze(1)
        # attn and norm
        x_glb_attn1, _, _ = self.cross_attn(x_glb, x_attn, x_attn)
        x_glb = x_glb + self.dropout_attn1(x_glb_attn1)
        x_glb = self.norm_attn1(x_glb)  

        x_glb_attn2,_,_ = self.self_attn(x_glb, x_glb, x_glb)
        x_glb = x_glb + self.dropout_attn2(x_glb_attn2)
        x_glb = self.norm_attn2(x_glb)

        # ffn and norm
        x_glb_ff = self.ff(x_glb)
        x_glb = x_glb + self.dropout_ff(x_glb_ff)
        x_glb = self.norm_ff(x_glb)
        # reshape for output
        x_glb = x_glb.squeeze(1)
        x_glb = rearrange(x_glb, '(b c) d -> b c d', b=B, c=C)
        return x_glb
       

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_timestamp=False, d_k=None, d_v=None, d_ff=None,
                        topk=4, n_points=1, length=1, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., bias=True, activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super(EncoderBlock, self).__init__()
        
        self.get_cross = SelectCross(topk=topk, n_points=n_points, d_model=d_model, length=length, 
                                          d_ff=d_ff, bias=bias, activation=activation, use_timestamp=use_timestamp)
        
        self.time_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, use_timestamp=use_timestamp, topk=topk, 
                                                      d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      bias=bias, activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn)for i in range(n_layers)])

        self.res_attention = res_attention
        self.n_layers = n_layers

    def forward(self, x:Tensor, x_glb:Tensor, glb_scores:Tensor,
                timestamp:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        # x: [bs x nvars x patch_num x d_model]
        # patch merge
        cross = self.get_cross(x, x_glb, glb_scores, timestamp) # cross: [b x channel x topk x n_points x d_model]
        output = x
        scores = None
        for i in range(self.n_layers):
            if self.res_attention:
                # channel scores: bs x n_heads x nvars x nvars
                results = self.time_layers[i](output, timestamp=timestamp, cross=cross, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                results = self.time_layers[i](output, timestamp=timestamp, cross=cross,key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            scores = [results[1], results[2]]
            output = results[0]
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, use_timestamp=False, topk=1, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        # Multi-Head attention
        self.res_attention = res_attention
        

        self.self_attn = MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout, scores_only=False)
        self.cross_attn = BroadAttention(d_model, n_heads, topk=topk, attn_dropout=attn_dropout, proj_dropout=dropout)

        # Add & Norm
        self.dropout_attn1 = nn.Dropout(dropout)
        self.dropout_attn2 = nn.Dropout(dropout)

        if "batch" in norm.lower():
            self.norm_attn1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm_attn2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn1 = nn.LayerNorm(d_model)
            self.norm_attn2 = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, timestamp:Optional[Tensor]=None, cross:Optional[Tensor]=None, prev:Optional[List[Tensor]]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        '''
        input/output shape: 
        src [bs x channel x length x d_model]
        x_glb [bs x channel x d_model]
        '''
        
        batch_size, nvars, _, _ = src.shape

        # Self attn
        scores1=None
        prev_score1 = prev[0] if prev is not None else None
        src = rearrange(src, 'b nvars patch_num d_model -> (b nvars) patch_num d_model', b = batch_size)
        src1, attn, scores1 = self.self_attn(src, src, src, prev=prev_score1, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        # Self Norm and add
        src = src + self.dropout_attn1(src1)
        src = self.norm_attn1(src)
        src = rearrange(src, '(b nvars) patch_num d_model -> b nvars patch_num d_model', b = batch_size, nvars = nvars)
        
        # cross attn
        src = rearrange(src, 'b nvars patch_num d_model -> (b nvars patch_num) d_model', b = batch_size)
        src = src.unsqueeze(1) # B x 1 x d
        cross = rearrange(cross, 'b c patch_num k l d -> (b c patch_num) k l d', b = batch_size)
        # Multi-Head attention
        scores2 = None
        prev_score2 = prev[1] if prev is not None else None
        src2, attn, scores2 = self.cross_attn(src, cross, cross, prev=prev_score2, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        if self.store_attn:
            self.attn = attn
        
        ## Cross Add & Norm
        src = src + self.dropout_attn2(src2) # Add: residual connection with residual dropout
        src = self.norm_attn2(src)


        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.norm_ffn(src)
        
        src = src.squeeze(1)
        src = rearrange(src, '(b nvars patch_num)  d_model -> b nvars patch_num d_model', b = batch_size, nvars = nvars)

        return [src, scores1, scores2]
        