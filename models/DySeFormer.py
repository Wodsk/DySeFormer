__all__ = ['DySeFormer']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.dyseformer.encoder import Encoder


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=False, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()

        # load parameters
        nvars = configs.enc_in
        seq_len = configs.seq_len
        pred_len = configs.pred_len

        topk = configs.topk

        n_blocks = configs.n_blocks 
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff

        patch_len = configs.patch_len
        stride = configs.stride
        use_timestamp = configs.use_timestamp
        
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
    
        padding_patch = configs.padding_patch
        
        individual = configs.individual
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
              
        self.pred_len = configs.pred_len
        norm = "LayerNorm"
        
        # model
        
        self.model = Encoder(nvars=nvars, seq_len = seq_len, pred_len=pred_len, max_seq_len=max_seq_len, 
                                patch_len=patch_len, stride=stride, n_blocks=n_blocks, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose, use_timestamp=use_timestamp, topk=topk, **kwargs)
    
    
    def forward(self, x, x_mark):           # x: [Batch, Input length, Channel], x_mark: [Batch, Input length, 7]
        
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        x = self.model(x, x_mark)
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        
        return x
    

class configs:
    def __init__(self):
        self.enc_in = 5
        self.seq_len = 96
        self.pred_len = 96
        self.n_blocks = 3
        self.patch_len = 16
        self.stride = 8
        self.win_size = 2
        self.e_layers = 1
        self.n_heads = 1
        self.d_model = 32
        self.d_ff = 256
        self.dropout = 0.1
        self.fc_dropout = 0.1
        self.head_dropout = 0.1
        self.individual = False
        self.padding_patch = 'end'
        self.revin = True
        self.affine = False
        self.subtract_last = False
        self.decomposition = 0
        self.kernel_size = 3
        self.topk = 2
        self.use_timestamp = False


if __name__ == '__main__':
    config = configs()
    x = torch.rand(2, 96, 5) # x: [Batch, Input length, Channel]
    timestamp  = torch.randn(2, 96, 6)
    self_model = Model(config)
    out = self_model(x, timestamp)
    print(out.shape)
