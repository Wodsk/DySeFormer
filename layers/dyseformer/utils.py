import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange,repeat

from typing import Optional, List
from layers.dyseformer.embedding import positional_encoding, Transpose, get_activation_fn


class Embedding_layer(nn.Module):
    
    def __init__(self, padding_patch, window, patch_len=16, stride=8, d_model=32, dropout=0, pe=None, learn_pe=None):
        super().__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.padding = False

        patch_num = int((window - patch_len)/stride + 1)
        if padding_patch == 'end' and (window - patch_len) % stride != 0: # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.padding = True
            patch_num += 1
    
        self.W_P = nn.Linear(patch_len, d_model)
    
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)  

    
    def forward(self, x):                                                       # x:[bs x nvars x window]

        # do patching
        if self.padding: 
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # x: [bs x nvars x patch_num x patch_len]
        x = x.permute(0,1,3,2)                                                              # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        # Position Encoding
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]
        
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]
        u = torch.reshape(u, (x.shape[0], n_vars, x.shape[2], x.shape[3]))       # u: [bs x nvars x patch_num x d_model]
        
        return u


class TimeEmbedding(nn.Module):
    def __init__(self, padding_patch, window, timeF=6, patch_len=16, stride=8, d_model=32):
        super().__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        patch_num = int((window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
    
        self.projection = nn.Linear(patch_len*timeF, d_model)
    
    def forward(self, x):
        # do patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # x: [bs x timeF x patch_num x patch_len]
        x = x.permute(0,1,3,2)                                                              # x: [bs x timeF x patch_len x patch_num]
        x = rearrange(x, 'b tf patch_len patch_num -> b patch_num (tf patch_len)')                                           
        x = self.projection(x)                                                              # x: [bs x patch_num x d_model]
        return x


class SelectCross(nn.Module):
    def __init__(self, length, d_model, topk=3, n_points=4, d_ff=256, bias=True, 
                 activation="gelu", dropout=0., add_self=False, use_timestamp=False):
        super().__init__()
        self.topk = topk
        self.n_points = n_points
        self.add_self = add_self
        
        self.MLP = nn.Sequential(nn.Linear(2*d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, length, bias=bias))
        

    def sample_patch(self, src, x, x_glb, channel_index):
        """
        src: [B, d_model] 
        x:      [B, C, patch_num, d_model]
        x_glb:  [B, C, d_model]
        channel_index: [B, K] related channels index
        """ 
        B, C, patch_num, d_model = x.shape
        B1, K = channel_index.shape
        assert B==B1
        # get all patches
        feat_expand_index = channel_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, patch_num, d_model)
        channel_feat = torch.gather(x, dim=1, index=feat_expand_index)
        # get glb token
        glb_expand_index = channel_index.unsqueeze(-1).expand(-1, -1, d_model)
        channel_glb = torch.gather(x_glb, dim=1, index=glb_expand_index)
        # generate scores
        src = src.unsqueeze(1).expand(-1, K, -1)
        concat_feat = torch.cat([src, channel_glb], dim=-1)
        scores = self.MLP(concat_feat)
        scores = scores.squeeze(-1) # [B, K, L]
        # select patches
        topk_scores, topk_indices = torch.topk(scores, k=self.n_points, dim=-1)  # [B, K, n_points]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, d_model)  # [B, K, n_points, D]
        channel_feat = torch.gather(channel_feat, dim=2, index=topk_indices_expanded)  # [B, K, n_points, D]
        
        return channel_feat

    def forward(self, x, x_glb, scores, timestamp=None):
        """
        x:      [B, C, patch_num, d_model]
        x_glb:  [B, C, d_model]
        scores: [B, C, patch_num, C]

        return:  cross: [B, C, patch_num, K, n_points, d_model]
        """
        B, C, patch_num, d_model = x.shape
        if self.n_points > patch_num:
            raise ValueError(f"n_points ({self.n_points}) must be less than patch_num ({patch_num})")
        
        cross_list = []

        for c in range(C):
            patch_cross_list = []
            
            for patch in range(patch_num):
                src = x[:, c, patch, :] # [B, d_model]
                weights = scores[:, c, patch, :]  # [B, C]
                weights[:, c] = float('-inf') 
               
                _, topk_idx = torch.topk(weights, self.topk, dim=-1)                  # [B, K]
                if self.add_self:
                    self_idx = torch.full((B, 1), c, dtype=torch.long, device=x.device)  # [B, 1]
                    topk_idx = torch.cat([topk_idx, self_idx], dim=1)  # [B, K+1]
                
                selected = self.sample_patch(src, x, x_glb, topk_idx)
                patch_cross_list.append(selected.unsqueeze(1))  # [B, 1, K, n_points, d_model]

            channel_cross = torch.cat(patch_cross_list, dim=1)  # [B, patch_num, K, n_points, d_model]
            cross_list.append(channel_cross.unsqueeze(1))
        cross = torch.cat(cross_list, dim=1) # [B, C, patch_num, K, n_points, d_model]
        return cross
    








    

        
        