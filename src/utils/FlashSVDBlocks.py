import os
import sys
import time
import itertools
import torch
import torch.nn as nn
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
# from evaluate import load as load_metric
from typing import Callable, Tuple
import math
import torch.nn.functional as F

import functools

# we need to access this directory first
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# M7: Use kernel_api wrapper instead of direct kernel imports
from src.utils.kernel_api import (
    call_flash_svd_attention,
    call_flashsvd_ffn_v1,
    call_flashsvd_ffn_v2,
    call_flashsvd_ffn,
)
from src.utils.svd_helpers import build_plain_svd_helpers



class BertFlashSVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn, rank_ff, svd_per_head, svd_low_rank, rank_wo):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        
        # factor Q/K/V
        WqT, bq = cfg.query.weight.data.t(), cfg.query.bias.data.view(1,H,1,dh)
        WkT, bk = cfg.key.weight.data.t(),   cfg.key.bias.data.view(1,H,1,dh)
        WvT, bv = cfg.value.weight.data.t(), cfg.value.bias.data.view(1,H,1,dh)
        self.Pq, self.Vq = map(nn.Parameter, svd_per_head(WqT, rank_attn))
        self.Pk, self.Vk = map(nn.Parameter, svd_per_head(WkT, rank_attn))
        self.Pv, self.Vv = map(nn.Parameter, svd_per_head(WvT, rank_attn))
        self.bq, self.bk, self.bv = map(nn.Parameter, (bq,bk,bv))

        # factor FFN
        Wi, bi   = hf_layer.intermediate.dense.weight.data.t(), hf_layer.intermediate.dense.bias.data
        WoT, bo2 = hf_layer.output.dense.weight.data.t(),      hf_layer.output.dense.bias.data
        self.U1, self.V1 = map(nn.Parameter, svd_low_rank(Wi,   rank_ff))
        self.U2, self.V2 = map(nn.Parameter, svd_low_rank(WoT, rank_ff))
        self.b1, self.b2 = map(nn.Parameter, (bi, bo2))

        # output projection (attn)
        Wo_full  = hf_layer.attention.output.dense.weight.data
        bo_attn  = hf_layer.attention.output.dense.bias.data
        self.Uo, self.Vo = map(nn.Parameter, svd_low_rank(Wo_full.t(), rank_wo))
        self.bo_attn    = nn.Parameter(bo_attn)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H, R     = self.Pq.shape[0], self.Pq.shape[2]
        dh       = dm // H

        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq)
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk)
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv)

        Vq_full = self.Vq.expand(B,H,R,dh).contiguous()
        Vk_full = self.Vk.expand(B,H,R,dh).contiguous()
        Vv_full = self.Vv.expand(B,H,R,dh).contiguous()
        bq_full = self.bq.expand(B,H,1,dh).squeeze(2)
        bk_full = self.bk.expand(B,H,1,dh).squeeze(2)
        bv_full = self.bv.expand(B,H,1,dh).squeeze(2)

        mask4 = mask.view(B,1,1,M) if mask is not None else None
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()
        
        attn = attn_out.view(B,H,M,dh).transpose(1,2).reshape(B,M,dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn) # Q: what is the memory complexity of this one?

        mid = x1 @ self.U1 
        # flashsvdffn v2
        #y   = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        
        # flashsvdffn v1
        y = flashsvd_ffn_v1(
            mid,       # P
            self.V1,   # V1
            self.U2,   # U2
            self.V2,   # V2
            self.b1,   # b1
            self.b2    # b2
            # you can optionally override BL, BD, BR1, BR2 here
            # e.g. , BL=64, BD=128, BR1=32, BR2=32
        )
        
        # # If we do not use the below inference, it will slow down the process
        # midV = mid @ self.V1
        # midA = F.gelu(midV + self.b1)
        # y    = (midA @ self.U2) @ self.V2 + self.b2
        
        out = self.ln2(x1 + y)
        return out 





################### FlashFWSVD BERT ####################
class BertFlashFWSVDBlock(nn.Module):
    """
    M7 Phase 2.x: Added build_only mode support for v2 loader.
    """
    def __init__(self, hf_layer, rank_attn: int, rank_ff: int,
                 fwsvd_per_head: Callable, fwsvd_low_rank:  Callable, rank_wo: int=768,
                 build_only: bool = False):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features
        rank_wo = rank_wo or rank_attn  # default same as attention rank

        # M7 Phase 2.x: Handle build_only mode
        if build_only:
            # SKIP FWSVD decomposition - only create parameter shapes
            print("  [BertFWSVDBlock] build_only=True: SKIP decomposition, creating empty parameters")

            # Q/K/V parameters (per-head)
            self.Pq = nn.Parameter(torch.zeros(1, H, d_model, rank_attn))
            self.Vq = nn.Parameter(torch.zeros(1, H, rank_attn, dh))
            self.Pk = nn.Parameter(torch.zeros(1, H, d_model, rank_attn))
            self.Vk = nn.Parameter(torch.zeros(1, H, rank_attn, dh))
            self.Pv = nn.Parameter(torch.zeros(1, H, d_model, rank_attn))
            self.Vv = nn.Parameter(torch.zeros(1, H, rank_attn, dh))
            self.bq = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bk = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bv = nn.Parameter(torch.zeros(1, H, 1, dh))

            # Attn output projection
            self.Uo = nn.Parameter(torch.zeros(d_model, rank_wo))
            self.Vo = nn.Parameter(torch.zeros(rank_wo, d_model))
            self.bo_attn = nn.Parameter(torch.zeros(d_model))

            # FFN parameters
            self.U1 = nn.Parameter(torch.zeros(d_model, rank_ff))
            self.V1 = nn.Parameter(torch.zeros(rank_ff, d_ff))
            self.b1 = nn.Parameter(torch.zeros(d_ff))
            self.U2 = nn.Parameter(torch.zeros(d_ff, rank_ff))
            self.V2 = nn.Parameter(torch.zeros(rank_ff, d_model))
            self.b2 = nn.Parameter(torch.zeros(d_model))
        else:
            # NORMAL MODE: Perform FWSVD decomposition
            # ——— 1) factor Q/K/V ———
            WqT = hf_layer.attention.self.query.weight.data.t()
            WkT = hf_layer.attention.self.key.weight.data.t()
            WvT = hf_layer.attention.self.value.weight.data.t()
            bq  = hf_layer.attention.self.query.bias.data.view(1,H,1,dh)
            bk  = hf_layer.attention.self.key.bias.data.view(1,H,1,dh)
            bv  = hf_layer.attention.self.value.bias.data.view(1,H,1,dh)

            Uq,Vq = fwsvd_per_head(WqT, rank_attn)
            Uk,Vk = fwsvd_per_head(WkT, rank_attn)
            Uv,Vv = fwsvd_per_head(WvT, rank_attn)

            # ——— 2) factor Wᵢ and Wₒ for FFN ———
            Wi   = hf_layer.intermediate.dense.weight.data.t()   # [dm,d_ff]
            bi   = hf_layer.intermediate.dense.bias.data
            WoT  = hf_layer.output.dense.weight.data.t()         # [d_ff,dm]
            bo2  = hf_layer.output.dense.bias.data

            U1,V1 = fwsvd_low_rank(Wi,    rank_ff)
            U2,V2 = fwsvd_low_rank(WoT,   rank_ff)

            # ——— 3) factor attention-output projection Wₒ ———
            Wo_full = hf_layer.attention.output.dense.weight.data  # [dm,dm]
            bo_attn = hf_layer.attention.output.dense.bias.data
            Uo, Vo = fwsvd_low_rank(Wo_full.t(), rank_wo)

            # ——— stash everything ———
            self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
            self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
            self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

            self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
            self.U1, self.V1, self.b1    = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
            self.U2, self.V2, self.b2    = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    def forward(self, x, mask=None):  # x: [B,M,dm]
        # M7 Phase 2.x: Import kernel functions for FWSVD inference
        from src.kernels.flashsvdattn import flash_svd_attention
        from src.kernels.flashsvdffnv1 import flashsvd_ffn_v1

        B, M, dm = x.shape
        H, R = self.Pq.shape[1], self.Pq.shape[-1]
        dh   = dm // H
        scale = 1.0 / math.sqrt(dh)

        # ——— 1) project into low-rank Q/K/V ———
        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq[0]).contiguous()
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk[0]).contiguous()
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv[0]).contiguous()

        # expand V and biases
        Vq_full = self.Vq.expand(B, H, R, dh).contiguous()
        Vk_full = self.Vk.expand(B, H, R, dh).contiguous()
        Vv_full = self.Vv.expand(B, H, R, dh).contiguous()
        bq_full = self.bq.expand(B, H, 1, dh).contiguous().squeeze(2)
        bk_full = self.bk.expand(B, H, 1, dh).contiguous().squeeze(2)
        bv_full = self.bv.expand(B, H, 1, dh).contiguous().squeeze(2)

        # flash-SVD attention
        mask4 = mask.view(B,1,1,M) if mask is not None else None
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4,
            block_m=32,
            block_r=R,
        )  # [B,H,M,dh]
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()

        attn = attn_out.view(B, H, M, dh).transpose(1,2).reshape(B,M,dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn) # Q: what is the memory complexity of this one?

        # ——— flash-SVD FFN ———
        mid = x1 @ self.U1              # [B,M,rank_ff]                             # [B,M,dm]
        # y   = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        
        # flashsvdffn v1
        y = flashsvd_ffn_v1(
            mid,       # P
            self.V1,   # V1
            self.U2,   # U2
            self.V2,   # V2
            self.b1,   # b1
            self.b2    # b2
            # you can optionally override BL, BD, BR1, BR2 here
            # e.g. , BL=64, BD=128, BR1=32, BR2=32
        )
        
        # If we do not use the below inference, it will slow down the process
        # midV = mid @ self.V1
        # midA = F.gelu(midV + self.b1)
        # y    = (midA @ self.U2) @ self.V2 + self.b2
        
        return self.ln2(x1 + y)












######################## RoBERTA FlashSVD Blocks #########################
class RobertaFlashSVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn, rank_ff, svd_per_head, svd_low_rank, rank_wo):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        
        # factor Q/K/V
        WqT, bq = cfg.query.weight.data.t(), cfg.query.bias.data.view(1,H,1,dh)
        WkT, bk = cfg.key.weight.data.t(),   cfg.key.bias.data.view(1,H,1,dh)
        WvT, bv = cfg.value.weight.data.t(), cfg.value.bias.data.view(1,H,1,dh)
        self.Pq, self.Vq = map(nn.Parameter, svd_per_head(WqT, rank_attn))
        self.Pk, self.Vk = map(nn.Parameter, svd_per_head(WkT, rank_attn))
        self.Pv, self.Vv = map(nn.Parameter, svd_per_head(WvT, rank_attn))
        self.bq, self.bk, self.bv = map(nn.Parameter, (bq,bk,bv))

        # factor FFN
        Wi, bi   = hf_layer.intermediate.dense.weight.data.t(), hf_layer.intermediate.dense.bias.data
        WoT, bo2 = hf_layer.output.dense.weight.data.t(),      hf_layer.output.dense.bias.data
        self.U1, self.V1 = map(nn.Parameter, svd_low_rank(Wi,   rank_ff))
        self.U2, self.V2 = map(nn.Parameter, svd_low_rank(WoT, rank_ff))
        self.b1, self.b2 = map(nn.Parameter, (bi, bo2))

        # output projection (attn)
        Wo_full  = hf_layer.attention.output.dense.weight.data
        bo_attn  = hf_layer.attention.output.dense.bias.data
        self.Uo, self.Vo = map(nn.Parameter, svd_low_rank(Wo_full.t(), rank_wo))
        self.bo_attn    = nn.Parameter(bo_attn)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H, R     = self.Pq.shape[0], self.Pq.shape[2]
        dh       = dm // H

        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq)
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk)
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv)

        Vq_full = self.Vq.expand(B,H,R,dh).contiguous()
        Vk_full = self.Vk.expand(B,H,R,dh).contiguous()
        Vv_full = self.Vv.expand(B,H,R,dh).contiguous()
        bq_full = self.bq.expand(B,H,1,dh).squeeze(2)
        bk_full = self.bk.expand(B,H,1,dh).squeeze(2)
        bv_full = self.bv.expand(B,H,1,dh).squeeze(2)

        mask4 = mask.view(B,1,1,M) if mask is not None else None
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()

        attn = attn_out.view(B,H,M,dh).transpose(1,2).reshape(B,M,dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn) # Q: what is the memory complexity of this one?

        mid = x1 @ self.U1 
        #y   = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        
        # flashsvdffn v1
        y = flashsvd_ffn_v1(
            mid,       # P
            self.V1,   # V1
            self.U2,   # U2
            self.V2,   # V2
            self.b1,   # b1
            self.b2    # b2
            # you can optionally override BL, BD, BR1, BR2 here
            # e.g. , BL=64, BD=128, BR1=32, BR2=32
        )
        
        # # If we do not use the below inference, it will slow down the process
        # midV = mid @ self.V1
        # midA = F.gelu(midV + self.b1)
        # y    = (midA @ self.U2) @ self.V2 + self.b2
        
        return self.ln2(x1 + y)




######## Roberta FWSVD Blocks ########
class RobertaFlashFWSVDBlock(nn.Module):
    """
    M7 Phase 2.x: Added build_only mode support for v2 loader.
    """
    def __init__(self, hf_layer, rank_attn: int, rank_ff: int,
                 fwsvd_per_head: Callable, fwsvd_low_rank:  Callable, rank_wo: int=768,
                 build_only: bool = False):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features

        # M7 Phase 2.x: Handle build_only mode
        if build_only:
            # SKIP FWSVD decomposition - only create parameter shapes
            print("  [RobertaFWSVDBlock] build_only=True: SKIP decomposition, creating empty parameters")

            # Q/K/V parameters (per-head)
            self.Pq = nn.Parameter(torch.zeros(1, H, d_model, rank_attn))
            self.Vq = nn.Parameter(torch.zeros(1, H, rank_attn, dh))
            self.Pk = nn.Parameter(torch.zeros(1, H, d_model, rank_attn))
            self.Vk = nn.Parameter(torch.zeros(1, H, rank_attn, dh))
            self.Pv = nn.Parameter(torch.zeros(1, H, d_model, rank_attn))
            self.Vv = nn.Parameter(torch.zeros(1, H, rank_attn, dh))
            self.bq = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bk = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bv = nn.Parameter(torch.zeros(1, H, 1, dh))

            # Attn output projection
            self.Uo = nn.Parameter(torch.zeros(d_model, rank_wo))
            self.Vo = nn.Parameter(torch.zeros(rank_wo, d_model))
            self.bo_attn = nn.Parameter(torch.zeros(d_model))

            # FFN parameters
            self.U1 = nn.Parameter(torch.zeros(d_model, rank_ff))
            self.V1 = nn.Parameter(torch.zeros(rank_ff, d_ff))
            self.b1 = nn.Parameter(torch.zeros(d_ff))
            self.U2 = nn.Parameter(torch.zeros(d_ff, rank_ff))
            self.V2 = nn.Parameter(torch.zeros(rank_ff, d_model))
            self.b2 = nn.Parameter(torch.zeros(d_model))
        else:
            # NORMAL MODE: Perform FWSVD decomposition
            # 1) grab & factor Q/K/V_transpose
            WqT = hf_layer.attention.self.query.weight.data.t()
            WkT = hf_layer.attention.self.key.weight.data.t()
            WvT = hf_layer.attention.self.value.weight.data.t()
            bq  = hf_layer.attention.self.query.bias.data.view(1,H,1,dh)
            bk  = hf_layer.attention.self.key.bias.data.view(1,H,1,dh)
            bv  = hf_layer.attention.self.value.bias.data.view(1,H,1,dh)

            Uq,Vq = fwsvd_per_head(WqT, rank_attn)
            Uk,Vk = fwsvd_per_head(WkT, rank_attn)
            Uv,Vv = fwsvd_per_head(WvT, rank_attn)

            # 2) grab & factor FFN transpose
            Wi   = hf_layer.intermediate.dense.weight.data.t()   # [dm,d_ff]
            bi   = hf_layer.intermediate.dense.bias.data
            WoT  = hf_layer.output.dense.weight.data.t()         # [d_ff,dm]
            bo2  = hf_layer.output.dense.bias.data

            U1,V1 = fwsvd_low_rank(Wi,    rank_ff)
            U2,V2 = fwsvd_low_rank(WoT,   rank_ff)

            # 3) factor attention-output projection Wₒ
            Wo_full = hf_layer.attention.output.dense.weight.data  # [dm,dm]
            bo_attn = hf_layer.attention.output.dense.bias.data
            Uo, Vo = fwsvd_low_rank(Wo_full.t(), rank_wo)

            # stash everything
            self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
            self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
            self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

            self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
            self.U1, self.V1, self.b1= nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
            self.U2, self.V2, self.b2= nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        self.ln1, self.ln2 = hf_layer.attention.output.LayerNorm, hf_layer.output.LayerNorm

    def forward(self, x, mask=None):  # x: [B,M,dm]
        # M7 Phase 2.x: Import kernel functions for FWSVD inference
        from src.kernels.flashsvdattn import flash_svd_attention
        from src.kernels.flashsvdffnv1 import flashsvd_ffn_v1

        B, M, dm = x.shape
        H, R = self.Pq.shape[1], self.Pq.shape[-1]
        dh   = dm // H
        scale = 1.0 / math.sqrt(dh)

        # ——— 1) project into low-rank Q/K/V ———
        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq[0]).contiguous()
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk[0]).contiguous()
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv[0]).contiguous()

        # expand V and biases
        Vq_full = self.Vq.expand(B, H, R, dh).contiguous()
        Vk_full = self.Vk.expand(B, H, R, dh).contiguous()
        Vv_full = self.Vv.expand(B, H, R, dh).contiguous()
        bq_full = self.bq.expand(B, H, 1, dh).contiguous().squeeze(2)
        bk_full = self.bk.expand(B, H, 1, dh).contiguous().squeeze(2)
        bv_full = self.bv.expand(B, H, 1, dh).contiguous().squeeze(2)

        # flash-SVD attention
        mask4 = mask.view(B,1,1,M) if mask is not None else None
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4,
            block_m=32,
            block_r=R,
        )  # [B,H,M,dh]
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()

        attn = attn_out.view(B, H, M, dh).transpose(1,2).reshape(B,M,dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn) # Q: what is the memory complexity of this one?

        # ——— flash-SVD FFN ———
        mid = x1 @ self.U1              # [B,M,rank_ff]                             # [B,M,dm]
        #y   = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        
        # flashsvdffn v1
        y = flashsvd_ffn_v1(
            mid,       # P
            self.V1,   # V1
            self.U2,   # U2
            self.V2,   # V2
            self.b1,   # b1
            self.b2    # b2
            # you can optionally override BL, BD, BR1, BR2 here
            # e.g. , BL=64, BD=128, BR1=32, BR2=32
        )
        
        # If we do not use the below inference, it will slow down the process
        # midV = mid @ self.V1
        # midA = F.gelu(midV + self.b1)
        # y    = (midA @ self.U2) @ self.V2 + self.b2
        
        return self.ln2(x1 + y)








################### AdaSVD Blocks (M7) ####################
# Moved from src/flashsvd/compression/adasvd.py
# Uses per-operation ranks from adaptive rank selection

def _get_full_name(module: nn.Module) -> str:
    """Get the _ars_fullname attribute set during attach_fullnames."""
    return getattr(module, "_ars_fullname", "")


class BertAdaSVDBlock(nn.Module):
    """
    FlashSVD block using per-operation ranks from AdaSVD.
    Extracted from BERTAda/profile_flashsvd.py:138-209 and adapted for M7.

    M7 Phase 2.1: Supports build_only mode for v2 loader (no SVD decomposition).
    """

    def __init__(
        self,
        hf_layer,
        ranks_dict: dict,
        svd_per_head: Callable,
        svd_low_rank: Callable,
        ffn_kernel: str = "v1",
        build_only: bool = False,
    ):
        super().__init__()
        cfg = hf_layer.attention.self
        d_model = cfg.all_head_size
        H = cfg.num_attention_heads
        dh = d_model // H

        # Get submodules
        q_lin = hf_layer.attention.self.query
        k_lin = hf_layer.attention.self.key
        v_lin = hf_layer.attention.self.value
        o_lin = hf_layer.attention.output.dense
        i_lin = hf_layer.intermediate.dense
        o2_lin = hf_layer.output.dense

        def rk(m):
            """Get rank for a module from ranks_dict (FAIL LOUDLY if missing)."""
            name = _get_full_name(m)
            if name not in ranks_dict:
                raise KeyError(
                    f"Missing rank for '{name}' in ranks.json.\n"
                    f"This likely means:\n"
                    f"  1. ranks.json was generated for a different model\n"
                    f"  2. ranks.json is incomplete\n"
                    f"  3. Model architecture changed\n"
                    f"\nAvailable keys in ranks.json: {len(ranks_dict.keys())}\n"
                    f"First few keys: {list(ranks_dict.keys())[:5]}"
                )
            return max(1, int(ranks_dict[name]))

        # M7 Phase 2.1: build_only mode for v2 loader
        if build_only:
            # SKIP SVD decomposition - only create parameter shapes
            print("  [AdaSVD] build_only=True: SKIP decomposition, creating empty parameters")

            # Q/K/V parameters (per-head)
            rank_q, rank_k, rank_v = rk(q_lin), rk(k_lin), rk(v_lin)
            self.Pq = nn.Parameter(torch.zeros(H, d_model, rank_q))
            self.Vq = nn.Parameter(torch.zeros(H, rank_q, dh))
            self.Pk = nn.Parameter(torch.zeros(H, d_model, rank_k))
            self.Vk = nn.Parameter(torch.zeros(H, rank_k, dh))
            self.Pv = nn.Parameter(torch.zeros(H, d_model, rank_v))
            self.Vv = nn.Parameter(torch.zeros(H, rank_v, dh))
            self.bq = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bk = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bv = nn.Parameter(torch.zeros(1, H, 1, dh))

            # FFN parameters
            d_ff = i_lin.weight.shape[0]  # Intermediate dim
            rank_ffn1, rank_ffn2 = rk(i_lin), rk(o2_lin)
            self.U1 = nn.Parameter(torch.zeros(d_model, rank_ffn1))
            self.V1 = nn.Parameter(torch.zeros(rank_ffn1, d_ff))
            self.b1 = nn.Parameter(torch.zeros(d_ff))
            self.U2 = nn.Parameter(torch.zeros(d_ff, rank_ffn2))
            self.V2 = nn.Parameter(torch.zeros(rank_ffn2, d_model))
            self.b2 = nn.Parameter(torch.zeros(d_model))

            # Attn output projection
            rank_o = rk(o_lin)
            self.Uo = nn.Parameter(torch.zeros(d_model, rank_o))
            self.Vo = nn.Parameter(torch.zeros(rank_o, d_model))
            self.bo_attn = nn.Parameter(torch.zeros(d_model))

        else:
            # NORMAL MODE: Perform SVD decomposition
            # Q/K/V factorization per head
            WqT, bq = q_lin.weight.data.t(), q_lin.bias.data.view(1, H, 1, dh)
            WkT, bk = k_lin.weight.data.t(), k_lin.bias.data.view(1, H, 1, dh)
            WvT, bv = v_lin.weight.data.t(), v_lin.bias.data.view(1, H, 1, dh)

            Uq, Vq = svd_per_head(WqT, rk(q_lin))
            Uk, Vk = svd_per_head(WkT, rk(k_lin))
            Uv, Vv = svd_per_head(WvT, rk(v_lin))

            self.Pq, self.Vq = map(nn.Parameter, (Uq, Vq))  # [H,dm,k], [H,k,dh]
            self.Pk, self.Vk = map(nn.Parameter, (Uk, Vk))
            self.Pv, self.Vv = map(nn.Parameter, (Uv, Vv))
            self.bq, self.bk, self.bv = map(nn.Parameter, (bq, bk, bv))

            # FFN
            Wi, bi = i_lin.weight.data.t(), i_lin.bias.data
            WoT, bo2 = o2_lin.weight.data.t(), o2_lin.bias.data
            U1, V1 = svd_low_rank(Wi, rk(i_lin))
            U2, V2 = svd_low_rank(WoT, rk(o2_lin))
            self.U1, self.V1, self.b1 = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
            self.U2, self.V2, self.b2 = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

            # Attn output projection
            Wo_full = o_lin.weight.data
            bo_attn = o_lin.bias.data
            Uo, Vo = svd_low_rank(Wo_full.t(), rk(o_lin))
            self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

        # FFN kernel selection (v1 or v2)
        assert ffn_kernel in ("v1", "v2"), f"ffn_kernel must be 'v1' or 'v2', got {ffn_kernel}"
        self.ffn_kernel = ffn_kernel

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H, R = self.Pq.shape[0], self.Pq.shape[2]
        dh = dm // H

        # Project into rank-R head spaces (following BERTAda/profile_flashsvd.py:197-200)
        tmp_q = torch.einsum("bmd,hdr->bhmr", x, self.Pq)
        tmp_k = torch.einsum("bmd,hdr->bhmr", x, self.Pk)
        tmp_v = torch.einsum("bmd,hdr->bhmr", x, self.Pv)

        # Expand V/bias for kernel (following BERTAda/profile_flashsvd.py:202-207)
        Vq_full = self.Vq.expand(B, H, R, dh).contiguous()
        Vk_full = self.Vk.expand(B, H, R, dh).contiguous()
        Vv_full = self.Vv.expand(B, H, R, dh).contiguous()
        bq_full = self.bq.expand(B, H, 1, dh).squeeze(2)
        bk_full = self.bk.expand(B, H, 1, dh).squeeze(2)
        bv_full = self.bv.expand(B, H, 1, dh).squeeze(2)

        mask4 = mask.view(B, 1, 1, M) if mask is not None else None

        # Call kernel directly (same as other blocks in this file)
        from src.kernels.flashsvdattn import flash_svd_attention
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()

        # Attention output projection
        attn = attn_out.view(B, H, M, dh).transpose(1, 2).reshape(B, M, dm)
        x1 = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # FFN (kernel selection)
        mid = x1 @ self.U1
        if self.ffn_kernel == "v1":
            from src.kernels.flashsvdffnv1 import flashsvd_ffn_v1
            y = flashsvd_ffn_v1(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        else:  # v2
            from src.kernels.flashsvdffnv2 import flashsvd_ffn
            y = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)

        out = self.ln2(x1 + y)
        return out


class RobertaAdaSVDBlock(nn.Module):
    """
    FlashSVD block for RoBERTa using per-operation ranks from AdaSVD.
    Same as BertAdaSVDBlock but for RoBERTa architecture.

    M7 Phase 2.1: Supports build_only mode for v2 loader (no SVD decomposition).
    """

    def __init__(
        self,
        hf_layer,
        ranks_dict: dict,
        svd_per_head: Callable,
        svd_low_rank: Callable,
        ffn_kernel: str = "v1",
        build_only: bool = False,
    ):
        super().__init__()
        cfg = hf_layer.attention.self
        d_model = cfg.all_head_size
        H = cfg.num_attention_heads
        dh = d_model // H

        # Get submodules (same structure as BERT)
        q_lin = hf_layer.attention.self.query
        k_lin = hf_layer.attention.self.key
        v_lin = hf_layer.attention.self.value
        o_lin = hf_layer.attention.output.dense
        i_lin = hf_layer.intermediate.dense
        o2_lin = hf_layer.output.dense

        def rk(m):
            """Get rank for a module from ranks_dict (FAIL LOUDLY if missing)."""
            name = _get_full_name(m)
            if name not in ranks_dict:
                raise KeyError(
                    f"Missing rank for '{name}' in ranks.json.\n"
                    f"This likely means:\n"
                    f"  1. ranks.json was generated for a different model\n"
                    f"  2. ranks.json is incomplete\n"
                    f"  3. Model architecture changed\n"
                    f"\nAvailable keys in ranks.json: {len(ranks_dict.keys())}\n"
                    f"First few keys: {list(ranks_dict.keys())[:5]}"
                )
            return max(1, int(ranks_dict[name]))

        # M7 Phase 2.1: build_only mode for v2 loader
        if build_only:
            # SKIP SVD decomposition - only create parameter shapes
            print("  [RobertaAdaSVD] build_only=True: SKIP decomposition, creating empty parameters")

            # Q/K/V parameters (per-head)
            rank_q, rank_k, rank_v = rk(q_lin), rk(k_lin), rk(v_lin)
            self.Pq = nn.Parameter(torch.zeros(H, d_model, rank_q))
            self.Vq = nn.Parameter(torch.zeros(H, rank_q, dh))
            self.Pk = nn.Parameter(torch.zeros(H, d_model, rank_k))
            self.Vk = nn.Parameter(torch.zeros(H, rank_k, dh))
            self.Pv = nn.Parameter(torch.zeros(H, d_model, rank_v))
            self.Vv = nn.Parameter(torch.zeros(H, rank_v, dh))
            self.bq = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bk = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bv = nn.Parameter(torch.zeros(1, H, 1, dh))

            # FFN parameters
            d_ff = i_lin.weight.shape[0]  # Intermediate dim
            rank_ffn1, rank_ffn2 = rk(i_lin), rk(o2_lin)
            self.U1 = nn.Parameter(torch.zeros(d_model, rank_ffn1))
            self.V1 = nn.Parameter(torch.zeros(rank_ffn1, d_ff))
            self.b1 = nn.Parameter(torch.zeros(d_ff))
            self.U2 = nn.Parameter(torch.zeros(d_ff, rank_ffn2))
            self.V2 = nn.Parameter(torch.zeros(rank_ffn2, d_model))
            self.b2 = nn.Parameter(torch.zeros(d_model))

            # Attn output projection
            rank_o = rk(o_lin)
            self.Uo = nn.Parameter(torch.zeros(d_model, rank_o))
            self.Vo = nn.Parameter(torch.zeros(rank_o, d_model))
            self.bo_attn = nn.Parameter(torch.zeros(d_model))

        else:
            # NORMAL MODE: Perform SVD decomposition
            # Q/K/V factorization per head
            WqT, bq = q_lin.weight.data.t(), q_lin.bias.data.view(1, H, 1, dh)
            WkT, bk = k_lin.weight.data.t(), k_lin.bias.data.view(1, H, 1, dh)
            WvT, bv = v_lin.weight.data.t(), v_lin.bias.data.view(1, H, 1, dh)

            Uq, Vq = svd_per_head(WqT, rk(q_lin))
            Uk, Vk = svd_per_head(WkT, rk(k_lin))
            Uv, Vv = svd_per_head(WvT, rk(v_lin))

            self.Pq, self.Vq = map(nn.Parameter, (Uq, Vq))
            self.Pk, self.Vk = map(nn.Parameter, (Uk, Vk))
            self.Pv, self.Vv = map(nn.Parameter, (Uv, Vv))
            self.bq, self.bk, self.bv = map(nn.Parameter, (bq, bk, bv))

            # FFN
            Wi, bi = i_lin.weight.data.t(), i_lin.bias.data
            WoT, bo2 = o2_lin.weight.data.t(), o2_lin.bias.data
            U1, V1 = svd_low_rank(Wi, rk(i_lin))
            U2, V2 = svd_low_rank(WoT, rk(o2_lin))
            self.U1, self.V1, self.b1 = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
            self.U2, self.V2, self.b2 = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

            # Attn output projection
            Wo_full = o_lin.weight.data
            bo_attn = o_lin.bias.data
            Uo, Vo = svd_low_rank(Wo_full.t(), rk(o_lin))
            self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

        # FFN kernel selection (v1 or v2)
        assert ffn_kernel in ("v1", "v2"), f"ffn_kernel must be 'v1' or 'v2', got {ffn_kernel}"
        self.ffn_kernel = ffn_kernel

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H, R = self.Pq.shape[0], self.Pq.shape[2]
        dh = dm // H

        # Project into rank-R head spaces
        tmp_q = torch.einsum("bmd,hdr->bhmr", x, self.Pq)
        tmp_k = torch.einsum("bmd,hdr->bhmr", x, self.Pk)
        tmp_v = torch.einsum("bmd,hdr->bhmr", x, self.Pv)

        # Expand V/bias for kernel
        Vq_full = self.Vq.expand(B, H, R, dh).contiguous()
        Vk_full = self.Vk.expand(B, H, R, dh).contiguous()
        Vv_full = self.Vv.expand(B, H, R, dh).contiguous()
        bq_full = self.bq.expand(B, H, 1, dh).squeeze(2)
        bk_full = self.bk.expand(B, H, 1, dh).squeeze(2)
        bv_full = self.bv.expand(B, H, 1, dh).squeeze(2)

        mask4 = mask.view(B, 1, 1, M) if mask is not None else None

        # Call kernel directly (same as other blocks in this file)
        from src.kernels.flashsvdattn import flash_svd_attention
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()

        # Attention output projection
        attn = attn_out.view(B, H, M, dh).transpose(1, 2).reshape(B, M, dm)
        x1 = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # FFN (kernel selection)
        mid = x1 @ self.U1
        if self.ffn_kernel == "v1":
            from src.kernels.flashsvdffnv1 import flashsvd_ffn_v1
            y = flashsvd_ffn_v1(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        else:  # v2
            from src.kernels.flashsvdffnv2 import flashsvd_ffn
            y = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)

        out = self.ln2(x1 + y)
        return out


################### Whiten/DRONE Blocks (M7) ####################
# Moved from src/flashsvd/compression/whiten.py
# Uses DRONE-style data-aware low-rank factorization with input covariances

class BertWhitenSVDBlock(nn.Module):
    """
    Flash-attention + Flash-FFN block using DRONE-style data-aware low-rank.
    Extracted from BERTWhiten/profile_flashsvd.py:122-209 and adapted for M7.

    M7 Phase 2.x: Added build_only mode support for v2 loader.
    """

    def __init__(
        self,
        hf_layer,
        rank_attn: int,
        rank_ff: int,
        rank_wo: int,
        cov_attn_in: torch.Tensor = None,
        cov_attn_out: torch.Tensor = None,
        cov_ffn_in: torch.Tensor = None,
        cov_ffn_out: torch.Tensor = None,
        data_aware_per_head: Callable = None,
        data_aware_low_rank: Callable = None,
        build_only: bool = False,
    ):
        super().__init__()
        cfg = hf_layer.attention.self
        d_model = cfg.all_head_size
        H = cfg.num_attention_heads
        dh = d_model // H
        d_ff = hf_layer.intermediate.dense.out_features

        # M7 Phase 2.x: Handle build_only mode
        if build_only:
            # SKIP DRONE decomposition - only create parameter shapes
            print("  [BertWhiten] build_only=True: SKIP decomposition, creating empty parameters")

            # Q/K/V parameters (per-head)
            self.Pq = nn.Parameter(torch.zeros(H, d_model, rank_attn))
            self.Vq = nn.Parameter(torch.zeros(H, rank_attn, dh))
            self.Pk = nn.Parameter(torch.zeros(H, d_model, rank_attn))
            self.Vk = nn.Parameter(torch.zeros(H, rank_attn, dh))
            self.Pv = nn.Parameter(torch.zeros(H, d_model, rank_attn))
            self.Vv = nn.Parameter(torch.zeros(H, rank_attn, dh))
            self.bq = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bk = nn.Parameter(torch.zeros(1, H, 1, dh))
            self.bv = nn.Parameter(torch.zeros(1, H, 1, dh))

            # Attn output projection
            self.Uo = nn.Parameter(torch.zeros(d_model, rank_wo))
            self.Vo = nn.Parameter(torch.zeros(rank_wo, d_model))
            self.bo_attn = nn.Parameter(torch.zeros(d_model))

            # FFN parameters
            self.U1 = nn.Parameter(torch.zeros(d_model, rank_ff))
            self.V1 = nn.Parameter(torch.zeros(rank_ff, d_ff))
            self.b1 = nn.Parameter(torch.zeros(d_ff))
            self.U2 = nn.Parameter(torch.zeros(d_ff, rank_ff))
            self.V2 = nn.Parameter(torch.zeros(rank_ff, d_model))
            self.b2 = nn.Parameter(torch.zeros(d_model))
        else:
            # NORMAL MODE: Perform DRONE-style decomposition
            # --- Attention Q/K/V (per-head data-aware) ---
            WqT, bq = cfg.query.weight.data.t(), cfg.query.bias.data.view(1, H, 1, dh)
            WkT, bk = cfg.key.weight.data.t(), cfg.key.bias.data.view(1, H, 1, dh)
            WvT, bv = cfg.value.weight.data.t(), cfg.value.bias.data.view(1, H, 1, dh)

            Uq, Vq = data_aware_per_head(WqT, rank_attn, cov_attn_in, H)
            Uk, Vk = data_aware_per_head(WkT, rank_attn, cov_attn_in, H)
            Uv, Vv = data_aware_per_head(WvT, rank_attn, cov_attn_in, H)

            self.Pq, self.Vq = nn.Parameter(Uq), nn.Parameter(Vq)  # [H,dm,R], [H,R,dh]
            self.Pk, self.Vk = nn.Parameter(Uk), nn.Parameter(Vk)
            self.Pv, self.Vv = nn.Parameter(Uv), nn.Parameter(Vv)
            self.bq, self.bk, self.bv = map(nn.Parameter, (bq, bk, bv))

            # --- FFN (data-aware) ---
            Wi, bi = hf_layer.intermediate.dense.weight.data.t(), hf_layer.intermediate.dense.bias.data
            WoT, bo2 = hf_layer.output.dense.weight.data.t(), hf_layer.output.dense.bias.data

            U1, V1 = data_aware_low_rank(Wi, rank_ff, cov_ffn_in)  # dm->d_ff
            U2, V2 = data_aware_low_rank(WoT, rank_ff, cov_ffn_out)  # d_ff->dm

            self.U1, self.V1 = nn.Parameter(U1), nn.Parameter(V1)
            self.U2, self.V2 = nn.Parameter(U2), nn.Parameter(V2)
            self.b1, self.b2 = nn.Parameter(bi), nn.Parameter(bo2)

            # --- Attention output projection Wo (data-aware) ---
            Wo_full = hf_layer.attention.output.dense.weight.data  # [dm, dm]
            bo_attn = hf_layer.attention.output.dense.bias.data
            Uo, Vo = data_aware_low_rank(Wo_full.t(), rank_wo, cov_attn_out)
            self.Uo, self.Vo = nn.Parameter(Uo), nn.Parameter(Vo)
            self.bo_attn = nn.Parameter(bo_attn)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H, R = self.Pq.shape[0], self.Pq.shape[2]
        dh = dm // H

        # Project to per-head rank-R space (following BERTWhiten:179-181)
        tmp_q = torch.einsum("bmd,hdr->bhmr", x, self.Pq)
        tmp_k = torch.einsum("bmd,hdr->bhmr", x, self.Pk)
        tmp_v = torch.einsum("bmd,hdr->bhmr", x, self.Pv)

        # Expand V/bias for Flash SVD attention kernel (following BERTWhiten:184-189)
        Vq_full = self.Vq.expand(B, H, R, dh).contiguous()
        Vk_full = self.Vk.expand(B, H, R, dh).contiguous()
        Vv_full = self.Vv.expand(B, H, R, dh).contiguous()
        bq_full = self.bq.expand(B, H, 1, dh).squeeze(2)
        bk_full = self.bk.expand(B, H, 1, dh).squeeze(2)
        bv_full = self.bv.expand(B, H, 1, dh).squeeze(2)

        mask4 = mask.view(B, 1, 1, M) if mask is not None else None

        # Call kernel directly (same signature as other blocks)
        from src.kernels.flashsvdattn import flash_svd_attention
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()

        # Attention output projection
        attn = attn_out.view(B, H, M, dh).transpose(1, 2).reshape(B, M, dm)
        x1 = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # FFN via Flash kernels
        mid = x1 @ self.U1  # [B,M,R1]
        from src.kernels.flashsvdffnv1 import flashsvd_ffn_v1
        y = flashsvd_ffn_v1(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        out = self.ln2(x1 + y)
        return out
