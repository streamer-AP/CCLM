import os
import torch
import torch.nn as nn
from .local_bl import build_loss as build_loss_local_bl
from .local_bl_v3 import build_loss as build_loss_local_bl_v3
from .local_bl_v8 import build_loss as build_loss_local_bl_v8
from .local_bl_v9 import build_loss as build_loss_local_bl_v9
from .local_bl_v10 import build_loss as build_loss_local_bl_v10
from .local_bl_v11 import build_loss as build_loss_local_bl_v11
from .local_bl_v12 import build_loss as build_loss_local_bl_v12
from .local_bl_v13 import build_loss as build_loss_local_bl_v13
from .local_bl_v14 import build_loss as build_loss_local_bl_v14
from .local_ot import build_loss as build_loss_local_ot
from .local_bl_v12_gauss import build_loss as build_loss_local_bl_v12_gauss
from .local_bl_v12_var import build_loss as build_loss_local_bl_v12_var
from .iis import build_loss as build_loss_iis
from .divden import build_loss as build_loss_divden
from .rank_loss import build_loss as build_loss_rank
from .rank_loss_v2 import build_loss as build_loss_rank_v2
from .rank_loss_v3 import build_loss as build_loss_rank_v3
from .rank_loss_v4 import build_loss as build_loss_rank_v4
from .rank_loss_v2_bg import build_loss as build_loss_rank_v2_bg
from .rank_loss_v2_2 import build_loss as build_loss_rank_v2_2
from .rank_loss_v2_3 import build_loss as build_loss_rank_v2_3
from .rank_loss_v2_4 import build_loss as build_loss_rank_v2_4
from .rank_loss_v2_5 import build_loss as build_loss_rank_v2_5
from .rank_loss_v2_6 import build_loss as build_loss_rank_v2_6
from .rank_loss_v2_2_dense import build_loss as build_loss_rank_v2_2_dense
from .rank_loss_v2_2_auto_w import build_loss as build_loss_rank_v2_2_auto_w
from .rank_loss_v2_2_pred_w import build_loss as build_loss_rank_v2_2_pred_w
from .rank_loss_v2_2_est_var import build_loss as build_loss_rank_v2_2_est_var
from .rank_loss_v2_2_offset_reg import build_loss as build_loss_rank_v2_2_offset_reg
from .rank_loss_v2_2_offset_sp import build_loss as build_loss_rank_v2_2_offset_sp
from .rank_loss_v2_2_offset_sp2 import build_loss as build_loss_rank_v2_2_offset_sp2
from .rank_loss_v2_2_offset_sp3 import build_loss as build_loss_rank_v2_2_offset_sp3
from .rank_loss_v2_2_offset_sp4 import build_loss as build_loss_rank_v2_2_offset_sp4
from .rank_loss_v2_2_offset_sp5 import build_loss as build_loss_rank_v2_2_offset_sp5
from .rank_loss_v2_2_offset_sp6 import build_loss as build_loss_rank_v2_2_offset_sp6
from .rank_loss_v2_2_offset_sp7 import build_loss as build_loss_rank_v2_2_offset_sp7
from .rank_loss_final import build_loss as build_loss_rank_final
from .rank_loss_final_no_mask import build_loss as build_loss_rank_final_no_mask
from .l2 import build_loss as build_loss_l2
from .dmap_loss import build_loss as build_loss_dmap

def build_loss(cfg):
    if cfg.name == "counting":
        if cfg.type == "local_bl":
            return build_loss_local_bl(cfg)
        if cfg.type == "local_bl_v3":
            return build_loss_local_bl_v3(cfg)  
        if cfg.type == "local_bl_v8":
            return build_loss_local_bl_v8(cfg)
        if cfg.type == "local_bl_v9":
            return build_loss_local_bl_v9(cfg)
        if cfg.type == "local_bl_v10":
            return build_loss_local_bl_v10(cfg)
        if cfg.type == "local_bl_v11":
            return build_loss_local_bl_v11(cfg)
        if cfg.type == "local_bl_v12":
            return build_loss_local_bl_v12(cfg)
        if cfg.type == "local_bl_v12_gauss":
            return build_loss_local_bl_v12_gauss(cfg)
        if cfg.type == "local_bl_v12_var":
            return build_loss_local_bl_v12_var(cfg)
        if cfg.type == "local_bl_v13":
            return build_loss_local_bl_v13(cfg)
        if cfg.type == "local_bl_v14":
            return build_loss_local_bl_v14(cfg)
        if cfg.type == "local_ot":
            return build_loss_local_ot(cfg)
        if cfg.type == "iis":
            return build_loss_iis(cfg)
        if cfg.type == "divden":
            return build_loss_divden(cfg)
        if cfg.type == "rank_loss":
            return build_loss_rank(cfg)
        if cfg.type == "rank_loss_v2":
            return build_loss_rank_v2(cfg)
        if cfg.type == "rank_loss_v2_bg":
            return build_loss_rank_v2_bg(cfg)
        if cfg.type == "rank_loss_v3":
            return build_loss_rank_v3(cfg)
        if cfg.type == "rank_loss_v4":
            return build_loss_rank_v4(cfg)
        if cfg.type == "rank_loss_v2_2":
            return build_loss_rank_v2_2(cfg)
        if cfg.type == "rank_loss_v2_3":
            return build_loss_rank_v2_3(cfg)
        if cfg.type == "rank_loss_v2_4":
            return build_loss_rank_v2_4(cfg)
        if cfg.type == "rank_loss_v2_5":
            return build_loss_rank_v2_5(cfg)
        if cfg.type == "rank_loss_v2_6":
            return build_loss_rank_v2_6(cfg)
        if cfg.type == "rank_loss_v2_2_dense":
            return build_loss_rank_v2_2_dense(cfg)
        if cfg.type == "rank_loss_v2_2_auto_w":
            return build_loss_rank_v2_2_auto_w(cfg)
        if cfg.type == "rank_loss_v2_2_pred_w":
            return build_loss_rank_v2_2_pred_w(cfg)
        if cfg.type == "rank_loss_v2_2_est_var":
            return build_loss_rank_v2_2_est_var(cfg)
        if cfg.type == "rank_loss_v2_2_offset_reg":
            return build_loss_rank_v2_2_offset_reg(cfg)
        if cfg.type == "rank_loss_v2_2_offset_sp":
            return build_loss_rank_v2_2_offset_sp(cfg)
        if cfg.type == "rank_loss_v2_2_offset_sp2":
            return build_loss_rank_v2_2_offset_sp2(cfg)
        if cfg.type == "rank_loss_v2_2_offset_sp3":
            return build_loss_rank_v2_2_offset_sp3(cfg)
        if cfg.type == "rank_loss_v2_2_offset_sp4":
            return build_loss_rank_v2_2_offset_sp4(cfg)
        if cfg.type == "rank_loss_v2_2_offset_sp5":
            return build_loss_rank_v2_2_offset_sp5(cfg)
        if cfg.type == "rank_loss_v2_2_offset_sp6":
            return build_loss_rank_v2_2_offset_sp6(cfg)
        if cfg.type == "rank_loss_v2_2_offset_sp7":
            return build_loss_rank_v2_2_offset_sp7(cfg)
        if cfg.type == "rank_loss_final":
            return build_loss_rank_final(cfg)
        if cfg.type == "rank_loss_final_no_mask":
            return build_loss_rank_final_no_mask(cfg)
        if cfg.type == "l2":
            return build_loss_l2(cfg)
        if cfg.type == "dmap_loss":
            return build_loss_dmap(cfg)
        raise ValueError("type not support")