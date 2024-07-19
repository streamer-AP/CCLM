import os
import torch
import torch.nn as nn
from .rank_loss_final import build_loss as build_loss_rank_final
from .rank_loss_final_no_mask import build_loss as build_loss_rank_final_no_mask
from .dmap_loss import build_loss as build_loss_dmap

def build_loss(cfg):
    if cfg.name == "counting":
        if cfg.type == "rank_loss_final":
            return build_loss_rank_final(cfg)
        if cfg.type == "rank_loss_final_no_mask":
            return build_loss_rank_final_no_mask(cfg)
        if cfg.type == "dmap_loss":
            return build_loss_dmap(cfg)
        raise ValueError("type not support")