import os
import torch
import torch.nn as nn
from .divden import build_loss as build_loss_divden
from .ucl_loss import build_loss as build_loss_ucl
from .dmap_loss import build_loss as build_loss_dmap
def build_loss(cfg):
    if cfg.name == "counting":
        if cfg.type == "divden":
            return build_loss_divden(cfg)
        if cfg.type == "ucl":
            return build_loss_ucl(cfg)
        if cfg.type == "dmap_loss":
            return build_loss_dmap(cfg)
        raise ValueError("type not support")