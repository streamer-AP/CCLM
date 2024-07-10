# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from timm import create_model
#from util.misc import NestedTensor, is_main_process
from timm.models import features


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,pretrained:bool,out_indices:List[int], train_backbone: bool,others:Dict):
        super(Backbone,self).__init__()
        backbone=create_model(name,pretrained=pretrained,features_only=True, out_indices=out_indices,**others)
        self.train_backbone = train_backbone
        self.backbone=backbone
        self.out_indices=out_indices
        if not self.train_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)
    def forward(self,x):
        return self.backbone(x)
    
    @property
    def feature_info(self):
        return features._get_feature_info(self.backbone,out_indices=self.out_indices)


def build_backbone(args):
    backbone = Backbone(args.name, args.pretrained, args.out_indices, args.train_backbone, args.others)
    return backbone
