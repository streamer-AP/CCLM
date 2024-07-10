# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch.utils.data
from .torchvision_datasets import CocoDetection
from torch.nn import functional as F
from .coco import build as build_coco
from .nwpu_crowd import build_NWPU as build_nwpu_crowd
from .jhu_crowd import build_JHU as build_jhu_crowd
from .sta_crowd import build_STA as build_sta_crowd
from .sta_crowd_2048 import build_STA as build_sta_crowd_2048
from .stb_crowd import build_STB as build_stb_crowd
from .fdst_crowd import build_FDST as build_fdst_crowd
from .transcos_crowd import build_TRANS as build_trans_crowd
from .tree_crowd import build_TREE as build_tree_crowd
def get_coco_api_from_dataset(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.name == "jhu_crowd":
        return build_jhu_crowd(image_set,args)
    if args.name == "sta_crowd":
        return build_sta_crowd(image_set,args)
    if args.name == "sta_crowd_2048":
        return build_sta_crowd_2048(image_set,args)
    if args.name == "stb_crowd":
        return build_stb_crowd(image_set,args)
    if args.name == "fdst_crowd":
        return build_fdst_crowd(image_set,args)
    if args.name=="trans_crowd":
        return build_trans_crowd(image_set,args)
    if args.name=="tree_crowd":
        return build_tree_crowd(image_set,args)
    raise ValueError(f'dataset {args.name} not supported')

def collate_fn(batch):
    images,target = list(zip(*batch))
    max_h = max([image.shape[1] for image in images])
    max_w = max([image.shape[2] for image in images])
    new_images = []
    for image in images:
        image=F.pad(image,(0,max_w-image.shape[2],0,max_h-image.shape[1]),value=0)
        new_images.append(image)
    images = torch.stack(new_images,dim=0)
    
    max_h_dmap=max([t['gt_dmaps'].shape[1] for t in target])
    max_w_dmap=max([t['gt_dmaps'].shape[2] for t in target])
    
    targets={
        "gt_dmaps":torch.stack(
                [F.pad(t['gt_dmaps'],(0,max_w_dmap-t['gt_dmaps'].shape[2],0,max_h_dmap-t['gt_dmaps'].shape[1]),value=0) 
                        for t in target],
                dim=0),
        "num":torch.stack([t['num'] for t in target],dim=0),
        "points":torch.stack([F.pad(t['points'],(0,0,0,max_h_dmap-t['points'].shape[0]),value=0) for t in target],dim=0),
    }

    return images,targets