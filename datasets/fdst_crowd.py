import os
from pathlib import Path

import albumentations as A
from albumentations import (ShiftScaleRotate,ColorJitter,Compose,Normalize,PadIfNeeded,Resize,RandomResizedCrop)
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from misc.utils import get_local_rank, get_local_size

import datasets.transforms as T

from .torchvision_datasets.coco import CocoDetection
import einops
import numpy as np
import random
from .label_processing import build_label_processing
import math
from torch.nn import functional as F
from .jhu_crowd import JHUCounting_train, JHUCounting_test
_scence_labels_ = {'stadium':1, 'street':2, 'crowd-gathering':3, 'protest':4, 'conference':5, 'concert':6, 'rally':7, 'gathering':8, 'other':0}

def make_transform(image_set):
    if image_set == "train":
        return A.Compose([
            ColorJitter(),
            A.ToGray(p=0.3),
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=15,
                             p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.LongestMaxSize(1536),
            A.PadIfNeeded(min_height=1536, min_width=1536, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(),
            A.Normalize(),
        ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=[
                                             'class_labels'], remove_invisible=True)
        )
    elif image_set == "val":
        return A.Compose([
            A.LongestMaxSize(2048),
            A.Normalize(),
        ])


def build_FDST(image_set, args):
    # print("build jhu dataset")
    img_prefix=args.img_prefix
    ann_file=args.ann_file
    assert os.path.exists(img_prefix),f"image prefix {img_prefix} not exists"
    assert os.path.exists(ann_file),f"annotation file {ann_file} not exists"
    if image_set == "train":
        dataset = JHUCounting_train(img_prefix, ann_file, max_len=args.max_len,transforms=make_transform(image_set),
                           cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    elif image_set == "test" or image_set == "val":
        dataset = JHUCounting_test(img_prefix, 
                                    ann_file,
                                    max_len=args.max_len, 
                                    transforms=make_transform(image_set),
                                    cache_mode=args.cache_mode, 
                                    local_rank=get_local_rank(), 
                                    local_size=get_local_size())
    else:
        raise ValueError("image_set {} should be train, test or val".format(image_set))
    return dataset
