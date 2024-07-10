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
import torch.distributed as dist
from .torchvision_datasets.coco import CocoDetection
import einops
import numpy as np
import random
from .label_processing import build_label_processing
import math
from torch.nn import functional as F


def make_transform(image_set):
    if image_set == "train":
        return A.Compose([
            ColorJitter(),
            A.ToGray(p=0.3),
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=15,
                             p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomResizedCrop(1024,1024, scale=(0.9, 1.1), p=1),
            A.HorizontalFlip(),
            A.Normalize(),
        ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=[
                                             'class_labels'], remove_invisible=True)
        )
    elif image_set == "val" or image_set=="test":
        return A.Compose([
            A.Normalize(),
        ])
    
def make_train_transform():
    maxlength = 3072
    alb_trans1 = A.Compose([
            ColorJitter(),
            A.ToGray(p=0.3),
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=15,
                            p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomResizedCrop(1024,1024, scale=(0.9, 1.1), p=1),
            A.HorizontalFlip(),
            A.Normalize(),
        ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=[
                                            'class_labels'], remove_invisible=True)
        )

    alb_trans2 = A.Compose([
        ColorJitter(),
        A.ToGray(p=0.3),
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=15,
                        p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.LongestMaxSize(maxlength),
        A.RandomResizedCrop(1024,1024, scale=(0.9, 1.1), p=1),
        A.HorizontalFlip(),
        A.Normalize(),
    ],
        keypoint_params=A.KeypointParams(format='xy', label_fields=[
                                        'class_labels'], remove_invisible=True)
    )
    return alb_trans1, alb_trans2

class NwpuCounting_train_base(CocoDetection):
    def __init__(self,
                 root,
                 annFile,
                 transforms=None,
                 max_len=5000,
                 cache_mode=False,
                 local_rank=0,
                 local_size=1):
        super().__init__(root,
                         annFile,
                         transform=None,
                         target_transform=None,
                         transforms=None,
                         cache_mode=cache_mode,
                         local_rank=local_rank,
                         local_size=local_size)
        self.alb_trans1, self.alb_trans2 = make_train_transform()
        self.to_tensor = ToTensorV2()
        self.max_len = max_len


    def __getitem__(self, index):

        image, target = super().__getitem__(index)
        
        w, h = image.size
        img_id = self.ids[index]
        if max(w, h) > 3072:
            alb_trans = self.alb_trans2
        else:
            alb_trans = self.alb_trans1

        image = np.array(image)
        bboxes_with_classes = [(obj["bbox"], obj["category_id"])
                               for obj in target]
        clses, kpses = [], []
        # filter out the out of range bboxes
        for bbox, cls in bboxes_with_classes:
            x, y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            if x >= w or y >= h or x <= 0 or y <= 0:
                continue
            else:
                clses.append(cls)
                kpses.append((x, y))
        data = alb_trans(image=image,
                        keypoints=kpses,
                        class_labels=clses)
        image = data["image"]
        labels = {}
        keep = [
            idx for idx, v in enumerate(data["keypoints"])
            if v[1] < (image.shape[0] - 1) and v[0] < (image.shape[1] - 1)
        ]
        labels["num"] = torch.as_tensor(len(keep), dtype=torch.long)
        if len(keep) == 0:
            return self.__getitem__(random.randint(0, len(self) - 1))
        kpses = torch.as_tensor(data["keypoints"], dtype=torch.float32)[keep]
        clses = torch.as_tensor(data["class_labels"], dtype=torch.long)[keep]
        assert kpses.shape[0] == clses.shape[
            0], f"{kpses.shape[0]},{clses.shape[0]}"
        image = self.to_tensor(image=image)["image"]
        labels["points"] = torch.zeros((self.max_len, 2), dtype=torch.float32)
        labels["classes"] = torch.zeros(self.max_len, dtype=torch.long)
        labels["id"] = torch.as_tensor(int(img_id), dtype=torch.long)

        if labels["num"] > 0:

            labels["points"][:kpses.shape[0]] = kpses[:self.max_len]
            labels["classes"][:clses.shape[0]] = clses[:self.max_len]

        if labels["num"] > self.max_len:
            print(
                f"Warning: the number of points {labels['num']} is larger than max_len {self.max_len}"
            )
            labels["num"] = torch.as_tensor(self.max_len, dtype=torch.long)
        

        return image, labels


class NwpuCounting_train(NwpuCounting_train_base):
    def __init__(self,
                 root,
                 annFile,
                 transforms=None,
                 max_len=5000,
                 cache_mode=False,
                 local_rank=0,
                 local_size=1):
        super().__init__(root, annFile, transforms, max_len, cache_mode,
                         local_rank, local_size)
        self.LabelProcessing = lambda *args: None

    def __getitem__(self, index):
        image, labels = super().__getitem__(index)
        self.LabelProcessing(image, labels)
        return image, labels


class NwpuCounting_test(CocoDetection):
    def __init__(self,
                 root,
                 annFile,
                 transforms=None,
                 max_len=5000,
                 cache_mode=False,
                 local_rank=0,
                 local_size=1):
        super().__init__(root,
                         annFile,
                         transform=None,
                         target_transform=None,
                         transforms=None,
                         cache_mode=cache_mode,
                         local_rank=local_rank,
                         local_size=local_size)
        self.alb_transforms = transforms
        self.to_tensor = ToTensorV2()
        self.max_len = max_len

    def __getitem__(self, index):

        image, target = super().__getitem__(index)
        img_id = self.ids[index]
        w, h = image.size
        image = np.array(image)
        bboxes_with_classes = [(obj["bbox"], obj["category_id"])
                               for obj in target]
        kpses = []
        for bbox, cls in bboxes_with_classes:
            x, y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            kpses.append((x, y))
        data = self.alb_transforms(image=image)
        image = data["image"]
        max_edge = max(image.shape[0], image.shape[1])

        if max_edge > 2560:
            scale = 2560 / max_edge
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        if max_edge < 1024:
            scale= 1024/max_edge
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        labels = {}

        labels["num"] = torch.as_tensor(len(kpses), dtype=torch.long)
        labels["wh"] = torch.as_tensor([w, h], dtype=torch.long)
        labels["id"] = torch.as_tensor(int(img_id), dtype=torch.long)
        kpses = torch.as_tensor(kpses, dtype=torch.float32)

        image = self.to_tensor(image=image)["image"]
        labels["points"] = torch.zeros((self.max_len, 2), dtype=torch.float32)
        if labels["num"] > 0:
            labels["points"][:kpses.shape[0]] = kpses[:self.max_len]
        if labels["num"] > self.max_len:
            print(
                f"Warning: the number of points {labels['num']} is larger than max_len {self.max_len}"
            )
            labels["num"] = torch.as_tensor(self.max_len, dtype=torch.long)
        # 把 image pad成64的倍数
        h1, w1 = image.shape[-2], image.shape[-1]
        padsize = 64
        h2 = math.ceil(h1 / padsize) * padsize
        w2 = math.ceil(w1 / padsize) * padsize
        h_pad = h2 - h1
        w_pad = w2 - w1
        image_pad = F.pad(image, pad=(0, w_pad, 0, h_pad))
        labels["w1h1"] = torch.as_tensor([w1, h1], dtype=torch.long)

        return image_pad, labels


def build_Nwpu(image_set, args):
    # print("build Nwpu dataset")
    img_prefix = args.img_prefix
    ann_file = args.ann_file
    assert os.path.exists(img_prefix), f"image prefix {img_prefix} not exists"
    assert os.path.exists(ann_file), f"annotation file {ann_file} not exists"
    if image_set == "train":
        dataset = NwpuCounting_train(img_prefix,
                                    ann_file,
                                    max_len=args.max_len,
                                    transforms=None,
                                    cache_mode=args.cache_mode,
                                    local_rank=get_local_rank(),
                                    local_size=get_local_size())
    elif image_set == "test" or image_set == "val":
        dataset = NwpuCounting_test(img_prefix,
                                   ann_file,
                                   max_len=args.max_len,
                                   transforms=make_transform(image_set),
                                   cache_mode=args.cache_mode,
                                   local_rank=get_local_rank(),
                                   local_size=get_local_size())
    else:
        raise ValueError(
            "image_set {} should be train, test or val".format(image_set))
    dataset.LabelProcessing = build_label_processing(args.labelprocessing)
    return dataset



