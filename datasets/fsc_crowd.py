import os

import albumentations as A
from albumentations import (ShiftScaleRotate,ColorJitter,Compose,Normalize,PadIfNeeded,Resize,RandomResizedCrop)
import cv2
from misc.utils import get_local_rank, get_local_size

from .label_processing import build_label_processing
from .base import Counting_train, Counting_test
import torch


def make_transform(image_set):
    if image_set == "train":
        return A.Compose([
            ColorJitter(),
            # ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=15,
            #                  p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.LongestMaxSize(768),
            A.PadIfNeeded(min_height=768, min_width=768, border_mode=cv2.BORDER_CONSTANT, value=0),

            A.HorizontalFlip(),
            A.Normalize(),
        ],
            # keypoint_params=A.KeypointParams(format='xy', label_fields=[
            #                                  'class_labels'], remove_invisible=True)
            bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'])
        )
    elif image_set == "val" or image_set=="test":
        return A.Compose([
            A.LongestMaxSize(1024),
            A.Normalize(),
        ])

class FSC147_train(Counting_train):
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
    def __getitem__(self, index):
        image, labels=super().__getitem__(index)
        # filter out the labels with class_id=0
        new_labels = {}
        new_labels["points"] = torch.zeros((self.max_len, 2), dtype=torch.float32)
        new_labels["classes"] = torch.zeros(self.max_len, dtype=torch.long)
        new_labels["id"] = labels["id"]
        new_labels["wh"] = labels["wh"]
        new_labels["boxes"] = torch.zeros((self.max_len, 4), dtype=torch.float32)
        exampler_mask=torch.zeros((1,image.shape[1],image.shape[2]),dtype=torch.float32)
        new_labels["exampler"]= torch.zeros((self.max_len, 4), dtype=torch.float32)
        target_cnt=0
        exampler_cnt=0
        for i in range(labels["num"]):
            if labels["classes"][i] == 0:
                new_labels["points"][target_cnt] = labels["points"][i]
                new_labels["classes"][target_cnt] = labels["classes"][i]
                new_labels["boxes"][target_cnt] = labels["boxes"][i]
                target_cnt+=1
            else:
                new_labels["exampler"][exampler_cnt] = labels["boxes"][i]
                x,y,w,h=labels["boxes"][i]
                x,y,w,h=int(x),int(y),int(w),int(h)
                exampler_mask[0,y:y+h,x:x+w]=1
                exampler_cnt+=1
        new_labels["num"] = target_cnt
        image = torch.cat([image,exampler_mask],dim=0)
        return image, new_labels

class FSC147_test(Counting_test):
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
    def __getitem__(self, index):
        image, labels=super().__getitem__(index)
        # filter out the labels with class_id=0
        new_labels = {}
        new_labels["points"] = torch.zeros((self.max_len, 2), dtype=torch.float32)
        new_labels["classes"] = torch.zeros(self.max_len, dtype=torch.long)
        new_labels["id"] = labels["id"]
        new_labels["wh"] = labels["wh"]
        new_labels["w1h1"]=labels["w1h1"]
        new_labels["boxes"] = torch.zeros((self.max_len, 4), dtype=torch.float32)
        new_labels["exampler"]= torch.zeros((self.max_len, 4), dtype=torch.float32)
        new_labels["exampler_ori"]= torch.zeros((self.max_len, 4), dtype=torch.float32)
        
        target_cnt=0
        exampler_cnt=0
        exampler_mask=torch.zeros((1,image.shape[1],image.shape[2]),dtype=torch.float32)
        ori_w,ori_h=labels["wh"]#original image size
        resized_w,resized_h=labels["w1h1"]#resized image size
        w_scale=ori_w/resized_w
        h_scale=ori_h/resized_h
        
        for i in range(labels["num"]):
            if labels["classes"][i] == 0:
                new_labels["points"][target_cnt] = labels["points"][i]
                new_labels["classes"][target_cnt] = labels["classes"][i]
                new_labels["boxes"][target_cnt] = labels["boxes"][i]
                target_cnt+=1
            else:
                x,y,w,h=labels["boxes"][i]
                new_labels["exampler_ori"][exampler_cnt] = torch.tensor([x,y,w,h],dtype=torch.float32)
                x,w=x/w_scale,w/w_scale
                y,h=y/h_scale,h/h_scale
                x,y,w,h=int(x),int(y),int(w),int(h)
                new_labels["exampler"][exampler_cnt] = torch.tensor([x,y,w,h],dtype=torch.float32)
                exampler_mask[0,y:y+h,x:x+w]=1
                exampler_cnt+=1
        new_labels["num"] = target_cnt
        image = torch.cat([image,exampler_mask],dim=0)
        return image, new_labels
            
def build_FSC(image_set, args):
    img_prefix = args.img_prefix
    ann_file = args.ann_file
    assert os.path.exists(img_prefix), f"image prefix {img_prefix} not exists"
    assert os.path.exists(ann_file), f"annotation file {ann_file} not exists"
    if image_set == "train":
        dataset = FSC147_train(img_prefix,
                                    ann_file,
                                    max_len=args.max_len,
                                    transforms=make_transform(image_set),
                                    cache_mode=args.cache_mode,
                                    local_rank=get_local_rank(),
                                    local_size=get_local_size())
    elif image_set == "test" or image_set == "val":
        dataset = FSC147_test(img_prefix,
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
