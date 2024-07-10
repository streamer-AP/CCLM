import os

import albumentations as A
from albumentations import (ShiftScaleRotate,ColorJitter,Compose,Normalize,PadIfNeeded,Resize,RandomResizedCrop)
import cv2
from misc.utils import get_local_rank, get_local_size


from .label_processing import build_label_processing
from .base import Counting_train, Counting_test

def make_transform(image_set):
    if image_set == "train":
        return A.Compose([

            ColorJitter(),

            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=15,
                             p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.LongestMaxSize(1024),
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=0),

            A.HorizontalFlip(),

            A.Normalize(),
        ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=[
                                             'class_labels'], remove_invisible=True)
        )
    elif image_set == "val":

        return A.Compose([
            A.LongestMaxSize(1024),
            A.Normalize(),
        ])


def build_TRANS(image_set, args):
    # print("build jhu dataset")
    img_prefix=args.img_prefix
    ann_file=args.ann_file
    assert os.path.exists(img_prefix),f"image prefix {img_prefix} not exists"
    assert os.path.exists(ann_file),f"annotation file {ann_file} not exists"
    if image_set == "train":
        dataset = Counting_train(img_prefix, ann_file, max_len=args.max_len,transforms=make_transform(image_set),
                           cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    elif image_set == "test" or image_set == "val":
        dataset = Counting_test(img_prefix, 
                                    ann_file,
                                    max_len=args.max_len, 
                                    transforms=make_transform(image_set),
                                    cache_mode=args.cache_mode, 
                                    local_rank=get_local_rank(), 
                                    local_size=get_local_size())
    else:
        raise ValueError("image_set {} should be train, test or val".format(image_set))
    dataset.LabelProcessing = build_label_processing(args.labelprocessing)
    return dataset
