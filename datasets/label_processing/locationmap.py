import torch
import numpy as np


def pt2lmap(pts, h, w, scale=16):
    # 3 -> (num_tgt, sum_tgt_h, sum_tgt_w)
    lmap = np.zeros((3, h, w))
    for pt in pts:
        pt_h = pt[1]/scale
        pt_w = pt[0]/scale
        lmap[0, int(pt_h), int(pt_w)] += 1
        lmap[1, int(pt_h), int(pt_w)] += pt_h
        lmap[2, int(pt_h), int(pt_w)] += pt_w

    return lmap

class DrawLocationMap:
    def __init__(self, args):

        self.scale = args.scale                
    
    def __call__(self, image, labels):
        _, H, W = image.shape
        h, w = H // self.scale, W // self.scale
        lmap = pt2lmap(labels["points"][: labels["num"]], h, w, self.scale)
        lmap = torch.from_numpy(lmap)
        labels["gt_lmaps"] = lmap
        labels["gt_dmaps"] = lmap[:1, :, :].clone()
        return image, labels