import torch
import numpy as np
from scipy.ndimage import gaussian_filter


def pt2dmap(pts, h, w, num_factor=1, sigma=0, scale=1):
    dmap = np.zeros((h, w))
    for pt in pts:
        dmap[int(pt[1]/scale), int(pt[0]/scale)] += num_factor
    if sigma > 0:
        dmap = gaussian_filter(dmap, sigma, mode="constant", cval=0.0)
    return dmap


class DrawDenseMapB3:
    def __init__(self, args):
        self.num_factor = args.num_factor                      
    
    def __call__(self, image, labels):
        _, H, W = image.shape
        H1, W1 = H // 2, W // 2
        H2, W2 = H * 3 // 8, W * 3 // 8
        H3, W3 = H // 4, W // 4
        dmap1 = pt2dmap(labels["points"][: labels["num"]], H1, W1, self.num_factor, 0, 2)
        dmap1 = torch.from_numpy(dmap1).unsqueeze(0)
        dmap2 = pt2dmap(labels["points"][: labels["num"]], H2, W2, self.num_factor, 0, 8.0 / 3)
        dmap2 = torch.from_numpy(dmap2).unsqueeze(0)
        dmap3 = pt2dmap(labels["points"][: labels["num"]], H3, W3, self.num_factor, 0, 4)
        dmap3 = torch.from_numpy(dmap3).unsqueeze(0)
        labels["gt_dmaps"] = dmap1
        labels["gt_dmaps1"] = dmap1
        labels["gt_dmaps2"] = dmap2
        labels["gt_dmaps3"] = dmap3
        return image, labels