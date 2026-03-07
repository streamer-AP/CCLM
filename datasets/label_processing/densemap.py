import torch
import numpy as np
from scipy.ndimage import gaussian_filter


def pt2dmap(pts, h, w, num_factor=1, sigma=0, scale=1):
    dmap = np.zeros((h, w))
    pts=pts.cpu().numpy()
    for pt in pts:
        y,x=round(pt[1]/scale), round(pt[0]/scale)
        y=min(h-1,y)
        x=min(w-1,x)
        dmap[y,x] += num_factor
    if sigma > 0:
        dmap = gaussian_filter(dmap, sigma, mode="constant", cval=0.0)
    return dmap


class DrawDenseMap:
    def __init__(self, args):
        self.num_factor = args.num_factor       # 1
        self.kernel_size = args.kernel_size     # 0
        self.scale = args.scale                 # 8
    
    def __call__(self, image, labels):
        _, H, W = image.shape
        h, w = H // self.scale, W // self.scale
        dmap = pt2dmap(labels["points"][: labels["num"]], h, w, self.num_factor, self.kernel_size, self.scale)
        dmap = torch.from_numpy(dmap).unsqueeze(0)
        labels["gt_dmaps"] = dmap
        return image, labels