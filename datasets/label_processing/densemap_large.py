import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter

def create_density_kernel(kernel_size, sigma):
    
    kernel = np.zeros((kernel_size, kernel_size))
    mid_point = kernel_size//2
    kernel[mid_point, mid_point] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)

    return kernel

class GaussianKernel(nn.Module):

    def __init__(self, kernel_weights):
        super().__init__()
        self.kernel = nn.Conv2d(1,1,kernel_weights.shape, bias=False, padding=kernel_weights.shape[0]//2)
        kernel_weights = torch.tensor(kernel_weights).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.kernel.weight = nn.Parameter(kernel_weights)
    
    def forward(self, density):
        return self.kernel(density).squeeze(0)

def pt2dmap(pts, h, w, num_factor=1, scale=1):
    dmap = np.zeros((h, w))
    for pt in pts:
        dmap[int(pt[1]/scale), int(pt[0]/scale)] += num_factor
    return dmap


class DrawLargeDenseMap:
    def __init__(self, args):
        self.num_factor = args.num_factor           # 1
        # self.kernel_size = args.kernel_size     # 0
        self.scale = args.scale                 # 16

        self.large_kernel_size = args.large_kernel_size  # 3
        self.large_scale = args.large_scale       # 2
        self.large_kernel_sigma = args.large_kernel_sigma   # 0.5

        self.large_kernel = GaussianKernel(create_density_kernel(self.large_kernel_size, self.large_kernel_sigma))
    
    def __call__(self, image, labels):
        _, H, W = image.shape
        h, w = H // self.scale, W // self.scale
        dmap = pt2dmap(labels["points"][: labels["num"]], h, w, self.num_factor, self.scale)
        dmap = torch.from_numpy(dmap).unsqueeze(0)
        labels["gt_dmaps"] = dmap
        # print("dmap", dmap.shape)

        h, w = H // self.large_scale, W // self.large_scale
        large_dmap = pt2dmap(labels["points"][: labels["num"]], h, w, self.num_factor, self.large_scale)
        large_dmap = torch.from_numpy(large_dmap).unsqueeze(0).unsqueeze(0)
        # print("large_1", large_dmap.shape)
        with torch.no_grad():
            large_dmap = self.large_kernel(large_dmap)
        labels["gt_large_dmaps"] = large_dmap
        # print("large", large_dmap.shape)

        return image, labels