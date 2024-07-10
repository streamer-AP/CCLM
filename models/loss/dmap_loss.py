import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import torch.distributed as dist 
import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
def gen_coord(ks):
    M = torch.arange(ks) - ks // 2
    grid_h, grid_w = torch.meshgrid(M, M, indexing="ij")
    grid = torch.dstack((grid_h, grid_w))
    return grid.reshape(1, -1, 2).transpose(1, 2)

def gen_distance_transform(pts, H, W, padding):
    a = np.ones((H+2*padding, W+2*padding))
    for pt in pts:
        a[padding + pt[0], padding + pt[1]] = 0
    a = distance_transform_edt(a)
    a = np.power(a, 2)
    return a

def gen_local_mask(pts, H, W, padding):
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    N = pts.shape[0]
    pts_pad = pts + padding
    # 计算格点坐标和距离矩阵
    grid_i, grid_j = np.indices((H_pad, W_pad))  
    grid_pts_ori = np.stack([grid_i, grid_j], axis=-1).reshape(-1, 2)
    grid_pts = grid_pts_ori
    if N == 1:
        dist_mat = cdist(grid_pts, pts_pad)
        idx1 = np.argmin(dist_mat, axis=1) + 1
        dist1 = np.min(dist_mat, axis=1)
        mask1 = (dist1 < 16)
    else:
        # 使用KDtree求最近的点的距离
        kd_tree = KDTree(data=pts_pad)
        dist1, index1 = kd_tree.query(grid_pts, k=1)
        idx1 = index1 + 1
        mask1 = (dist1 < 16)

    labels = np.zeros(shape=(H_pad*W_pad), dtype=np.int32)
    labels[mask1] = idx1[mask1]
    # labels[mask2] = idx2[mask2]

    mask = labels != 0
    gt_offset = np.zeros((H_pad*W_pad, 2))
    gt_offset[mask] = pts_pad[labels[mask]-1]
    gt_offset[mask] = gt_offset[mask] - grid_pts_ori[mask]
    gt_offset = gt_offset.reshape(H_pad, W_pad, 2)
    return gt_offset, labels.reshape(H_pad, W_pad)

class DMap_Loss(nn.Module):
    def __init__(self, cfg, radius=8, stride=2):
        super().__init__()
        self.weight_offset = 0.01
        self.weight_cnt=10
        self.stride=2
        self.radius = radius
        self.window_size = 2 * self.radius + 1
        self.padding = self.window_size // 2+1
    def update_weight(self, epoch):
        if epoch < self.smooth_epoch:
            self.smooth_weight = (epoch / self.smooth_epoch) * (1 - self.smooth_init_rate) + self.smooth_init_rate
        else:
            self.smooth_weight = 1

    def forward(self, inputs, targets):
        loss_dict = {}
        loss_dict["ann"] = 0
        loss_dict["bg"] = 0

        predict = inputs["predict_counting_map"]

        num = targets["num"]
        device = predict.device
        B, C, H, W = predict.shape
        predict = F.pad(predict, (self.padding, self.padding, self.padding, self.padding))

        for b in range(B):
            N = int(num[b].item())
            pred_map = predict[b]
            loss_bg = (pred_map.abs().pow(2) / H).sum() / W
            if N==0:
                # loss_bg = (pred_map.abs() / H).sum() / W
                loss_ann = torch.as_tensor(0.0, device=device)
            else:
                # mask_bg = torch.ones_like(pred_map)
                with torch.no_grad():
                    gt_pts_ori = targets["points"][b, :N, :].flip(-1).numpy() / self.stride
                    gt_pts = np.round(gt_pts_ori).astype(np.int32)
                    gt_dmap=np.zeros((H, W))
                    for pt in gt_pts:
                        gt_dmap[int(pt[0]), int(pt[1])] = 1
                    gt_dmap = gaussian_filter(gt_dmap, sigma=4)
                    gt_dmap = torch.as_tensor(gt_dmap, device=device).half()
                    gt_dmap = gt_dmap.unsqueeze(0)
                    gt_dmap=F.pad(gt_dmap, (self.padding, self.padding, self.padding, self.padding))
                loss_dmap=torch.nn.functional.mse_loss(pred_map, gt_dmap, reduction='mean')

            loss_dict["dmap"] = loss_dmap
            loss_dict["cnt"] = torch.abs(torch.sum(gt_dmap)-torch.sum(pred_map))/H/W
        loss_dict["dmap"] /= B
        
        loss_dict["all"] =  self.weight_cnt*(loss_dict["dmap"] + loss_dict["cnt"])

        
        return loss_dict


def build_loss(cfg):
    return DMap_Loss(cfg)



if __name__ == "__main__":
    # loss = DMap_Loss(2, 1)
    # x = torch.zeros(6, 6).float()
    # x[1, 1] = 0.8
    # x[0, 1] = 0.1
    # x[1, 5] = 1.2
    # x[4, 3] = 0.8
    # x = x.cuda().unsqueeze(0).unsqueeze(0)
    # fea = torch.zeros_like(x)
    # inputs = {"predict_counting_map": x, "offset_map": torch.zeros(1, 2, 6, 6).cuda()}
    # print(x)
    # x.requires_grad = True
    # N = 3
    # y = torch.as_tensor([[1, 1], [5, 1], [3, 4]]).float().reshape(1, N, 2)
    # y.requires_grad = False
    # targets = {"points": y, "num": torch.as_tensor(N).reshape(1)}
    # # targets = {"points": y, "num": torch.as_tensor(0).reshape(1)}
    # loss_dict = loss(inputs, targets)
    # print(loss_dict)
    # loss_dict["all"].backward()
    # print(loss_dict)
    class CFG(object):
        def __init__(cfg):
            cfg.weight_all = 100
            cfg.weight_bg = 20
            cfg.weight_ann = 100
            cfg.stride = 1
        
        def __contains__(self, x):
            return True
    
    cfg = CFG()
    loss = DMap_Loss(cfg, 2, 1)
    x = torch.zeros(6, 6).float()
    x[1, 1] = 0.8
    x[0, 1] = 0.1
    x[1, 5] = 1.2
    x[4, 3] = 0.8
    # for i in range(6):
    #     for j in range(6):
    #         if i != 0 or j!= 0:
    #             x[i,j] = 1.0
    x = x.cuda().unsqueeze(0).unsqueeze(0)
    x.requires_grad = True
    fea = torch.zeros_like(x)
    inputs = {"predict_counting_map": x, "offset_map": torch.zeros(1, 2, 6, 6).cuda()}
    print(x)
    x.requires_grad = True

    off = torch.zeros(1, 2, 6, 6).cuda()
    off.requires_grad = True

    ############# 定义y
    N = 6 * 6
    gt_points = []
    for i in range(6):
        for j in range(6):
            gt_points.append([i, j])
    y = torch.as_tensor(gt_points).float().reshape(1, N, 2)
    # N = 3
    # y = torch.as_tensor([[1, 1], [5, 1], [3, 4]]).float().reshape(1, N, 2)
    y.requires_grad = False

    ############# 定义y
    targets = {"points": y, "num": torch.as_tensor(N).reshape(1)}
    # targets = {"points": y, "num": torch.as_tensor(0).reshape(1)}
    
    lr = 0.01

    torch.set_printoptions(sci_mode=False, linewidth=100)
    
    for i in range(100):
        print(x)
        # print(off)
        inputs = {"predict_counting_map": x, "offset_map": off}
        (grad_x, grad_off) = torch.autograd.grad(loss(inputs, targets)["all"], (x, off))
        # print(grad_x.shape)
        print(grad_x * 0.36)
        x = F.relu(x - lr * grad_x)
        off = off - lr * grad_off
        
    
    # a = gen_nearest_offset(y.flip(-1)[0].numpy(), 6, 6)
    # print(a)
    # loss_dict = loss(inputs, targets)
    # print(loss_dict)
    # loss_dict["all"].backward()
    # print(loss_dict)
    
    # pts = [[1, 1], [1, 5], [4, 3]]
    # _, a = gen_distance_transform(pts, 6, 6, 1)
    # print(a[:, :, 0])
    # print(a[:, :, 1])