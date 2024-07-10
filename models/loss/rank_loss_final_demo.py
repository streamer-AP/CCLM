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
    def __init__(self, cfg, radius=8):
        super().__init__()
        if "stride" in cfg:
            self.stride=cfg.stride
        else:
            self.stride=2
        
        self.radius = radius
        self.window_size = 2 * self.radius + 1
        self.padding = self.window_size // 2+1
        self.weight_reg = 1.0 / 128
        self.register_buffer("coord", gen_coord(self.window_size))

        self.weight_offset = 0.1
        self.smooth_weight = 1

        self.weight_all = cfg.weight_all
        self.weight_bg = cfg.weight_bg
        self.weight_ann = cfg.weight_ann

        self.smooth_epoch = cfg.smooth_epoch # 让负样本的权重缓慢增加
    
    def update_weight(self, epoch):
        if epoch < self.smooth_epoch:
            self.smooth_weight = epoch / self.smooth_epoch
        else:
            self.smooth_weight = 1

    def forward(self, inputs, targets):
        loss_dict = {}
        loss_dict["ann"] = 0
        loss_dict["bg"] = 0
        loss_dict["offset"] = 0

        predict = inputs["predict_counting_map"]
        offset = inputs["offset_map"]

        num = targets["num"]
        device = predict.device
        B, C, H, W = predict.shape
        predict = F.pad(predict, (self.padding, self.padding, self.padding, self.padding))
        offset = F.pad(offset, (self.padding, self.padding, self.padding, self.padding))

        for b in range(B):
            N = int(num[b].item())
            pred_map = predict[b]
            offset_map = offset[b]
            loss_bg = (pred_map.abs() / H).sum() / W
            if N==0:
                loss_ann =  torch.as_tensor(0.0, device=device)
                loss_offset = torch.as_tensor(0.0, device=device)
            else:
                with torch.no_grad():
                    gt_pts_ori = targets["points"][b, :N, :].flip(-1).numpy() / self.stride
                    gt_pts = np.round(gt_pts_ori).astype(np.int32)
                    gt_offset, seg = gen_local_mask(gt_pts, H, W, self.padding)
                    gt_offset = torch.as_tensor(gt_offset).float().to(device)
                    seg = torch.as_tensor(seg, dtype=torch.long, device=device).unsqueeze(0)
                    mask_0 = (seg > 0.5).float()
                ns_local = []
                offset_local = []
                pred_local = []
                for n in range(N):
                    pti, ptj = gt_pts[n]
                    tmap = pred_map[:, pti - self.radius + self.padding: pti + self.radius + self.padding + 1,\
                                   ptj - self.radius + self.padding: ptj + self.radius + self.padding + 1]
                    pred_local.append(tmap)
                    omap = offset_map[:, pti - self.radius + self.padding: pti + self.radius + self.padding + 1,\
                                   ptj - self.radius + self.padding: ptj + self.radius + self.padding + 1]
                    offset_local.append(omap.unsqueeze(0))
                    smap = seg[:, pti - self.radius + self.padding: pti + self.radius + self.padding + 1,\
                                   ptj - self.radius + self.padding: ptj + self.radius + self.padding + 1]
                    ns_map = (smap == (n + 1)).float()
                    ns_local.append(ns_map)
                
                pred_local = torch.cat(pred_local, dim=0)    # [N K K]
                pred_local = pred_local.reshape(N, -1)
                ns_local = torch.cat(ns_local, dim=0)    # [N K K]
                ns_local = ns_local.reshape(N, -1)

                pred_local = pred_local * ns_local

                offset_local = torch.cat(offset_local, dim=0)
                offset_local = offset_local.reshape(N, 2, -1)  # [N 2 K*K]

                dis = (offset_local.detach() + self.coord.to(device)).pow(2).sum(dim=1) * self.weight_reg  # [N K*K]
                score = (pred_local + 1e-6) * (1 - dis)    # [N K*K]
                rank, rand_id = torch.sort(score, dim=-1, descending=True)  
                # print(rand_id)
                pred_pos = torch.gather(pred_local, dim=-1, index=rand_id)
                dis_pos = torch.gather(dis, dim=-1, index=rand_id)

                pred_cum = torch.cumsum(pred_pos, dim=-1)
                cnt = (1 - pred_cum).abs()
                loc_pos = pred_pos * dis_pos
                loc_cum = torch.cumsum(loc_pos, dim=-1)
                log_prob = cnt + loc_cum
                
                loss_ann = torch.min(log_prob, dim=-1)[0].sum() / (H * W)

                loc_err =  (offset_map.permute(1, 2, 0) - gt_offset).norm(2, dim=-1)
                loss_offset = loc_err / (gt_offset.norm(2, dim=-1) + 1)
                loss_offset = (loss_offset * mask_0).sum() / (H * W)
                

            loss_dict["ann"] += loss_ann
            loss_dict["offset"] += loss_offset
            loss_dict["bg"] += loss_bg
        loss_dict["bg"] /= B
        loss_dict["ann"] /= B
        loss_dict["offset"] /= B 
        
        loss_dict["all"] = self.weight_bg * self.smooth_weight * loss_dict["bg"] \
                           + self.weight_ann * loss_dict["ann"] + self.weight_offset * loss_dict["offset"]

        
        return loss_dict


def build_loss(cfg):
    return DMap_Loss(cfg)



if __name__ == "__main__":
    class CFG(object):
        def __init__(cfg):
            cfg.weight_all = 100
            cfg.weight_bg = 20
            cfg.weight_ann = 100
            cfg.smooth_epoch = 0.0
            cfg.stride = 1
        
        def __contains__(self, x):
            return True
    
    cfg = CFG()
    loss = DMap_Loss(cfg, 2)
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

    ############# 定义y
    # N = 6 * 6
    # gt_points = []
    # for i in range(6):
    #     for j in range(6):
    #         gt_points.append([i, j])
    # y = torch.as_tensor(gt_points).float().reshape(1, N, 2)
    N = 3
    y = torch.as_tensor([[1, 1], [5, 1], [3, 4]]).float().reshape(1, N, 2)
    y.requires_grad = False

    off = torch.zeros(1, 2, 6, 6).cuda()
    off.requires_grad = True

    ############# 定义y
    targets = {"points": y, "num": torch.as_tensor(N).reshape(1)}
    # targets = {"points": y, "num": torch.as_tensor(0).reshape(1)}
    
    lr = 0.05
    
    for i in range(10):
        print(x)
        print(off)
        inputs = {"predict_counting_map": x, "offset_map": off}
        (grad_x, grad_off) = torch.autograd.grad(loss(inputs, targets)["all"], (x, off))
        # print(grad_x.shape)
        x = F.relu(x - lr * grad_x)
        off = off - lr * grad_off
        
    
    # loss_dict = loss(inputs, targets)
    # print(loss_dict)
    # loss_dict["all"].backward()
    # print(loss_dict)