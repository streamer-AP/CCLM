import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from geomloss import SamplesLoss


def create_density_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    mid_point = kernel_size // 2
    kernel[mid_point, mid_point] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)

    return kernel


def create_center(kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    mid_point = kernel_size // 2
    kernel[mid_point, mid_point] = 1
    return kernel


class DMap_Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ks1 = 5
        sg1 = 0.5
        ks2 = 9
        sg2 = 0.9
        ks3 = 17
        sg3 = 1.7

        ks_cnt = 37

        self.ks1 = ks1
        self.ks2 = ks2
        self.ks3 = ks3
        self.ks = ks3
        self.pd = ks3 // 2

        self.pad1 = (self.ks3 - self.ks1) // 2
        self.pad2 = (self.ks3 - self.ks2) // 2

        ones = torch.ones(1, 1, ks_cnt, ks_cnt).float()
        self.pad_ones = ks_cnt // 2
        self.register_buffer("ones", ones)
        ones1 = torch.ones(1, 1, ks1, ks1).float()
        self.pad_ones1 = ks1 // 2
        self.register_buffer("ones1", ones1)
        ones2 = torch.ones(1, 1, ks2, ks2).float()
        self.pad_ones2 = ks2 // 2
        self.register_buffer("ones2", ones2)
        ones3 = torch.ones(1, 1, ks3, ks3).float()
        self.pad_ones3 = ks3 // 2
        self.register_buffer("ones3", ones3)

        t = (ks3 - ks1) // 2
        kernel1 = np.zeros((ks3, ks3))
        kernel1[t: -t, t: -t] = create_density_kernel(ks1, sg1)
        kernel1 = torch.as_tensor(kernel1).float().reshape(1, ks3 ** 2, 1)
        self.register_buffer("k1", kernel1)

        t = (ks3 - ks2) // 2
        kernel2 = np.zeros((ks3, ks3))
        kernel2[t: -t, t: -t] = create_density_kernel(ks2, sg2)
        kernel2 = torch.as_tensor(kernel2).float().reshape(1, ks3 ** 2, 1)
        self.register_buffer("k2", kernel2)

        kernel3 = create_density_kernel(ks3, sg3)
        kernel3 = torch.as_tensor(kernel3).float().reshape(1, ks3 ** 2, 1)
        self.register_buffer("k3", kernel3)

        center = create_center(ks3)
        center = torch.as_tensor(center).float().reshape(1, ks3 ** 2, 1)
        self.register_buffer("cen", center)

        x = torch.arange(0, self.ks3, dtype=torch.float32)
        y = torch.arange(0, self.ks3, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x, y)
        coord = torch.dstack([grid_x, grid_y])
        coord = einops.rearrange(coord, "h w d->(h w) d")
        self.register_buffer("coord", coord)


        self.mae = nn.L1Loss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")
        self.ot = SamplesLoss(
            "sinkhorn", p=2, blur=0.05, reach=1, scaling=0.9, debias=False, potentials=False
        )

        self.weight_all = 100

    def forward(self, inputs, targets):
        predict = inputs["predict_counting_map"]
        # print("pred", predict.sum())
        device = predict.device
        B, C, H, W = predict.shape
        assert C == 1

        with torch.no_grad():
            # 按照17x17内总人数分类
            dmap = targets["gt_dmaps"].float().cuda(device)
            seg = (dmap > 0.5).float()
            cnt = F.conv2d(dmap, self.ones.cuda(device), padding=self.pad_ones)
            # 按照 17x17 内的总标注点数目将所有标注点分成三类
            cls1 = (cnt >= 4.5).float() * dmap
            cls2 = ((cnt < 4.5) * (cnt >= 1.5)).float() * dmap
            cls3 = ((cnt < 1.5) * (cnt >= 0.5)).float() * dmap
            # print("num", (cls1 + cls2 + cls3).sum(), dmap.sum())

            # 对于HW上每个点，计算出在其对应尺寸中是否有 对应cls的点
            cnt1 = F.conv2d(cls1, self.ones1.cuda(device), padding=self.pad_ones1)
            cnt2 = F.conv2d(cls2, self.ones2.cuda(device), padding=self.pad_ones2)
            cnt3 = F.conv2d(cls3, self.ones3.cuda(device), padding=self.pad_ones3)

            if1 = (cnt1 > 0.5).float().flatten(-2)
            if2 = ((cnt1 <= 0.5) * (cnt2 > 0.5)).float().flatten(-2)
            if3 = ((cnt1 <= 0.5) * (cnt2 <= 0.5) * (cnt3 > 0.5)).float().flatten(-2)
            if0 = 1 - if1 - if2 - if3

            # all_con = if1 + if2 + if3 + if0
            # print(all_con.shape, all_con.sum(), "all_con")

            dmap_unfold = F.unfold(dmap, self.ks, padding=self.pd)
            dmap_unfold = dmap_unfold + if0 * self.cen.cuda(device)
            kernel = if1 * self.k1.cuda(device) + if2 * self.k2.cuda(device) + if3 * self.k3.cuda(
                device) + if0 * self.cen.cuda(device)
            # print(kernel.shape, kernel.sum())
            dmap_unfold = dmap_unfold * kernel
            dmap_unfold = F.normalize(dmap_unfold, p=1, dim=1)  # B kxk HW
            # print(dmap_unfold.shape, dmap_unfold.sum(), "dmap_un")

            theta = torch.tensor([[1, 0], [0, 1]]).unsqueeze(0).float().cuda(device)
            theta = theta.repeat(self.ks3 * self.ks3, 1, 1)
            arange = torch.arange(self.ks3).float().flip(0).cuda(device) - self.ks3 // 2
            bw = arange * 2 / W
            bh = arange * 2 / H
            bw, bh = torch.meshgrid(bw, bh, indexing="xy")
            bg = torch.cat((bw.unsqueeze(-1), bh.unsqueeze(-1)), dim=-1)
            bg = bg.reshape(self.ks3 * self.ks3, 2, 1)
            t_matrix = torch.cat((theta, bg), dim=-1)   # kk 2 3
            grid = F.affine_grid(t_matrix, (self.ks3 * self.ks3, B, H, W), align_corners=False)




        bl_map = predict.reshape(B, C, H * W)

        bl_map = bl_map * dmap_unfold

        bl_map = bl_map.transpose(0, 1).reshape(self.ks3 * self.ks3, B, H, W)

        bl_transpose_map = F.grid_sample(bl_map, grid, mode="nearest", align_corners=False)
        bl_transpose_map = bl_transpose_map.transpose(0, 1).reshape(B, self.ks3, self.ks3, H, W)


        alpha = einops.rearrange(bl_transpose_map, "b k1 k2 h w -> (b h w) (k1 k2)") + 1e-5
        # print(alpha.shape, bcoord.shape, beta.shape, ccoord.shape)

        loss_dict = {}


        loss_dict["bg"] = 0.1 * ((1 - seg) * predict).abs().mean()


        segp = (dmap > 0.5).reshape(B * H * W)
        n = segp.long().sum()
        if n > 0.5:
            with torch.no_grad():
                bcoord = self.coord.cuda(device).unsqueeze(0).repeat(n, 1, 1)
                beta = torch.ones(n, 1).cuda(device)
                ccoord = torch.as_tensor([self.ks3 // 2, self.ks3 // 2], device=device).float().reshape(1, 1, 2).repeat(
                    n, 1, 1)
            ot_loss = self.ot(alpha[segp], bcoord, beta, ccoord)
            # print(ot_loss.shape)
            loss_dict["ot"] = torch.sum(ot_loss)/(H*W*B)
        else:
            loss_dict["ot"] = torch.as_tensor(0.0).float().cuda(device)

        loss_dict["all"] = loss_dict["ot"]*self.weight_all + loss_dict["bg"]*self.weight_all

        return loss_dict


def build_loss(cfg):
    return DMap_Loss(cfg)


if __name__ == "__main__":
    t0 = torch.ones(9, 9) * 0.0001
    t0 = torch.as_tensor(t0).cuda().unsqueeze(0).unsqueeze(0).float()
    t1 = torch.zeros(9, 9)
    dotlist = [
        [1, 2], [1, 7],
        [2, 4],
        [3, 1], [3, 6],
        [4, 3],
        [5, 5],
        [6, 2], [6, 7],
        [7, 4],
    ]
    for i, j in dotlist:
        t1[i, j] = 1
    t1 = torch.as_tensor(t1).cuda().unsqueeze(0).unsqueeze(0).float()
    loss = DMap_Loss(None)
    input = {"predict_counting_map": t0}
    tg = {"gt_dmaps": t1}
    print(loss(input, tg))