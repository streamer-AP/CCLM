import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from scipy import spatial as ss
from torch.amp import autocast

def gen_coord(ks, sigma):
    coord = torch.arange(ks).float() - ks // 2
    coord_h = coord.unsqueeze(1).repeat(1, ks)
    coord_w = coord.unsqueeze(0).repeat(ks, 1)

    gaussian = torch.exp(-0.5 * (coord_h.abs().pow(2) + coord_w.abs().pow(2)) / sigma ** 2)
    
    return coord_h.reshape(-1), coord_w.reshape(-1), gaussian.unsqueeze(0)

def gen_dis(ks):
    coord = torch.arange(ks).float() - ks // 2
    coord_h = coord.unsqueeze(1).repeat(1, ks)
    coord_w = coord.unsqueeze(0).repeat(ks, 1)

    dis =  (coord_h.abs().pow(2) + coord_w.abs().pow(2)) / (ks // 2) ** 2
    
    return dis.reshape(-1)

class DMap_Loss(nn.Module):
    def __init__(self, radius=16, stride=2):
        super().__init__()
        self.radius = radius
        self.window_size = 2 * self.radius + 1
        self.padding = self.window_size // 2
        self.stirde = stride
        self.sigma = radius * 0.5

        # dis = gen_dis(self.window_size)
        # self.register_buffer("dis", dis)
        # coord_h, coord_w, gaussian = gen_coord(self.window_size, self.sigma)
        # self.register_buffer("cdh", coord_h)
        # self.register_buffer("cdw", coord_w)
        # self.register_buffer("gauss", gaussian)
        
        self.eta = 0.01
        self.weight_bg = 10
        self.weight_ann = 100
        # self.weight_reg = 100

        

    @autocast("cuda")
    def forward(self, inputs, targets):
        loss_dict = {}
        loss_dict["bg"] = 0
        loss_dict["ann"] = 0
        # loss_dict["reg"] = 0

        predict = inputs["predict_counting_map"]
        feature = inputs["feature"]
        weight_model = inputs["weight_model"]
        num = targets["num"]
        device = predict.device
        B, C, H, W = predict.shape
        assert H == feature.shape[-2] and W == feature.shape[-1]

        predict = F.pad(predict, (self.padding, self.padding, self.padding, self.padding))
        feature = F.pad(feature, (self.padding, self.padding, self.padding, self.padding))

        for b in range(B):
            N = int(num[b].item())
            pred_map = predict[b, 0]
            fea_map = feature[b, :]
            if N == 0:
                loss_bg = pred_map.abs().sum() / (H * W)
                loss_ann = torch.as_tensor(0.0, dtype=pred_map.dtype, device=device)
                # loss_reg = torch.as_tensor(0.0, dtype=pred_map.dtype, device=device)
            else:
                gt_pts = np.round(targets["points"][b, :N, :].numpy() / self.stirde)
                pt_coord = [[int(gt_pts[n][1]), int(gt_pts[n][0])] for n in range(N)]
                bg_mask = torch.ones_like(pred_map, requires_grad=False)
                # 从特征图上裁剪出 以gt为中心的正方形
                fea_local = []
                for n in range(N):
                    pti, ptj = pt_coord[n]
                    fea = fea_map[:, pti - self.radius + self.padding: pti + self.radius + self.padding + 1,\
                                   ptj - self.radius + self.padding: ptj + self.radius + self.padding + 1]
                    fea_local.append(fea.unsqueeze(0))
                    bg_mask[pti - self.radius + self.padding: pti + self.radius + self.padding + 1, \
                                ptj - self.radius + self.padding: ptj + self.radius + self.padding + 1] *= 0
                fea_local = torch.cat(fea_local, dim=0)    # [N C K K]
                weight_density = (weight_model(fea_local).squeeze(1).exp())   # [N K K]

                ###################### 把weight 归一化
                weight_sum_mask = torch.zeros_like(pred_map)
                for n in range(N):
                    pti, ptj = pt_coord[n]
                    weight_sum_mask[pti - self.radius + self.padding: pti + self.radius + self.padding + 1,\
                                   ptj - self.radius + self.padding: ptj + self.radius + self.padding + 1] \
                        += weight_density[n]
                # print("sum", weight_sum_mask)
                
                weight_den_norm = []
                for n in range(N):
                    pti, ptj = pt_coord[n]
                    weight_den_norm.append(
                    (weight_density[n] / (weight_sum_mask[pti - self.radius + self.padding: pti + self.radius + self.padding + 1,\
                                   ptj - self.radius + self.padding: ptj + self.radius + self.padding + 1])).unsqueeze(0)
                    )
                weight_den_norm = torch.cat(weight_den_norm, dim=0)
                # print("weight", weight_den_norm)
                ###################### 把weight 归一化

                # 回归每个区域内的总密度
                loss_ann = 0
                # loss_reg = 0
                for n in range(N):
                    pti, ptj = pt_coord[n]
                    den = pred_map[pti - self.radius + self.padding: pti + self.radius + self.padding + 1,\
                                   ptj - self.radius + self.padding: ptj + self.radius + self.padding + 1]
                    den = den * weight_den_norm[n]
                    # print("den", den)
                    loss_ann += ((den).sum() - 1).abs() / (H * W)

                    # 回归标注点的坐标 根据密度和一个高斯核的先验进行加权平均
                    # reg_weight = F.normalize((den * self.gauss.to(device)).reshape(-1), p=1, dim=0)
                    # loss_reg += ( (reg_weight * self.cdh.to(device)).sum().abs() + (reg_weight * self.cdw.to(device)).sum().abs() ) / (H * W)

                loss_bg = (pred_map * bg_mask).abs().sum() / (H * W)

            loss_dict["bg"] += loss_bg
            loss_dict["ann"] += loss_ann
            # loss_dict["reg"] += loss_reg

        loss_dict["bg"] /= B
        loss_dict["ann"] /= B
        # loss_dict["reg"] /= B

        loss_dict["all"] = self.weight_bg * loss_dict["bg"] \
                           + self.weight_ann * loss_dict["ann"] 
                        #    + self.weight_reg * loss_dict["reg"]

        return loss_dict


def build_loss(cfg):
    return DMap_Loss()



if __name__ == "__main__":
    loss = DMap_Loss(2, 1)

    weight_model = lambda x: x * 0 + torch.ones_like(x)

    x = torch.zeros(6, 6).float()
    x[1, 1] = 0.8
    x[1, 5] = 1.2
    x[4, 3] = 0.8
    x = x.cuda().unsqueeze(0).unsqueeze(0)
    fea = torch.zeros_like(x)
    inputs = {"predict_counting_map": x, "feature": fea, "weight_model": weight_model}
    print(x)
    x.requires_grad = True
    N = 3
    y = torch.as_tensor([[1, 1], [5, 1], [3, 4]]).float().reshape(1, N, 2)
    y.requires_grad = False
    targets = {"points": y, "num": torch.as_tensor(N).reshape(1)}
    loss_dict = loss(inputs, targets)
    print(loss_dict)
    loss_dict["all"].backward()
    print(loss_dict)

    # r = 10
    # coord_h, coord_w, gaussian = gen_coord(2*r + 1, 0.5 * r)
    # print(gaussian)