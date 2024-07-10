import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from scipy import spatial as ss

def gen_label(targets, B, C, H, W):
    for b in range(B):
        pass

class DMap_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        loss_dict = {}

        map1 = inputs["pred_map1"]
        map2 = inputs["pred_map2"]
        map3 = inputs["pred_map3"]
        num = targets["num"]
        pts = targets["points"]
        device = map1.device
        B, C, H, W = map1.shape

        
        return loss_dict


def build_loss(cfg):
    return DMap_Loss()



if __name__ == "__main__":
    loss = DMap_Loss(3, 1)

    x = torch.zeros(6, 6).float()
    x[1, 1] = 0.8
    x[1, 5] = 1.2
    x[4, 3] = 0.8
    x = x.cuda().unsqueeze(0).unsqueeze(0)
    scale_map = torch.ones_like(x) * 4
    inputs = {"predict_counting_map": x, "scale_map": scale_map}
    N = 3
    y = torch.as_tensor([[1, 1], [5, 1], [3, 4]]).float().reshape(1, N, 2)
    targets = {"gt_points": y, "num": torch.as_tensor(N).reshape(1)}
    loss_dict = loss(inputs, targets)
    print(loss_dict)

