import torch
import torch.nn.functional as F
import numpy as np
import math

def pad_image(image, labels, padsize = 64):

    h1, w1 = image.shape[-2], image.shape[-1]
    
    h2 = math.ceil(h1 / padsize) * padsize
    w2 = math.ceil(w1 / padsize) * padsize
    h_pad = h2 - h1
    w_pad = w2 - w1
    h_top = h_pad // 2
    h_bottom = h_pad - h_top
    w_left = w_pad // 2
    w_right = w_pad - w_left
    image_pad = F.pad(image, pad=(w_left, w_right, h_top, h_bottom))
    labels["w1h1"] = torch.as_tensor([w1, h1], dtype=torch.long)
    labels["pad_lt"] = torch.as_tensor([w_left, h_top], dtype=torch.long)

    return image_pad, labels

def points_affine(pred_pts, labels):
    i = 0
    w, h = labels["wh"][i][0].item(), labels["wh"][i][1].item()
    w1, h1 = labels["w1h1"][i][0].item(), labels["w1h1"][i][1].item()
    w_left, h_top = labels["pad_lt"][i][0].item(), labels["pad_lt"][i][1].item()
    result_pts=[]
    result_str=[]
    for pt in pred_pts[i]:
        x, y= pt[0] - w_left, pt[1] - h_top
        y, x = round(y * h / h1), round(x * w / w1)
        result_str.append([x, y])
        result_pts += [str(x), str(y)]
    return result_pts, result_str