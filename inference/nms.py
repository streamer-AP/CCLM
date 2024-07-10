import torch
import torch.nn.functional as F
import numpy as np
import math
import einops

def forward_points(model, x):
    assert x.shape[0]==1
    z = model.backbone(x)
    out_dict = model.decoder_layers(z)
    counting_map = out_dict["predict_counting_map"].half()
    offset_map = out_dict["offset_map"].half()

    pred_points = divide_map_to_points(counting_map, offset_map, device=x.device)

    return [pred_points], counting_map

def divide_map_to_points(counting_map, offset_map, device, slide_size=128):
    h, w = counting_map.shape[-2], counting_map.shape[-1]
    h_block = math.ceil(h / slide_size)
    w_block = math.ceil(w / slide_size)
    h_pad = h_block * slide_size - h
    w_pad = w_block * slide_size - w
    slide_counting_map = F.pad(counting_map, pad=(0, w_pad, 0, h_pad))
    slide_counting_map = einops.rearrange(slide_counting_map, "b c (h d) (w e) -> h w b c d e", d=slide_size, e=slide_size)
    slide_offset_map = F.pad(offset_map, pad=(0, w_pad, 0, h_pad))
    slide_offset_map = einops.rearrange(slide_offset_map, "b c (h d) (w e) -> h w b c d e", d=slide_size, e=slide_size)

    pred_points = []
    for i in range(h_block):
        for j in range(w_block):
            block_pts = map_to_points(slide_counting_map[i][j], slide_offset_map[i][j], device=device)
            for x, y in block_pts:
                pred_points.append([x + j * slide_size, y + i * slide_size])
    
    pred_points = points_unsample(pred_points)
    return pred_points


def points_unsample(pred_points):
    return [[2 * x + 1.5, 2 * y + 1.5] for x, y in pred_points]

def gen_local_coord(ks):
    M = torch.arange(ks) - ks // 2
    grid_h, grid_w = torch.meshgrid(M, M, indexing="ij")
    grid = torch.dstack((grid_h, grid_w))

    return grid.reshape(1, -1, 2).transpose(1, 2)

def gen_global_grid(H, W, padding):
    grid_h, grid_w = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid = grid_h * W + grid_w
    grid = grid.reshape(1, H, W, 1).permute(0, 3, 1, 2)
    grid = F.pad(grid, (padding, padding, padding, padding), mode="constant", value=-1)
    return grid


def map_to_points(counting_map, offset_map, device="cuda"):
    '''
    minimize a mask S = argmin_S |1-\sum_{j\in S}c_j| + w \sum_{j\in S} ||z_j - g||^2
    on every pixle g
    '''
    weight_reg = 1.0 / 32
    radius = 8
    window_size = 2 * radius + 1
    padding = radius

    B, _, H, W = counting_map.shape
    counting_map = counting_map * (counting_map > 0.01).half()
    est_cnt = counting_map.sum(dim=[1,2,3]).round().to(torch.int32).cpu().item()
    assert B == 1
    idmap = gen_global_grid(H, W, padding).to(device)   # [1 2 H W]
    
    idmap_unfold = F.unfold(idmap.float(), window_size).to(torch.int32)   # [1 K*K HW]
    
    px_idmap_local = idmap_unfold.transpose(1, 2).squeeze(0)    # [HW K*K]

    cmap_unfold = F.unfold(counting_map, window_size, padding=padding)  # [1 K*K HW]
    omap_unfold = F.unfold(offset_map, window_size, padding=padding)    # [1 2*K*K HW]
    cmap_local = cmap_unfold.transpose(1, 2).squeeze(0)
    omap_local = omap_unfold.transpose(1, 2).reshape(-1, 2, window_size*window_size) # [HW 2 K*K]
    grid_coord = gen_local_coord(window_size).to(device)
    dis = (omap_local + grid_coord).pow(2).sum(dim=1) * weight_reg
    local_px_score = cmap_local * (1 - dis)    # [HW K*K]
    _, local_px_score_id = torch.sort(local_px_score, dim=-1, descending=True)  
    pred_pos = torch.gather(cmap_local, dim=-1, index=local_px_score_id)
    dis_pos = torch.gather(dis, dim=-1, index=local_px_score_id)
    px_id_pos = torch.gather(px_idmap_local, dim=-1, index=local_px_score_id)
    
    pred_cum = torch.cumsum(pred_pos, dim=-1)
    cnt = (1 - pred_cum).abs()
    loc_pos = pred_pos * dis_pos
    loc_cum = torch.cumsum(loc_pos, dim=-1)
    log_prob = cnt + loc_cum

    local_score, stop_id = torch.min(log_prob, dim=-1)  # [HW]

    score, px_id = torch.sort(local_score, dim=0, descending=False)  
    
    score = score.cpu().numpy()
    px_id = px_id.cpu()
    stop_id = stop_id.cpu()[px_id].numpy()
    px_id_pos = px_id_pos.cpu()[px_id, :].numpy()

    pred_pos = pred_pos.cpu()[px_id, :].numpy()
    px_id = px_id.numpy()
    
    px_id_pos = px_id_pos.astype(np.int32)

    px_id = px_id.astype(np.int32)
    pred_pos = pred_pos.astype(np.float32)
    score = score.astype(np.float32)
    stop_id = stop_id.astype(np.int32)

    pred_points = nms(px_id_pos,px_id,pred_pos,score,stop_id,H,W,est_cnt,0.3)

    return pred_points


def nms(px_id_pos,
        px_id,
        pred_pos,
        score,
        stop_id,
        H, W, est_cnt, iou_threshold):

    pt_list = []
    pred_points = []

    n = 0
    for i in range(H * W):
        pt_dict = {}
        t = stop_id[i]
        idpx = px_id_pos[i, :t+1]
        pred = pred_pos[i, :t+1]
        pt_dict["id2cnt"]={k:v for k,v in zip(idpx,pred)}
        pt_dict["cnt"] = (np.sum(pred))
        pt_dict["loc"] = px_id[i]
        pt_dict["i"] = px_id[i] // W
        pt_dict["j"] = px_id[i] % W
        pt_dict["score"] = score[i]

        T = True
        for previous_pt in pt_list:
            if iou(previous_pt, pt_dict) > iou_threshold:
                T = False
                break
        if T:
            if n >= est_cnt:
                break
            else:
                pt_list.append(pt_dict)
                n += 1
    pred_points = []
    for pt in pt_list:
        i, j = pt["i"], pt["j"]
        pred_points.append([j, i])

    return pred_points


def iou(pt_dict1, pt_dict2):


    i1, j1 = pt_dict1["i"], pt_dict1["j"]
    i2, j2 = pt_dict2["i"], pt_dict2["j"]
    if abs(i1 - i2) >= 17 or abs(j1 - j2) >= 17:
        return 0.0

    c1 = pt_dict1["cnt"]
    c2 = pt_dict2["cnt"]

    I = sum([pt_dict2["id2cnt"][k] for k in pt_dict1["id2cnt"].keys()&pt_dict2["id2cnt"].keys()])
    iou = I / (c1 + c2 - I + 1e-6)

    return iou

def mean_loc(id2cnt, cnt, W):
    i, j = 0, 0
    for k, v in id2cnt.items():
        if k != -1:
            i_cur, j_cur = k // W, k % W
            i += i_cur * v
            j += j_cur * v
    i = i / (cnt+ 1e-7)
    j = j / (cnt+ 1e-7)
    return i, j