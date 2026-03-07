import argparse
import json
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

import torch
import torch.nn.functional as F
import cv2
from easydict import EasyDict as edict
from termcolor import cprint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import build_dataset
from misc import utils

from misc.utils import MetricLogger, is_main_process
from models import build_model
from models.utils import module2model
from eingine.utils import reduce_dict, is_main_process, SmoothedValue
from torch.nn import functional as F
from math import sqrt
import numpy as np
from scipy import spatial as ss
import json
from scipy.sparse import coo_matrix
import time
from nms import nms
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

################################################## 
def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]
    def dfs(u):
        for v in graph[u]:
            if vis[v]: continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum): vis[i] = False
        if dfs(a): ans += 1

    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign


def compute_metrics(dist_matrix,match_matrix,pred_num,sigma):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p,:]
        match_matrix[i_pred_p,:] = pred_dist<=sigma
        
    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]
    return tp,fp,fn

def draw_points(img,points):
    r = 4
    for point in points:
        img = cv2.circle(img,(int(point[0]),int(point[1])),r,(0,0,255), 1)
    return img
def draw_dmap(dmap):
    dmap = dmap[0].cpu().numpy()
    dmap = dmap / dmap.max()
    dmap = dmap * 255
    dmap = dmap.astype(np.uint8)
    dmap = cv2.applyColorMap(dmap, cv2.COLORMAP_JET)
    return dmap

@torch.no_grad()
def evaluate_counting_and_locating(ann_file_path, model, data_loader, metric_logger, epoch,
                                   args, vis_path):
    model.eval()
    metric_logger.meters.clear()

    metric_logger.add_meter('mse',
                            SmoothedValue(window_size=1, fmt='{value:.5f}'))
    metric_logger.add_meter('mae',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('mae_sum',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('tp_s',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('fp_s',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('fn_s',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('tp_l',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('fp_l',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('fn_l',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('cnt',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    header = "Test"
    metric_logger.set_header(header)

    sigma_s = 4
    sigma_l = 8

    save_path = vis_path
    os.makedirs(save_path, exist_ok=True)
    id_filename={}
    with open(ann_file_path,"r") as f:
        info=json.load(f)
        id_filename={v["id"]: v["file_name"] for v in info["images"]}
    for inputs, labels in metric_logger.log_every(data_loader):
        inputs = inputs.to(args.gpu)
        assert inputs.shape[0] == 1
        print(f"-------------{labels['id'].item()}----------------------------")
        pred_points, pred_map = forward_points(model, inputs)

        ####################################### draw 

        i = 0
        w, h = labels["wh"][i][0].item(), labels["wh"][i][1].item()
        w1, h1 = labels["w1h1"][i][0].item(), labels["w1h1"][i][1].item()
        result=[]
        for pt in pred_points[i]:
            x, y= pt
            y, x = y * h / h1, x * w / w1
            result.append([x, y])

        img_prefix=args.Dataset.val.img_prefix
        img_path = os.path.join(img_prefix, id_filename[labels["id"][i].item()])
        img=cv2.imread(img_path)
        img = draw_points(img, result)
        cv2.imwrite(os.path.join(save_path, str(labels["id"][i].item())+"_img.jpg"),img)
        dmap = draw_dmap(pred_map[0])
        cv2.imwrite(os.path.join(save_path,str(labels["id"][i].item())+"_dmap.jpg"),dmap)

        ####################################### 
        count_nums = labels["num"].to(args.gpu).float()
        mae = torch.abs(len(pred_points[0]) - count_nums).data.mean()
        mae_sum = torch.abs(pred_map.sum(dim=[1,2,3])- count_nums).data.mean()
        mse = ((len(pred_points[0]) - count_nums)**2).data.mean()

        ########################################
        num = labels["num"][0].item()

        w, h = labels["wh"][0][0].item(), labels["wh"][0][1].item()
        w1, h1 = labels["w1h1"][0][0].item(), labels["w1h1"][0][1].item()
        x_scale, y_scale = w / w1, h / h1
        print("size", w, h, w1, h1, x_scale, y_scale, "len", len(pred_points[0]))
        gt_pts = labels["points"][0][:num].numpy()
        pred_pts=[[x[0]*x_scale,x[1]*y_scale] for x in pred_points[0]]

        ########################################

        tp_s, fp_s, fn_s, tp_l, fp_l, fn_l = [0, 0, 0, 0, 0, 0]
        if len(pred_pts) != 0 and num == 0:
            fp_s = len(pred_pts)
            fp_l = len(pred_pts)

        if len(pred_pts) == 0 and num != 0:
            fn_s = num
            fn_l = num

        if len(pred_pts) != 0 and num != 0:

            pred_pts = np.array(pred_pts)
            gt_pts = np.array(gt_pts)

            dist_matrix = ss.distance_matrix(pred_pts, gt_pts, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
            tp_s, fp_s, fn_s = compute_metrics(dist_matrix, match_matrix,
                                               pred_pts.shape[0], sigma_s)
            tp_l, fp_l, fn_l = compute_metrics(dist_matrix, match_matrix,
                                               pred_pts.shape[0], sigma_l)

        tp_s = torch.as_tensor(tp_s, device=args.gpu)
        fp_s = torch.as_tensor(fp_s, device=args.gpu)
        fn_s = torch.as_tensor(fn_s, device=args.gpu)
        tp_l = torch.as_tensor(tp_l, device=args.gpu)
        fp_l = torch.as_tensor(fp_l, device=args.gpu)
        fn_l = torch.as_tensor(fn_l, device=args.gpu)

        ########################################
        loss_dict_reduced = reduce_dict(
            {
                "mae": mae,
                "mae_sum": mae_sum,
                "mse": mse,
                "tp_s": tp_s,
                "fp_s": fp_s,
                "fn_s": fn_s,
                "tp_l": tp_l,
                "fp_l": fp_l,
                "cnt": torch.as_tensor(1., device=args.gpu),
                "fn_l": fn_l,
            },
            average=True)

        metric_logger.update(mae=loss_dict_reduced['mae'])
        metric_logger.update(mae_sum=loss_dict_reduced['mae_sum'])
        metric_logger.update(mse=loss_dict_reduced['mse'])
        metric_logger.update(tp_s=loss_dict_reduced['tp_s'])
        metric_logger.update(fp_s=loss_dict_reduced['fp_s'])
        metric_logger.update(fn_s=loss_dict_reduced['fn_s'])
        metric_logger.update(tp_l=loss_dict_reduced['tp_l'])
        metric_logger.update(fp_l=loss_dict_reduced['fp_l'])
        metric_logger.update(fn_l=loss_dict_reduced['fn_l'])
        metric_logger.update(cnt=loss_dict_reduced['cnt'])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.total for k, meter in metric_logger.meters.items()}
    print(metric_logger.meters["cnt"].total,metric_logger.meters["cnt"].count)
    ap_s = stats['tp_s'] / (stats['tp_s'] + stats['fp_s'] + 1e-7)
    ar_s = stats['tp_s'] / (stats['tp_s'] + stats['fn_s'] + 1e-7)
    f1m_s = 2 * ap_s * ar_s / (ap_s + ar_s+ 1e-7)

    ap_l = stats['tp_l'] / (stats['tp_l'] + stats['fp_l'] + 1e-7)
    ar_l = stats['tp_l'] / (stats['tp_l'] + stats['fn_l'] + 1e-7)
    f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l+ 1e-7)
    stats["ap_s"] = ap_s
    stats["ar_s"] = ar_s
    stats["f1m_s"] = f1m_s
    stats["ap_l"] = ap_l
    stats["ar_l"] = ar_l
    stats["f1m_l"] = f1m_l
    stats["mae"] = stats["mae"]/ stats["cnt"]
    stats["mae_sum"] = stats["mae_sum"]/ stats["cnt"]
    stats["mse"] = sqrt(stats["mse"] / stats["cnt"])
    return stats

################################################## 

################################################## 
@torch.no_grad()
def forward_points(model, x):
    assert x.shape[0]==1
    z = model.backbone(x)
    out_dict = model.decoder_layers(z)
    counting_map = out_dict["predict_counting_map"].detach().float()
    offset_map = out_dict["offset_map"].detach().float()
    pred_points = map_to_points(counting_map, offset_map,device=x.device)
    return [pred_points], counting_map

def gen_local_coord(ks):
    M = torch.arange(ks) - ks // 2
    grid_h, grid_w = torch.meshgrid(M, M, indexing="ij")
    grid = torch.dstack((grid_h, grid_w))

    return grid.reshape(1, -1, 2).transpose(1, 2)

def gen_global_grid(H, W, padding):
    grid_h, grid_w = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid = grid_h * W + grid_w
    grid = grid.reshape(1, H, W, 1).permute(0, 3, 1, 2).float()
    grid = F.pad(grid, (padding, padding, padding, padding), mode="constant", value=-1)
    return grid

@torch.no_grad()
def map_to_points(counting_map, offset_map, device="cuda"):
    '''
    minimize a mask S = argmin_S |1-\sum_{j\in S}c_j| + w \sum_{j\in S} ||z_j - g||^2
    on every pixle g
    '''
    weight_reg = 1.0 / 128
    radius = 8
    window_size = 2 * radius + 1
    padding = radius

    B, _, H, W = counting_map.shape
    est_cnt = counting_map.sum(dim=[1,2,3]).round().long().cpu().item() 
    print("est_cnt", est_cnt, H, W)
    # est_cnt = torch.clamp_max(est_cnt, H * W - 1)
    assert B == 1
    
    idmap = gen_global_grid(H, W, padding).to(device)   # [1 2 H W]
    idmap_unfold = F.unfold(idmap, window_size).long()    # [1 K*K HW]
    px_idmap_local = idmap_unfold.transpose(1, 2).reshape(-1, window_size*window_size)    # [HW K*K]
    
    cmap_unfold = F.unfold(counting_map, window_size, padding=padding)  # [1 K*K HW]
    omap_unfold = F.unfold(offset_map, window_size, padding=padding)    # [1 2*K*K HW]
    cmap_local = cmap_unfold.transpose(1, 2).reshape(-1, window_size*window_size) 
    omap_local = omap_unfold.transpose(1, 2).reshape(-1, 2, window_size*window_size) # [HW 2 K*K]
    grid_coord = gen_local_coord(window_size).to(device)

    dis = (omap_local + grid_coord).pow(2).sum(dim=1) * weight_reg
    local_px_score = cmap_local * (1 - dis)    # [HW K*K]
    _, local_px_score_id = torch.sort(local_px_score, dim=-1, descending=True)
    local_px_score_id = local_px_score_id[:, :50]
    pred_pos = torch.gather(cmap_local, dim=-1, index=local_px_score_id)
    dis_pos = torch.gather(dis, dim=-1, index=local_px_score_id)
    px_id_pos = torch.gather(px_idmap_local, dim=-1, index=local_px_score_id)
    pred_cum = torch.cumsum(pred_pos, dim=-1)
    cnt = (1 - pred_cum).abs()
    loc_pos = pred_pos * dis_pos
    loc_cum = torch.cumsum(loc_pos, dim=-1)
    log_prob = cnt + loc_cum
    log_prob = log_prob
    local_score, stop_id = torch.min(log_prob, dim=-1)  # [HW]
    score, px_id = torch.sort(local_score, dim=0, descending=False)  
    stop_id = stop_id[px_id]
    px_id_pos = px_id_pos[px_id, :50]
    pred_pos = pred_pos[px_id, :50]
    stop_id = stop_id.cpu().numpy()
    score = score.cpu().numpy()
    px_id = px_id.cpu().numpy()
    px_id_pos = px_id_pos.cpu().numpy()
    pred_pos = pred_pos.cpu().numpy()
    st = score <= 0.5
    px_id_pos= px_id_pos[st, :]
    px_id = px_id[st]
    pred_pos = pred_pos[st, :]
    score = score[st]
    stop_id = stop_id[st]
    px_id_pos = px_id_pos.astype(np.int32)
    px_id = px_id.astype(np.int32)
    pred_pos = pred_pos.astype(np.float32)
    score = score.astype(np.float32)
    stop_id = stop_id.astype(np.int32)
    iou_threshold = 0.3 
    pred_points = nms(px_id_pos,
                      px_id,
                      pred_pos,
                      score,
                      stop_id,
                      H,W,
                      est_cnt,
                      px_id_pos.shape[0],
                      iou_threshold)
    

    return pred_points




def main(args,ckpt_path):
    utils.init_distributed_mode(args)
    model = model_without_ddp = build_model(args.Model)
    vis_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "draw")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if is_main_process():
        print("=> loading checkpoint '{}'".format(os.path.join(args.Saver.save_dir, ckpt_path)))
        print(ckpt["states"])
        print("epoch: {}".format(ckpt["epoch"]))
    state_dict = module2model(ckpt['model'])

    model_dict = model.state_dict()
    load_param_dict = {k: v for k, v in state_dict.items() if k in model_dict and k.find("grid") == -1}
    model_dict.update(load_param_dict)
    model_without_ddp.load_state_dict(model_dict)
    model.cuda().eval()
    dataset_val = build_dataset(image_set='val', args=args.Dataset.val)
 
    loader_val = DataLoader(dataset_val,
                            batch_size=args.Dataset.val.batch_size,
                            sampler=None,
                            shuffle=False,
                            num_workers=args.Dataset.val.num_workers,
                            pin_memory=True)

    logger = MetricLogger(args.Logger)

    ann_file_path = args.Dataset.val.ann_file
    stats = evaluate_counting_and_locating(ann_file_path, model, loader_val, logger, 0, args, vis_path)

    test_log_stats = {
        **{f'val_{k}': v
        for k, v in stats.items()}
    }
    print(test_log_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DenseMap Head ")
    parser.add_argument("--config", default="/data1/YanZiheng/CCLM/configs/sta/FIDT_test.json")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--ckpt",default="/data1/YanZiheng/hrcrowd/outputs/202601051629_rank_hrnet48_beta_256_gamma_20_labmda_01_R8/checkpoints/best.pth")
    
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
        cfg = edict(configs)
    print(cfg)

    main(cfg, args.ckpt)
