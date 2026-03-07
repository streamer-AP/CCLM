import imp
import torch
from typing import Iterable
import cv2
import os
import einops
import time
import math
from .utils import reduce_dict, is_main_process, get_total_grad_norm, SmoothedValue, predict_map2coord
from torch.nn import functional as F
from math import sqrt
import torch.distributed as dist
import numpy as np
from scipy import spatial as ss

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

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    metric_logger: object, scaler:torch.cuda.amp.GradScaler,epoch, args):
    model.train()
    criterion.train()

    metric_logger.meters.clear() 

    header = 'Epoch: [{}]'.format(epoch)
    metric_logger.set_header(header)
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for inputs, labels in metric_logger.log_every(data_loader):
        # rank = dist.get_rank()
        # log_dir = "/home/xinyan/workspace/counting/hrcrowd/outputs/debug"
        # log_path = os.path.join(log_dir, f"log_{rank}.txt") 
        # os.makedirs(log_dir, exist_ok=True)
        # file = open(log_path, "a+")
        # img_id = labels["id"]
        # file.write(f"{img_id}\n") 

        optimizer.zero_grad()
        inputs = inputs.to(args.gpu)
        
        outputs_dict = model(inputs)
        loss_dict = criterion(outputs_dict, labels)
        all_loss = loss_dict["all"]

        loss_dict_reduced = reduce_dict(loss_dict)
        all_loss_reduced = loss_dict_reduced["all"]
        loss_value = all_loss_reduced.item()

        scaler.scale(all_loss).backward()
        scaler.unscale_(optimizer)

        if args.Misc.clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.Misc.clip_max_norm)
        else:
            grad_total_norm = get_total_grad_norm(model.parameters(),
                                                  args.Misc.clip_max_norm)

        scaler.step(optimizer)
        scaler.update()

        for k in loss_dict_reduced.keys():
            metric_logger.update(**{k: loss_dict_reduced[k]})
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # file.write(f"end_iter\n") 
        # file.close()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_counting_and_locating(model, data_loader, metric_logger, epoch,
                                   args):
    model.eval()
    # criterion.eval()
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

    for inputs, labels in metric_logger.log_every(data_loader):
        inputs = inputs.to(args.gpu)
        assert inputs.shape[0] == 1
        if args.distributed:
            pred_points, pred_map=model.module.forward_points(inputs,threshold=0.9,loc_kernel_size=7)
        else:
            pred_points, pred_map=model.forward_points(inputs,threshold=0.9,loc_kernel_size=7)
        count_nums = labels["num"].to(args.gpu).float()
        mae = torch.abs(len(pred_points[0]) - count_nums).data.mean()
        # clamp_mask=(pred_map>0.005).float()
        # clamp_pred_map=pred_map*clamp_mask
        mae_sum = torch.abs(pred_map.sum(dim=[1,2,3])- count_nums).data.mean()
        mse = ((len(pred_points[0]) - count_nums)**2).data.mean()

        ########################################
        #读取边长 和点坐标
        num = labels["num"][0].item()

        w, h = labels["wh"][0][0].item(), labels["wh"][0][1].item()
        w1, h1 = labels["w1h1"][0][0].item(), labels["w1h1"][0][1].item()
        x_scale, y_scale = w / w1, h / h1
        gt_pts = labels["points"][0][:num].numpy()
        pred_pts=[[x[0]*x_scale,x[1]*y_scale] for x in pred_points[0]]

        ########################################

        # 计算定位指标
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


@torch.no_grad()
def evaluate_sliding_counting(model, criterion, data_loader, metric_logger,
                              drawer, epoch, args):
    model.eval()
    # criterion.eval()
    metric_logger.meters.clear()  # 每一轮开始前把所有的变量都删了

    metric_logger.add_meter('mse',
                            SmoothedValue(window_size=1, fmt='{value:.5f}'))
    metric_logger.add_meter('mae',
                            SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = "Test"
    metric_logger.set_header(header)
    with torch.no_grad():
        for inputs, labels in metric_logger.log_every(data_loader):
            inputs = inputs.to(args.gpu)
            B, _, H, W = inputs.shape
            assert H % 256 == 0 and W % 256 == 0 and B == 1, "{}, {}, {}".format(
                B, H, W)
            n1, n2 = H // 256, W // 256
            inputs = einops.rearrange(
                inputs,
                "B C (n1 d1) (n2 d2) -> (B n1 n2) C d1 d2",
                d1=256,
                d2=256)
            N = inputs.shape[0]
            m = 8
            T = math.ceil(N / m)
            for i in range(T):
                if i == T - 1:
                    pmap = model(inputs[i * m:])["predict_counting_map"]
                else:
                    pmap = model(inputs[i * m:(i + 1) *
                                        m])["predict_counting_map"]
                if i == 0:
                    predict_counting_map = pmap
                else:
                    predict_counting_map = torch.cat(
                        (predict_counting_map, pmap), dim=0)

            predict_counting_map = einops.rearrange(
                predict_counting_map,
                "(B n1 n2) C t1 t2 -> B C (n1 t1) (n2 t2)",
                n1=n1,
                n2=n2)

            count_nums = labels["num"].to(args.gpu).float()
            mae = (torch.abs(
                torch.sum(predict_counting_map, (1, 2, 3)) -
                count_nums)).data.mean()
            mse = ((torch.sum(predict_counting_map,
                              (1, 2, 3)) - count_nums)**2).data.mean()
            loss_dict_reduced = reduce_dict({"mae": mae, "mse": mse})
            # if is_main_process():
            #     drawer(epoch=epoch,inputs=inputs,outputs=predict_counting_map,header="Test")
            metric_logger.update(mae=loss_dict_reduced['mae'])
            metric_logger.update(mse=loss_dict_reduced['mse'])

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        stats = {
            k: meter.global_avg
            for k, meter in metric_logger.meters.items()
        }
    return stats