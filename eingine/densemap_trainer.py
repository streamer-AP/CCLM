import torch
from typing import Iterable
import einops
import math
from .utils import reduce_dict,  get_total_grad_norm, SmoothedValue,compute_metrics
from torch.nn import functional as F
from math import sqrt
import numpy as np
from scipy import spatial as ss
from matplotlib import pyplot as plt
import cv2
def draw_dmap(dmap):
    dmap = dmap / dmap.max()
    dmap = dmap * 255
    dmap = dmap.astype(np.uint8)
    dmap = cv2.applyColorMap(dmap, cv2.COLORMAP_JET)
    return dmap

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    metric_logger: object, scaler:torch.cuda.amp.GradScaler,epoch, args):
    model.train()
    criterion.train()

    metric_logger.meters.clear() 

    header = 'Epoch: [{}]'.format(epoch)
    metric_logger.set_header(header)
    criterion.update_weight(epoch)
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for inputs, labels in metric_logger.log_every(data_loader):

        optimizer.zero_grad()
        inputs = inputs.to(args.gpu)
        
        outputs_dict = model(inputs,labels["exampler"])
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
    metric_logger.add_meter('tp_10',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('fp_10',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('fn_10',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    metric_logger.add_meter('cnt',
                            SmoothedValue(window_size=1, fmt='{value:.1f}'))
    header = "Test"
    metric_logger.set_header(header)

    sigma_s = 4
    sigma_l = 8
    sigma_10=10
    draw=0
    for inputs, labels in metric_logger.log_every(data_loader):
        inputs = inputs.to(args.gpu)
        assert inputs.shape[0] == 1
        if args.distributed:
            pred_points, out_dict =model.module.forward_points(inputs,labels["exampler"],threshold=0.9,loc_kernel_size=11)
        else:
            pred_points, out_dict =model.forward_points(inputs,labels["exampler"],threshold=0.9,loc_kernel_size=11)
        pred_map=out_dict["predict_counting_map"].detach().float()
        if draw%101==1 and args.gpu==0:
            atten_map=out_dict["atten_map"]
            atten_map=atten_map[0,0].detach().cpu().numpy()
            plt.imshow(atten_map)
            plt.savefig(f"debugs/val_atten_map_{draw}.png")
            img=inputs[0][:3].detach().cpu().numpy()
            img=np.transpose(img,(1,2,0))
            plt.imshow(img)
            plt.savefig(f"debugs/val_img_{draw}.png")
            dmap=pred_map[0,0].detach().cpu().numpy()
            plt.imshow(dmap)
            plt.savefig(f"debugs/val_pred_map_{draw}.png")
            roi_feature=out_dict["roi_feature"]
            roi_feature=roi_feature[0].detach().cpu().numpy()
            plt.imshow(roi_feature[0][0])
            plt.savefig(f"debugs/val_roi_feature_{draw}.png")
        draw+=1
        count_nums = labels["num"].to(args.gpu).float()
        mae = torch.abs(len(pred_points[0]) - count_nums).data.mean()
        # clamp_mask=(pred_map>0.005).float()
        # clamp_pred_map=pred_map*clamp_mask
        mae_sum = torch.abs(pred_map.sum(dim=[1,2,3])- count_nums).data.mean()
        mse = ((len(pred_points[0]) - count_nums)**2).data.mean()

        num = labels["num"][0].item()

        w, h = labels["wh"][0][0].item(), labels["wh"][0][1].item()
        w1, h1 = labels["w1h1"][0][0].item(), labels["w1h1"][0][1].item()
        x_scale, y_scale = w / w1, h / h1
        gt_pts = labels["points"][0][:num].numpy()
        pred_pts=[[x[0]*x_scale,x[1]*y_scale] for x in pred_points[0]]


        tp_s, fp_s, fn_s, tp_l, fp_l, fn_l,tp_10,fp_10,fn_10 = [0, 0, 0, 0, 0, 0, 0,0,0]
        if len(pred_pts) != 0 and num == 0:
            fp_s = len(pred_pts)
            fp_l = len(pred_pts)
            fp_10=len(pred_pts)

        if len(pred_pts) == 0 and num != 0:
            fn_s = num
            fn_l = num
            fn_10=num

        if len(pred_pts) != 0 and num != 0:

            pred_pts = np.array(pred_pts)
            gt_pts = np.array(gt_pts)

            dist_matrix = ss.distance_matrix(pred_pts, gt_pts, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
            tp_s, fp_s, fn_s = compute_metrics(dist_matrix, match_matrix,
                                               pred_pts.shape[0], sigma_s)
            tp_l, fp_l, fn_l = compute_metrics(dist_matrix, match_matrix,
                                               pred_pts.shape[0], sigma_l)
            tp_10, fp_10, fn_10 = compute_metrics(dist_matrix, match_matrix,
                                                  pred_pts.shape[0], sigma_10)

        tp_s = torch.as_tensor(tp_s, device=args.gpu)
        fp_s = torch.as_tensor(fp_s, device=args.gpu)
        fn_s = torch.as_tensor(fn_s, device=args.gpu)
        tp_l = torch.as_tensor(tp_l, device=args.gpu)
        fp_l = torch.as_tensor(fp_l, device=args.gpu)
        fn_l = torch.as_tensor(fn_l, device=args.gpu)
        tp_10 = torch.as_tensor(tp_10, device=args.gpu)
        fp_10 = torch.as_tensor(fp_10, device=args.gpu)
        fn_10 = torch.as_tensor(fn_10, device=args.gpu)
        
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
                "tp_10": tp_10,
                "fp_10": fp_10,
                "fn_10": fn_10
                
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
        metric_logger.update(tp_10=loss_dict_reduced['tp_10'])
        metric_logger.update(fp_10=loss_dict_reduced['fp_10'])
        metric_logger.update(fn_10=loss_dict_reduced['fn_10'])
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
    
    ap_10=stats['tp_10'] / (stats['tp_10'] + stats['fp_10'] + 1e-7)
    ar_10=stats['tp_10'] / (stats['tp_10'] + stats['fn_10'] + 1e-7)
    f1m_10=2 * ap_10 * ar_10 / (ap_10 + ar_10+ 1e-7)
    stats["ap_10"] = ap_10
    stats["ar_10"] = ar_10
    stats["f1m_10"] = f1m_10
    
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
    metric_logger.meters.clear() 

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
            metric_logger.update(mae=loss_dict_reduced['mae'])
            metric_logger.update(mse=loss_dict_reduced['mse'])
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        stats = {
            k: meter.global_avg
            for k, meter in metric_logger.meters.items()
        }
    return stats