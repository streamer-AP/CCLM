import argparse
import json
import os

import math
import cv2
import torch
import numpy as np
from scipy import spatial as ss
from matplotlib import pyplot as plt

from termcolor import cprint
from easydict import EasyDict as edict

import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import build_dataset
from misc import utils
from misc.utils import MetricLogger, is_main_process, SmoothedValue
from models import build_model
from models.utils import module2model
from eingine.utils import reduce_dict, is_main_process, hungarian
from inference.nms import divide_map_to_points
from inference.localmaximum import forward_points as forward_points_local

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

@torch.no_grad()
def dmap_render(out, offset):
    B, _, H, W = out.shape
    grid_h, grid_w = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid = torch.dstack((grid_h, grid_w)).to(out.device)
    grid = grid.reshape(1, H, W, 2).permute(0, 3, 1, 2)
    coord = offset + grid
    coord_h = coord[:, 0, :, :].clamp(0, H-1).round().long()
    coord_w = coord[:, 1, :, :].clamp(0, W-1).round().long()
    coord_id = coord_h * H + coord_w    # [B H W]
    coord_id = coord_id.reshape(B, 1, -1)

    out_map = torch.zeros(B, 1, H * W, dtype=out.dtype, device=out.device)
    out_map = torch.scatter_add(out_map, dim=2, index=coord_id, src=out.reshape(B, 1, -1))
    out_map = out_map.reshape(B, 1, H, W)
    return out_map


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
    return tp,fp,fn,tp_pred_index,fp_pred_index,fn_gt_index

def draw_points(img,points,color=(0,0,255)):
    r = 8
    for point in points:
        img=cv2.rectangle(img,(int(point[0])-r,int(point[1])-r),(int(point[0])+r,int(point[1])+r),color,2)
    return img
def draw_dmap(dmap):
    dmap = dmap[0]
    dmap = dmap / dmap.max()
    dmap = dmap * 255
    dmap = dmap.astype(np.uint8)
    dmap = cv2.applyColorMap(dmap, cv2.COLORMAP_JET)
    return dmap

@torch.no_grad()
def evaluate_counting_and_locating(model, data_loader, args, ):
    model.eval()
    logger = MetricLogger(args.Logger)
    logger.meters.clear()
    meter_names=["mae","count_mae","mse","tp_s","fp_s","fn_s","tp_l","fp_l","fn_l","cnt"]
    meters=[SmoothedValue(window_size=1, fmt='{value:.5f}') for _ in range(len(meter_names))]
    logger.add_meters(meter_names,meters)
    header = "Test"
    logger.set_header(header)
    sigma_s = 8
    sigma_l = 16

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    id_filename={}
    with open(args.Dataset.test.ann_file,"r") as f:
        info=json.load(f)
        id_filename={v["id"]: v["file_name"] for v in info["images"]}
    f=open(os.path.join(save_path,"predict.txt"),"w")
    for inputs, labels in logger.log_every(data_loader):
        inputs = inputs.to(args.gpu)
        assert inputs.shape[0] == 1
        print(inputs.shape)
        pred_points, pred_map, offsetmap,attention_map = forward_points(model, inputs,labels["exampler"])
        i = 0
        hf, wf = offsetmap.shape[-2], offsetmap.shape[-1]

        
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
        max_edge = max(img.shape[0], img.shape[1])


        scale = 1024 / max_edge
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        h1, w1 = img.shape[0],  img.shape[1]
        padsize = 64
        h2 = math.ceil(h1 / padsize) * padsize
        w2 = math.ceil(w1 / padsize) * padsize
        h_pad = h2 - h1
        w_pad = w2 - w1
        img_=np.zeros((h2, w2, 3), dtype=np.uint8)
        img_[:h1, :w1, :] = img
        img=img_
        pred_map_=pred_map[0].detach().cpu().numpy()
        np.save(os.path.join(save_path,str(labels["id"][i].item())+"_dmap.npy"),pred_map_)
        dmap = draw_dmap(pred_map_)
        dmap=cv2.resize(dmap,(dmap.shape[1]*2,dmap.shape[0]*2))
        dmap = cv2.addWeighted(img, 0.6, dmap, 0.4, 0)

        count_nums = labels["num"].to(args.gpu).float()
        mae = torch.abs(len(pred_points[0]) - count_nums).data.mean()
        count_mae = torch.abs(pred_map.sum(dim=[1,2,3])- count_nums).data.mean()
        mse = ((len(pred_points[0]) - count_nums)**2).data.mean()

        num = labels["num"][0].item()

        w, h = labels["wh"][0][0].item(), labels["wh"][0][1].item()
        w1, h1 = labels["w1h1"][0][0].item(), labels["w1h1"][0][1].item()
        x_scale, y_scale = w / w1, h / h1
        gt_pts = labels["points"][0][:num].numpy()
        pred_pts=[[x[0]*x_scale,x[1]*y_scale] for x in pred_points[0]]

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
            tp_s, fp_s, fn_s, tp_s_idx, fp_s_idx, fn_s_idx = compute_metrics(dist_matrix, match_matrix,
                                               pred_pts.shape[0], sigma_s)
            tp_l, fp_l, fn_l, tp_l_idx, fp_l_idx, fn_l_idx = compute_metrics(dist_matrix, match_matrix,
                                               pred_pts.shape[0], sigma_l)

        if args.vis:
            tp_points=[]
            if len(pred_pts) != 0:
                for idx in tp_s_idx:
                    tp_points.append(pred_points[0][idx])
                dmap=draw_points(dmap,tp_points,(0,255,0))
                fp_points=[]
                for idx in fp_s_idx:
                    fp_points.append(pred_points[0][idx])
                dmap=draw_points(dmap,fp_points,(255,0,0))
                fn_points=[]
                for idx in fn_s_idx:
                    fn_points.append((gt_pts[idx][0]*w1/w,gt_pts[idx][1]*h1/h))
                dmap=draw_points(dmap,fn_points,(0,0,255))
            for box in labels["exampler"][0]:
                box=box.cpu().numpy()
                box=box.astype(np.int32)
                x,y,w,h=box
                cv2.rectangle(dmap,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.imwrite(os.path.join(save_path,str(labels["id"][i].item())+f"_dmap_{tp_l}_{fp_l}_{fn_l}.jpg"),dmap)
            attention_map=attention_map[0,0].detach().cpu().numpy()
            plt.imshow(attention_map)
            plt.savefig(os.path.join(save_path,str(labels["id"][i].item())+f"_atten_map_{tp_l}_{fp_l}_{fn_l}.jpg"))
            f.write(str(labels["id"][0].item())+" "+str(len(pred_pts)) )

        for pt in pred_pts:
            f.write(" "+str(pt[0])+" "+str(pt[1]))
        f.write("\n")
        
        tp_s = torch.as_tensor(tp_s, device=args.gpu)
        fp_s = torch.as_tensor(fp_s, device=args.gpu)
        fn_s = torch.as_tensor(fn_s, device=args.gpu)
        tp_l = torch.as_tensor(tp_l, device=args.gpu)
        fp_l = torch.as_tensor(fp_l, device=args.gpu)
        fn_l = torch.as_tensor(fn_l, device=args.gpu)

        stats = reduce_dict(
            {
                "mae": mae,
                "count_mae": count_mae,
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
        logger.update(**stats)

    logger.synchronize_between_processes()
    stats = {k: meter.total for k, meter in logger.meters.items()}
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
    stats["locate_mae"] = stats["mae"]/ stats["cnt"]
    stats["count_mae"] = stats["count_mae"]/ stats["cnt"]
    stats["rmse"] = math.sqrt(stats["mse"] / stats["cnt"])
    return stats

@torch.no_grad()
def forward_points(model, x,ext_info):
    assert x.shape[0]==1
    image=x[:,0:3]
    z = model.backbone(image)
    example_box=ext_info[:,:3,...]
    out_dict = model.decoder_layers(z,example_box)
    counting_map = out_dict["predict_counting_map"].half()
    offset_map = out_dict["offset_map"].half()
    attention_map=out_dict["atten_map"].half()
    pred_points = divide_map_to_points(counting_map, offset_map, device=x.device)

    return [pred_points], counting_map, offset_map,attention_map


def gen_global_grid(H, W, padding):
    grid_h, grid_w = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid = grid_h * W + grid_w
    grid = grid.reshape(1, H, W, 1).permute(0, 3, 1, 2).float()
    grid = F.pad(grid, (padding, padding, padding, padding), mode="constant", value=-1)
    return grid

def main(args,ckpt_path):
    utils.init_distributed_mode(args)
    model = model_without_ddp = build_model(args.Model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if is_main_process():
        print("=> loading checkpoint '{}'".format(os.path.join(args.Saver.save_dir, ckpt_path)))
        print(ckpt["states"])
        print("epoch: {}".format(ckpt["epoch"]))
    state_dict = module2model(ckpt['model'])

    model_dict = model.state_dict()
    load_param_dict = {k: v for k, v in state_dict.items() if k in model_dict and k.find("grid") == -1}
    model_dict.update(load_param_dict)
    
    model_without_ddp.load_state_dict(model_dict, strict=False)
    model.cuda().eval()
    
    dataset_val = build_dataset(image_set='test', args=args.Dataset.test)
 
    loader_val = DataLoader(dataset_val,
                            batch_size=args.Dataset.val.batch_size,
                            sampler=None,
                            shuffle=False,
                            num_workers=args.Dataset.val.num_workers,
                            pin_memory=True)


    stats = evaluate_counting_and_locating(model, loader_val,args)
    for key, value in stats.items():
        cprint(f'{key}:{value}', 'green')
    with open(os.path.join(args.save_path, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DenseMap Head ")
    parser.add_argument("--config", default="configs/FSC147/HRNET48.json")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--ckpt",default="outputs/HRNET48_202407192201/checkpoints/best.pth")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_path",default="outputs/fsc_7/")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
        cfg = edict(configs)
    print(cfg)
    cfg.save_path=args.save_path
    cfg.vis=args.vis
    main(cfg, args.ckpt)
