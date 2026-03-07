import argparse
import json
import math
import os
from pprint import pprint

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from inference.nms_debug_sliding_img import forward_points
from inference.nms_sliding import divide_map_to_points

from inference.padding_image import pad_image, points_affine
from datasets.torchvision_datasets.coco import CocoDetection
from misc import utils
from misc.utils import get_local_rank, get_local_size, is_main_process
from models import build_model
from models.utils import module2model
from tqdm import tqdm
import cv2
from nms import nms
from scipy.spatial import KDTree
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def forward_points(model, x):
    assert x.shape[0]==1
    z = model.backbone(x)
    out_dict = model.decoder_layers(z)
    counting_map = out_dict["predict_counting_map"].half()
    offset_map = out_dict["offset_map"].half()

    # sliding window inference
    pred_points = divide_map_to_points(counting_map, offset_map, device=x.device)

    return [pred_points], counting_map, offset_map


def color_trans(omap):
    d = np.power(omap, 2).sum(axis=-1)
    d = np.sqrt(d)
    # dmax = 8 * math.sqrt(2)
    # d = 255 *  (d[:, :, np.newaxis]) / dmax
    d = 255 * 2
    u = omap[:, :, :1] / 8
    v = omap[:, :, 1:] / 8
    nr = 6 + u + v
    # print(u.shape, v.shape, d.shape)

    B = (2 + u) / nr
    G = 2 / nr
    R = (2 + v) / nr

    out = np.concatenate((B, G, R), axis = -1)
    out = out * d
    # print(np.max(out))
    return out

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

def inference_transform():
    return A.Compose([
        A.Normalize(),
    ])

def draw_dmap(dmap):
    dmap_np = dmap[0].cpu().numpy()
    dmap = dmap_np / dmap_np.max()
    dmap = dmap * 255
    dmap = dmap.astype(np.uint8)
    dmap = cv2.applyColorMap(dmap, cv2.COLORMAP_JET)
    return dmap, dmap_np
def draw_points(img,points, c=(0,0,255)):
    h,w=img.shape[:2]
    r=5
    for point in points:
        cv2.circle(img,(int(point[0]),int(point[1])), r, c, 1)
    return img

class JHUCounting_test(CocoDetection):
    def __init__(self, root, annFile, transforms=None, max_len=5000, cache_mode=False, local_rank=0, local_size=1):
        super().__init__(root, annFile, transform=None, target_transform=None,
                         transforms=None, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.alb_transforms = transforms
        self.to_tensor = ToTensorV2()
        self.max_len = max_len

    def __getitem__(self, index):

        image, target = super().__getitem__(index)
        img_id = self.ids[index]
        w, h = image.size
        image = np.array(image)
        data = self.alb_transforms(image=image)
        image = data["image"]
        max_edge=max(w, h)
        size_large = 4096
        size_small = 2560
        if max_edge > size_large:
            scale = size_large/max_edge
            image = cv2.resize(image,(0,0),fx=scale,fy=scale)
        if max_edge < size_small:
            scale = size_small/max_edge
            image = cv2.resize(image,(0,0),fx=scale,fy=scale)
        labels = {}

        labels["wh"] = torch.as_tensor([w, h], dtype=torch.long)
        labels["id"] = torch.as_tensor(int(img_id), dtype=torch.long)
        image = self.to_tensor(image=image)["image"]
        image_pad, labels = pad_image(image, labels, padsize=64)
        return image_pad, labels



@torch.no_grad()
def inference(model, data_loader, dataset, args):
    id_filename={}
    img_prefix=args.Dataset.val.img_prefix if args.mode=="val" else args.Dataset.test.img_prefix
    ann_path = args.Dataset.val.ann_file if args.mode=="val" else args.Dataset.test.ann_file
    with open(ann_path,"r") as f:
        info=json.load(f)
        id_filename={v["id"]: v["file_name"] for v in info["images"]}

    model.eval()
    result = {}
    result_txt = open(os.path.join(args.output_dir, "result.txt"), "w")
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(args.gpu)
            # print("img shape", inputs.shape)
            assert inputs.shape[0] == 1
            if args.distributed:
                pred_pts,pred_maps, offset_map = forward_points(model, inputs)
            else:
                pred_pts,pred_maps, offset_map = forward_points(model, inputs)

            save_dict = {}
            # save_path = 
            cur_id = str(labels["id"][0].item())
            save_dict["pred_pts"] = pred_pts
            # save_dict["scale"] = [x_scale, y_scale]
            save_dict["pred_map"] = pred_maps.cpu()
            save_dict["offset_map"] = offset_map.cpu()
            torch.save(save_dict, os.path.join(args.draw_dir,cur_id+"_dict.pth"))

            i = 0
            result_pts, result_str = points_affine(pred_pts, labels)
            result[labels["id"][0].item()] = result_str

            line_str = []
            line_str.append(str(labels["id"][i].item()))
            line_str.append(str(len(pred_pts[i])))
            line_str += result_pts
            line_str = " ".join(line_str)
            result_txt.write(line_str + "\n")
            if args.draw:
                pred_map=pred_maps[0]
                img_path = os.path.join(img_prefix, id_filename[labels["id"][i].item()])
                img = cv2.imread(img_path)
                img = draw_points(img,result[labels["id"][i].item()])
                cv2.imwrite(os.path.join(args.draw_dir, str(labels["id"][i].item())+".jpg"),img)
                dmap, dmap_np=draw_dmap(pred_map)
                cv2.imwrite(os.path.join(args.draw_dir, str(labels["id"][i].item())+"_dmap.jpg"),dmap)

                offset = offset_map[0].permute(1, 2, 0).detach().cpu().numpy()
                colored_map = color_trans(offset)
                colored_map = colored_map.astype(np.uint8)
                cv2.imwrite(os.path.join(args.draw_dir, str(labels["id"][i].item())+"_omap.jpg"), colored_map)

                # rmap = dmap_render(pred_maps.float(), offset_map.float())
                # rmap, _ = draw_dmap(rmap[0])
                # cv2.imwrite(os.path.join(args.draw_dir, str(labels["id"][i].item())+"_rmap.jpg"), rmap)

    result_txt.close()
    return result


def main(args, ckpt_path):
    utils.init_distributed_mode(args)
    utils.set_randomseed(42 + utils.get_rank())

    # initilize the model
    model = model_without_ddp = build_model(args.Model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if is_main_process():
        pprint("=> loading checkpoint '{}'".format(
            os.path.join(args.Saver.save_dir, ckpt_path)))
        pprint(ckpt["states"])
        pprint("epoch: {}".format(ckpt["epoch"]))
    state_dict = module2model(ckpt['model'])
    model_dict = model.state_dict()
    load_param_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(load_param_dict)
    model_without_ddp.load_state_dict(model_dict)
    model.cuda().eval()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # build the dataset and dataloader
    if args.mode == "val":
        prefix=args.Dataset.val.img_prefix
        ann_file=args.Dataset.val.ann_file
        max_len=args.Dataset.val.max_len
    elif args.mode == "test":
        prefix=args.Dataset.test.img_prefix
        ann_file=args.Dataset.test.ann_file
        max_len=args.Dataset.test.max_len
    dataset_test = JHUCounting_test(prefix,
                                    ann_file,
                                    max_len=max_len,
                                    transforms=inference_transform(),
                                    cache_mode=args.Dataset.test.cache_mode,
                                    local_rank=get_local_rank(),
                                    local_size=get_local_size())
    print(prefix, ann_file)

    sampler_test = DistributedSampler(
        dataset_test, shuffle=False) if args.distributed else None
    loader_val = DataLoader(dataset_test,
                            batch_size=args.Dataset.test.batch_size,
                            sampler=sampler_test,
                            shuffle=False,
                            num_workers=args.Dataset.test.num_workers,
                            pin_memory=True)
    results = inference(model, loader_val, dataset_test, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DenseMap Head ")
    parser.add_argument("--config", default="outputs/nwpu/fidt_ucl.json")
    parser.add_argument("--mode", default="val")
    parser.add_argument("--draw_dir", default="outputs/nwpu/draw_val")
    parser.add_argument("--output_dir",default="outputs/nwpu/val_nwpu")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--ckpt",
        default="outputs/nwpu/checkpoints/best.pth")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
        cfg = edict(configs)
    print(cfg)

    cfg.draw_dir=args.draw_dir
    cfg.mode=args.mode
    # cfg.draw=args.vis
    cfg.draw = True
    cfg.output_dir=args.output_dir
    if is_main_process():
        os.makedirs(cfg.output_dir, exist_ok=True)
        if cfg.draw:
            os.makedirs(args.draw_dir, exist_ok=True)
    main(cfg, args.ckpt)
