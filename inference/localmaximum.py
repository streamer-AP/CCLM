import torch
import torch.nn.functional as F
import numpy as np


def forward_points(model, x):
    assert x.shape[0]==1
    z = model.backbone(x)
    out_dict = model.decoder_layers(z)
    counting_map = out_dict["predict_counting_map"]
    pred_points = map_to_points(counting_map, device=x.device)
    offset_map = out_dict["offset_map"]
    return [pred_points], counting_map, counting_map

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

def map_to_points(predict_counting_map, loc_kernel_size=13, device="cuda"):
    loc_padding=loc_kernel_size//2
    kernel=torch.ones(1,1,loc_kernel_size,loc_kernel_size).to(device).float()

    threshold=0.3
    low_resolution_map=F.interpolate(F.relu(predict_counting_map),scale_factor=1)
    H,W=low_resolution_map.shape[-2],low_resolution_map.shape[-1]

    unfolded_map=F.unfold(low_resolution_map,kernel_size=loc_kernel_size,padding=loc_padding)
    unfolded_max_idx=unfolded_map.max(dim=1,keepdim=True)[1]
    unfolded_max_mask=(unfolded_max_idx==loc_kernel_size**2//2).reshape(1,1,H,W)

    predict_cnt=F.conv2d(low_resolution_map,kernel,padding=loc_padding)
    predict_filter=(predict_cnt>threshold).float()
    predict_filter=predict_filter*unfolded_max_mask
    predict_filter=predict_filter.detach().cpu().numpy().astype(bool).reshape(H,W)

    pred_coord_weight=F.normalize(unfolded_map,p=1,dim=1)
    
    coord_h=torch.arange(H).reshape(-1,1).repeat(1,W).to(device).float()
    coord_w=torch.arange(W).reshape(1,-1).repeat(H,1).to(device).float()
    coord_h=coord_h.unsqueeze(0).unsqueeze(0)
    coord_w=coord_w.unsqueeze(0).unsqueeze(0)
    unfolded_coord_h=F.unfold(coord_h,kernel_size=loc_kernel_size,padding=loc_padding)
    pred_coord_h=(unfolded_coord_h*pred_coord_weight).sum(dim=1,keepdim=True).reshape(H,W).detach().cpu().numpy()
    unfolded_coord_w=F.unfold(coord_w,kernel_size=loc_kernel_size,padding=loc_padding)
    pred_coord_w=(unfolded_coord_w*pred_coord_weight).sum(dim=1,keepdim=True).reshape(H,W).detach().cpu().numpy()
    coord_h=pred_coord_h[predict_filter].reshape(-1,1)
    coord_w=pred_coord_w[predict_filter].reshape(-1,1)
    coord=np.concatenate([coord_w,coord_h],axis=1)

    pred_points=[[2*coord_w+0.5,2*coord_h+0.5] for coord_w,coord_h in coord]
    return pred_points

