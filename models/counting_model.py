import os

import torch
from mmcv.cnn import get_model_complexity_info
from torch import nn

from .backbone import build_backbone
from .counting_head import build_counting_head
from .utils import module2model
from torch.amp import autocast
from torch.nn import functional as F
import numpy as np
from monai.inferers import sliding_window_inference
class SingleScaleEncoderDecoderCounting(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.backbone = build_backbone(args.backbone)
        self.decoder_layers = build_counting_head(args.counting_head)
        if "stride" in args:
            self.stride=args.stride
        else:
            self.stride=2
        # self.get_model_complexity(input_shape=(3, 1536, 1536))
    @autocast("cuda")
    def forward(self, x, ext_info):
        image=x[:,0:3]
        example_box=ext_info[:,:5,...]# B*3*4 for x_min, y_min, width, height
        
        z = self.backbone(image)
        
        out_dict = self.decoder_layers(z,example_box)
        return out_dict
    @torch.no_grad()
    def sliding_window_infer(self,x,threshold=0.8,loc_kernel_size=3,roi_size=1024,mode="gaussian",sigma_scale=0.125):
        assert loc_kernel_size%2==1
        assert x.shape[0]==1
        out_dict=sliding_window_inference(x,sw_batch_size=4,roi_size=roi_size,predictor=self.forward,overlap=0.5,mode=mode,sigma_scale=0.125)
        predict_counting_map=out_dict["predict_counting_map"].detach().float()
        pred_points=self._map_to_points(predict_counting_map,threshold=threshold,loc_kernel_size=loc_kernel_size,device=x.device)

        return [pred_points],predict_counting_map
    @torch.no_grad()
    def forward_points(self, x, ext_info, threshold=0.8,loc_kernel_size=3):
        assert loc_kernel_size%2==1
        assert x.shape[0]==1
        image=x[:,0:3]
        example_box=ext_info[:,:5,...]# B*3*4 for x_min, y_min, width, height
        z = self.backbone(image)
        out_dict = self.decoder_layers(z,example_box)
        predict_counting_map=out_dict["predict_counting_map"].detach().float()
        pred_points=self._map_to_points(predict_counting_map,threshold=threshold,loc_kernel_size=loc_kernel_size,device=x.device)

        return [pred_points],out_dict 
    @torch.no_grad()
    def _map_to_points(self, predict_counting_map, threshold=0.8,loc_kernel_size=3,device="cuda"):
        loc_padding=loc_kernel_size//2
        kernel=torch.ones(1,1,loc_kernel_size,loc_kernel_size).to(device).float()

        # threshold=0.5
        threshold=0.5
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

        pred_points=[[self.stride*coord_w+0.5,self.stride*coord_h+0.5] for coord_w,coord_h in coord]
        return pred_points
    def get_model_complexity(self, input_shape):
        flops, params = get_model_complexity_info(self, input_shape)
        return flops, params

def build_counting_model(args):
    if args.type == "single_scale_encoder_decoder":
        model = SingleScaleEncoderDecoderCounting(args)

    if os.path.exists(args.ckpt):
        print("load param from", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = module2model(ckpt['model'])
        model.load_state_dict(state_dict, strict=False)

    return model
