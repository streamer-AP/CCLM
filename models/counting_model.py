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
class SingleScaleCounting(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.backbone = build_backbone(args.backbone)

        if args.counting_head.in_channels == -1:
            args.counting_head.in_channels = self.backbone.feature_info.channels(
            )[-1]
        self.decoder_layers = build_counting_head(args.counting_head)

        in_chans = 3
        try:
            in_chans = args.backbone.others.in_chans
        except:
            pass
        self.get_model_complexity(input_shape=(in_chans, 1024, 1024))

    def forward(self, x, ext_info=None):

        z = self.backbone(x)
        # print("ext_info_nf", ext_info)
        if ext_info is not None:
            out = self.decoder_layers(z[-1], ext_info=ext_info)
        else:
            out = self.decoder_layers(z[-1])

        out_dict = out
        out_dict["features"] = z[-1]
        return out_dict
    @torch.no_grad()
    def forward_points(self, x, threshold=0.25):
        outputs_dict=self.forward(x)
        predict_counting_maps = outputs_dict["predict_counting_map"].float()
        outputs=[]
        for predict_counting_map in predict_counting_maps:
            predict_counting_map = predict_counting_map.unsqueeze(0)
            pred_dmap_low = F.relu(predict_counting_map)
            pred_dmap_low = F.avg_pool2d(pred_dmap_low, 2) * 4
            H, W = pred_dmap_low.shape[-2], pred_dmap_low.shape[-1]
            pred_map_unflod = F.unfold(pred_dmap_low, kernel_size=3, padding=1)
            pred_max = pred_map_unflod.max(dim=1, keepdim=True)[1]
            pred_max = (pred_max == 3**2 // 2).reshape(1, 1, H, W)
            kernel3x3 = torch.ones(1, 1, 3, 3).float().cuda()
            pred_cnt = F.conv2d(pred_dmap_low, weight=kernel3x3, padding=1)
            pred_pts = []
            coord_h = torch.arange(H).reshape(1, 1, H, 1).repeat(1, 1, 1,
                                                                W).float().cuda()
            coord_w = torch.arange(W).reshape(1, 1, 1, W).repeat(1, 1, H,
                                                                1).float().cuda()
            coord_h_unflod = F.unfold(coord_h, kernel_size=3, padding=1)

            coord_w_unflod = F.unfold(coord_w, kernel_size=3, padding=1)
            for i in range(4):
                pred_filter = (pred_cnt > i+threshold)*(pred_cnt < i+1+threshold)
                pred_filter = pred_filter * pred_max
                pred_filter = pred_filter.detach().cpu().squeeze(0).squeeze(0)
                pred_coord_weight = F.normalize(pred_map_unflod, p=1, dim=1)

                coord_h_pred = (coord_h_unflod * pred_coord_weight).sum(
                    dim=1, keepdim=True).reshape(H, W).detach().cpu()
                coord_w_pred = (coord_w_unflod * pred_coord_weight).sum(
                    dim=1, keepdim=True).reshape(H, W).detach().cpu()
                coord_h = coord_h_pred[pred_filter].unsqueeze(1)
                coord_w = coord_w_pred[pred_filter].unsqueeze(1)
                coord = torch.cat((coord_h, coord_w), dim=1).numpy()
                for pt in coord:
                    y_coord, x_coord = pt
                    y_coord, x_coord = float(4 * y_coord + 1.5+i), float(4 * x_coord + 1.5+i)
                    pred_pts.append([x_coord, y_coord])
                if i>0:
                    print("i",i,"pred_pts",len(pred_pts))
            outputs.append(pred_pts)
        return outputs

    def get_model_complexity(self, input_shape):
        flops, params = get_model_complexity_info(self, input_shape)
        return flops, params


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
        example_box=ext_info[:,:3,...]# B*3*4 for x_min, y_min, width, height
        
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
        example_box=ext_info[:,:3,...]# B*3*4 for x_min, y_min, width, height
        z = self.backbone(image)
        out_dict = self.decoder_layers(z,example_box)
        predict_counting_map=out_dict["predict_counting_map"].detach().float()
        pred_points=self._map_to_points(predict_counting_map,threshold=threshold,loc_kernel_size=loc_kernel_size,device=x.device)

        return [pred_points],predict_counting_map
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

class DoubleEncoderDecoderCounting(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.backbone = build_backbone(args.backbone)
        self.decoder_layers = build_counting_head(args.counting_head1)
        self.decoder_layers2 = build_counting_head(args.counting_head2)

        self.get_model_complexity(input_shape=(3, 1536, 1536))
    @autocast("cuda")
    def forward(self, x, ext_info=None):

        z = self.backbone(x)
        out_dict = self.decoder_layers(z)
        out_dict2 = self.decoder_layers2(z)
        out_dict["scale_map"]=out_dict2["predict_counting_map"]*18*18
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
    def forward_points(self, x, threshold=0.8,loc_kernel_size=3):
        assert loc_kernel_size%2==1
        assert x.shape[0]==1
        z = self.backbone(x)
        out_dict = self.decoder_layers(z)
        predict_counting_map=out_dict["predict_counting_map"].detach().float()
        pred_points=self._map_to_points(predict_counting_map,threshold=threshold,loc_kernel_size=loc_kernel_size,device=x.device)

        return [pred_points],predict_counting_map
    @torch.no_grad()
    def _map_to_points(self, predict_counting_map, threshold=0.8,loc_kernel_size=3,device="cuda"):
        loc_padding=loc_kernel_size//2
        kernel=torch.ones(1,1,loc_kernel_size,loc_kernel_size).to(device).float()

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

        pred_points=[[2*coord_w,2*coord_h] for coord_w,coord_h in coord]
        return pred_points
    def get_model_complexity(self, input_shape):
        flops, params = get_model_complexity_info(self, input_shape)
        return flops, params


def build_counting_model(args):
    if args.type == "single_scale_encoder_decoder":
        model = SingleScaleEncoderDecoderCounting(args)
    elif args.type == "double_encoder_decoder":
        model = DoubleEncoderDecoderCounting(args)

    if os.path.exists(args.ckpt):
        print("load param from", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = module2model(ckpt['model'])
        model.load_state_dict(state_dict, strict=False)

    return model
