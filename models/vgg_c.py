import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from .transformer_cosine import TransformerEncoder, TransformerEncoderLayer
import os

from mmcv.cnn import get_model_complexity_info

from torch.amp import autocast
from torch.nn import functional as F
import numpy as np
from timm import create_model

__all__ = ['vgg19_trans']
model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

class VGG_Trans(nn.Module):
    def __init__(self, features):
        super(VGG_Trans, self).__init__()
        self.features = features

        d_model = 512
        nhead = 2
        num_layers = 4
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_layers, if_norm)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512+512+256+128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.reg_layer_1 = nn.Sequential(
            nn.Conv2d(512+512+256+128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            nn.Sigmoid()
        )
        get_model_complexity_info(self, (3, 2048, 2048))
    @autocast("cuda")
    def forward(self, x):
        b, c, h, w = x.shape
        rh = int(h) // 2
        rw = int(w) // 2
        x1,x2,x3,x4 = self.features(x)   # vgg network

        bs, c, h, w = x4.shape
        x4 = x4.flatten(2).permute(2, 0, 1)
        x4, features = self.encoder(x4, (h,w))   # transformer
        x4 = x4.permute(1, 2, 0).view(bs, c, h, w)
        #
        x4 = F.interpolate(x4, size=(rh, rw))
        x3 = F.interpolate(x3, size=(rh, rw))
        x2 = F.interpolate(x2, size=(rh, rw))
        x_fuse=torch.cat([x1,x2,x3,x4],dim=1)
        counting_map= self.reg_layer_0(x_fuse)   # counting head
        scale_map = self.reg_layer_1(x_fuse)   # offset head

        out_dict={
            "offset_map":scale_map,
            "predict_counting_map":counting_map,
        }
        return out_dict
    @torch.no_grad()
    def forward_points(self, x, threshold=0.8,loc_kernel_size=3):
        assert loc_kernel_size%2==1
        assert x.shape[0]==1
        out_dict = self.forward(x)
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

        pred_points=[[2*coord_w+1+0.5,2*coord_h+1+0.5] for coord_w,coord_h in coord]
        return pred_points
    def get_model_complexity(self, input_shape):
        flops, params = get_model_complexity_info(self, input_shape)
        return flops, params


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,"M"]
}

def build_counting_model():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    features=create_model("vgg19",pretrained=True,features_only=True, out_indices=[1,2,3,4])
    model = VGG_Trans(features)
    # model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']), strict=False)
    return model