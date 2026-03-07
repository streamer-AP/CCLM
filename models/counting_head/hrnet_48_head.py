import torch
import torch.nn as nn
import torch.nn.functional as F

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y
class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.GELU()
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

# class PSPModule(nn.Module):
#     # In the original inmplementation they use precise RoI pooling 
#     # Instead of using adaptative average pooling
#     def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
#         super(PSPModule, self).__init__()
#         out_channels = in_channels // len(bin_sizes)
#         self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
#                                                         for b_s in bin_sizes])
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
#                                     kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1)
#         )

#     def _make_stages(self, in_channels, out_channels, bin_sz):
#         prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
#         conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         bn = nn.BatchNorm2d(out_channels)
#         relu = nn.ReLU(inplace=True)
#         return nn.Sequential(prior, conv, bn, relu)
    
#     def forward(self, features):
#         h, w = features.size()[2], features.size()[3]
#         pyramids = [features]
#         pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
#                                         align_corners=True) for stage in self.stages])
#         output = self.bottleneck(torch.cat(pyramids, dim=1))
#         return output

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.layer0_conv = nn.Sequential(
            nn.Conv2d(64, 64, 7,padding=3),
            nn.GELU(),
            nn.Conv2d(64, 48, 7,padding=3),
            nn.GELU(),
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(48, 48, 7,padding=3),
            nn.GELU(),
            nn.Conv2d(48, 1, 1),
            nn.GELU(),

        )
        self.fpn_fuse=FPN_fuse(feature_channels=[48,48,96,192,384],fpn_out=48)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape)
        # torch.Size([1, 96, 256, 256])
        # torch.Size([1, 192, 128, 128])
        # torch.Size([1, 384, 64, 64])
        # torch.Size([1, 768, 32, 32])
        x0,x1, x2 , x3,x4  = x
        # x4 = F.interpolate(x4, scale_factor=16)
        # x3 = F.interpolate(x3, scale_factor=8)
        # x2 = F.interpolate(x2, scale_factor=4)
        # x1 = F.interpolate(x1, scale_factor=2)
        # scale_rate=0.0625
        # x0 = F.interpolate(x0, scale_factor=scale_rate)
        # x1 = F.interpolate(x1, scale_factor=scale_rate*2)
        # x2 = F.interpolate(x2, scale_factor=scale_rate*4)
        # x3 = F.interpolate(x3, scale_factor=scale_rate*8)
        # x4 = F.interpolate(x4, scale_factor=scale_rate)
        x0 = self.layer0_conv(x0)
        z=self.fpn_fuse([x0,x1,x2,x3,x4])
        out1 = self.last_layer(z)
        # out1 = F.interpolate(out1, scale_factor=1/scale_rate)
        out_dict = {}
        out_dict["predict_counting_map"] = out1
        return out_dict


def build_counting_head(args):

    return Simple()
