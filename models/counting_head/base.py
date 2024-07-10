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


class Simple(nn.Module):
    def __init__(self, args):
        super(Simple, self).__init__()
        self.last_layer = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

        self.offset_layer = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),
            nn.Sigmoid()
        )
        
        self.fpn_fuse1=FPN_fuse(feature_channels=[64,48,96,192,384],fpn_out=64)
        self.fpn_fuse2=FPN_fuse(feature_channels=[64,48,96,192,384],fpn_out=64)
        
        self.radius = 8 # 不要再改半径了
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
        x0, x1, x2 , x3, x4  = x
        z1=self.fpn_fuse1([x0, x1, x2, x3,x4])
        out1 = self.last_layer(z1)
        z2=self.fpn_fuse2([x0, x1, x2, x3,x4])
        offset = (2 * self.offset_layer(z2) - 1.0) * self.radius

        out_dict = {}
        out_dict["predict_counting_map"] = out1
        out_dict["offset_map"] = offset
        return out_dict
    


def build_counting_head(args):

    return Simple(args)


if __name__ == "__main__":
    x = torch.as_tensor([
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0]
    ]).float().cuda().reshape(1, 1, 3, 3)
    off = torch.as_tensor([
        [[0, 0, -1],
        [0, 0, 0],
        [0, 0, 0]],
        [[0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]],
    ]).float().cuda().reshape(1, 2, 3, 3)

    m = Simple(None)
    y = m.dmap_render(x, off)
    print(y)