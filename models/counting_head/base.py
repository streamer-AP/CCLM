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

def roi_pooling(input, boxes, output_size):
    """
    Args:
        input (torch.Tensor): Input feature map of shape (N, C, H, W).
        boxes (torch.Tensor): Tensor of shape (N, M, 4) containing the ROI boxes.
                              Each box is represented as [x_min, y_min, height, width].
        output_size (tuple): Size (height, width) of the output after ROI pooling.
        
    Returns:
        torch.Tensor: ROI pooled output of shape (N, M, C, output_size[0], output_size[1]).
    """
    N, C, H, W = input.shape
    M = boxes.shape[1]
    output = torch.zeros((N, M, C, output_size[0], output_size[1]), device=input.device)
    
    for i in range(N):
        for j in range(M):
            x_min, y_min, height, width = boxes[i, j].long()
            
            x_max = x_min + width+1
            y_max = y_min + height+1
            region = input[i, :, y_min:y_max, x_min:x_max]
            pooled_region = F.adaptive_max_pool2d(region, output_size)
            output[i, j] = pooled_region

    return output
    
    
    
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
        self.example_fuse=FPN_fuse(feature_channels=[64,48,96,192,384],fpn_out=64)
        self.radius = 8 
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

    def forward(self, x, x_example):
        x0, x1, x2 , x3, x4  = x
        box=x_example/2
        
        z1=self.fpn_fuse1([x0, x1, x2, x3,x4])
        z_e=roi_pooling(z1, box, (7, 7))#(N, M, C, 7, 7)
        # Reshape z_e and z1 for self-attention calculation
        N, M, C= z_e.shape[:3]
        H,W=z1.shape[2:]
        z_e_reshaped = z_e.view(N, M, C, -1)  # (N, M, C, 49)
        z1_reshaped = z1.view(N, C, -1)  # (N, C, H*W)

        # Compute attention scores for each sample independently
        attention_scores = torch.einsum('nmcl,nch->nmlh', z_e_reshaped, z1_reshaped)  # (N, M, 49, H*W)
        
        # Aggregate attention scores for each image
        attention_scores = attention_scores.sum(dim=1)  # (N, 49, H*W)

        # Compute attention map
        # attention_map = F.softmax(attention_scores, dim=-1)  # (N, 49, H*W)
        attention_map = attention_scores.mean(dim=1)  # (N, H*W)
        attention_map = attention_map.view(N, H, W)  # (N, H, W)
        attention_map = F.sigmoid(attention_map)
        out1 = self.last_layer(z1*attention_map.unsqueeze(1))
        z2=self.fpn_fuse2([x0, x1, x2, x3,x4])
        offset = (2 * self.offset_layer(z2) - 1.0) * self.radius

        out_dict = {}
        out_dict["predict_counting_map"] = out1
        out_dict["offset_map"] = offset
        out_dict["atten_map"] = attention_map
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