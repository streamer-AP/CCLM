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
            nn.ReLU()
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
            x_min, y_min, width, height = boxes[i, j].long()
            x_max = x_min + width+1
            y_max = y_min + height+1
            region = input[i, :, y_min:y_max, x_min:x_max]
            
            max_size = max(height, width)
            scale = max_size / max(output_size)
            new_height = int(height / scale)
            new_width = int(width / scale)
            new_height = max(new_height, 1)
            new_width = max(new_width, 1)
            region = F.interpolate(region.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=True)
            left_pad = (output_size[1] - new_width) // 2
            right_pad = output_size[1] - new_width - left_pad
            top_pad = (output_size[0] - new_height) // 2
            bottom_pad = output_size[0] - new_height - top_pad
            pooled_region = F.pad(region, (left_pad, right_pad, top_pad, bottom_pad)).squeeze(0)
                
            output[i, j] = pooled_region/(0.000001+pooled_region.sum())

    return output
    
    
    
class Simple(nn.Module):
    def __init__(self, args):
        super(Simple, self).__init__()
        self.channels=args.channels
        self.first_channel=self.channels[0]
        self.last_layer = nn.Sequential(
            nn.Conv2d(self.first_channel+60, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

        self.offset_layer = nn.Sequential(
            nn.Conv2d(self.first_channel, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),
            nn.Sigmoid()
        )
        # self.attention_layer = nn.Sequential(
        #     nn.Conv2d(36, 36, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(36, 36, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(36, 1, 1),
        # )
        
        
        self.fpn_fuse1=FPN_fuse(feature_channels=self.channels,fpn_out=self.first_channel)
        self.radius = 8 
        self.roi_size=args.roi_size
        self.map_scale=args.map_scale
        self.upsample_ratio=args.upsample_ratio
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
        boxes=x_example/self.map_scale
        z1=self.fpn_fuse1(x)
        with torch.no_grad():
            c_out=5
            c_in=self.first_channel
            B=z1.shape[0]
            H,W=z1.shape[2],z1.shape[3]
            conv_map=torch.zeros((B,c_out*3*4,H,W),dtype=torch.float32,device=z1.device)
            weight_H,weight_W=self.roi_size[0]*2**2+1,self.roi_size[0]*2**2+1
            weights=torch.zeros((B,c_out*3*4,c_in,weight_H,weight_W),dtype=torch.float32,device=z1.device)
            for i in range(3):
                roi_size=(self.roi_size[0]*2**i+1,self.roi_size[1]*2**i+1)
                z_e=roi_pooling(z1, boxes, roi_size)
            
                z_e=z_e.view(-1,c_out,c_in,roi_size[0],roi_size[1])
                z_e_rotate90_aug=torch.zeros((B,4*c_out,c_in,roi_size[0],roi_size[1]),dtype=torch.float32,device=z1.device)
                z_e_rotate90_aug[:,0:c_out]=z_e
                z_e_rotate90_aug[:,c_out:c_out*2]=z_e.flip(-1)
                z_e_rotate90_aug[:,c_out*2:c_out*3]=z_e.flip(-2)
                z_e_rotate90_aug[:,c_out*3:c_out*4]=z_e.flip(-1).flip(-2)
                padding_H=(weight_H-roi_size[0])//2
                padding_W=(weight_W-roi_size[1])//2
                weights[:,i*c_out*4:(i+1)*c_out*4,:,padding_H:padding_H+roi_size[0],padding_W:padding_W+roi_size[1]]=z_e_rotate90_aug
        for b in range(B):
            for i in range(3):
                weight=weights[b,i*c_out*4:(i+1)*c_out*4]
                conv_map[b,i:i+c_out*4,...]=F.conv2d(z1[b].unsqueeze(0),weight,padding=weight_H//2)
        attention_map=conv_map.mean(dim=1,keepdim=True)
        out1 = self.last_layer(torch.cat((conv_map,z1*attention_map),dim=1))

        offset = (self.map_scale * self.offset_layer(z1) - self.map_scale/2) * self.radius

        # with torch.no_grad():
        #     scales=torch.zeros((N,1))
        #     scales=scales.to(x0.device)
        #     for batch_idx,batch_boxes in enumerate(boxes):
        #         for box in batch_boxes:
        #             x_min, y_min, height, width = box.long()
        #             scales[batch_idx,0]+=out1[0,0,y_min:y_min+height,x_min:x_min+width].sum()
        #         scales[batch_idx,0]=scales[batch_idx,0]/batch_boxes.shape[0]
        #     scales=scales.unsqueeze(2).unsqueeze(3)
        #     scales=torch.clamp(scales,1,2)
        #     out1=out1/scales
                
                
        offset=F.interpolate(offset, scale_factor=self.upsample_ratio, mode='bilinear', align_corners=True)
        out1=F.interpolate(out1, scale_factor=self.upsample_ratio, mode='bilinear', align_corners=True)
        out_dict = {}
        out_dict["predict_counting_map"] = out1
        out_dict["offset_map"] = offset
        out_dict["atten_map"] = attention_map
        out_dict["roi_feature"] = z_e
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