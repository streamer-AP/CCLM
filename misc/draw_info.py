import imp
import os

import cv2
import einops
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("agg")

class Drawer_DenseMap():
    def __init__(self, args) -> None:
        self.draw_freq = args.draw_freq
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.draw_original = args.draw_original
        self.draw_denseMap = args.draw_denseMap
        self.draw_output = args.draw_output
        self.mean = args.mean
        self.std = args.std
        self.cnt = 0

    def click(self):
        self.cnt += 1
        return self.cnt % self.draw_freq == 0
    def clear(self):
        self.cnt = 0

    def __call__(self,epoch=-1, inputs=None, dMaps=None, out_dict=None, header="train"):
        if self.click():
            sub_dir = os.path.join(self.output_dir, f"epoch{epoch}")
            os.makedirs(sub_dir, exist_ok=True)
            if self.draw_original and inputs is not None:
                for j in range(inputs.shape[0]):
                    origin_input = einops.rearrange(inputs[j], "c h w->h w c")
                    origin_input = origin_input*torch.tensor([[self.std]]).to(origin_input.device)+torch.tensor([[self.mean]]).to(origin_input.device)
                    origin_input = (origin_input.detach().cpu().numpy()*255).astype(np.uint8).path.join(sub_dir, f"{header}_input_{self.cnt}_{j}.jpg"))
                    origin_input = cv2.cvtColor(origin_input, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(sub_dir, f"{header}_input_{self.cnt}_{j}.jpg"), origin_input)

            if self.draw_output and out_dict is not None:
                pred_outputs = out_dict["predict_counting_map"]
                for j in range(pred_outputs.shape[0]):
                    output = pred_outputs[j].detach().cpu().numpy()
                    output=output/(0.0000001+np.max(output))
                    output=output.transpose([1,2,0])
                    plt.figure(figsize=(10, 10))
                    plt.imshow(output)
                    plt.savefig(os.path.join(
                        sub_dir, f"{header}_output_{self.cnt}_{j}.jpg"))
                    plt.close()

            if self.draw_denseMap and dMaps is not None:
                dMaps = dMaps.detach().cpu().numpy()
                for j in range(dMaps.shape[0]):
                    dMap = dMaps[j]/(0.0000001+np.max(dMaps[j]))
                    dMap = dMap.transpose([1, 2, 0])
                    plt.figure(figsize=(10, 10))
                    plt.imshow(dMap)
                    plt.savefig(os.path.join(
                        sub_dir, f"{header}_dmap_{self.cnt}_{j}.jpg"))
                    plt.close()

            if "regress_result" in out_dict.keys():
                for j in range(len(out_dict["regress_result"])):
                    mask = out_dict["regress_result"][j]
                    if mask is not None:
                        mask = mask.detach().cpu()
                        # mask = mask.amax(dim=1, keepdim=True)
                        mask = einops.rearrange(mask, "b c h w -> (b h) (c w)")
                        mask = mask.numpy()
                        mask = mask /(0.0000001+np.max(mask))
                        plt.figure(figsize=(10, 10))
                        plt.imshow(mask)
                        plt.savefig(os.path.join(sub_dir, f"{header}_regress_{self.cnt}_{j}.jpg"))
                        plt.close()
                
            if "classify_result" in out_dict.keys():
                for j in range(len(out_dict["classify_result"])):
                    mask = out_dict["classify_result"][j]
                    if mask is not None:
                        mask = mask.detach().cpu()
                        mask = einops.rearrange(mask, "b c h w -> (b h) (c w)")
                        mask = mask.numpy()
                        mask = mask /(0.0000001+np.max(mask))
                        plt.figure(figsize=(10, 10))
                        plt.imshow(mask)
                        plt.savefig(os.path.join(sub_dir, f"{header}_classify_{self.cnt}_{j}.jpg"))
                        plt.close()

            
