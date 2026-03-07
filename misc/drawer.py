import imp
import os

import cv2
import einops
import numpy as np
import torch
from matplotlib import pyplot as plt


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

    def __call__(self,epoch=-1, inputs=None, dMaps=None, outputs=None,header="train"):
        if self.click():
            sub_dir = os.path.join(self.output_dir, f"epoch{epoch}")
            os.makedirs(sub_dir,exist_ok=True)
            if self.draw_original and inputs is not None:
                for j in range(inputs.shape[0]):
                    origin_input = einops.rearrange(inputs[j], "c h w->h w c")
                    origin_input = origin_input*torch.tensor([[self.std]]).to(
                        origin_input.device)+torch.tensor([[self.mean]]).to(origin_input.device)
                    origin_input = (origin_input.detach(
                    ).cpu().numpy()*255).astype(np.uint8)
                    # plt.imshow(origin_input)
                    # plt.savefig(os.path.join(sub_dir, f"{j}.png"))
                    # plt.close()
                    origin_input = cv2.cvtColor(
                        origin_input, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(
                        sub_dir, f"{header}_input_{self.cnt}_{j}.jpg"), origin_input)

            if self.draw_denseMap and dMaps is not None:
                for j in range(dMaps.shape[0]):
                    dMap = dMaps[j]/(0.0000001+np.max(dMaps[j]))
                    dMap = dMap.transpose([1, 2, 0])
                    plt.figure(figsize=(10, 10))
                    plt.imshow(dMap)
                    plt.savefig(os.path.join(
                        sub_dir, f"{header}_dmap_{self.cnt}_{j}.jpg"))
                    plt.close()
            if self.draw_output and outputs is not None:
                for j in range(outputs.shape[0]):
                    output = outputs[j].detach().cpu().numpy()
                    output=output/(0.0000001+np.max(output))
                    output=output.transpose([1,2,0])
                    plt.figure(figsize=(10, 10))
                    plt.imshow(output)
                    plt.savefig(os.path.join(
                        sub_dir, f"{header}_output_{self.cnt}_{j}.jpg"))
                    plt.close()
