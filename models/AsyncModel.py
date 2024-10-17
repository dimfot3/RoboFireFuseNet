import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os
from HeadPidnet import PIDNetHead
from SwinTransformer import SwinTransformer
from CrossSwinTransformer import CrossSwinTransformer
import math
import numpy as np


class AsyncModel(nn.Module):
    def __init__(self, in_planes):
        super(AsyncModel, self).__init__()
        self.in_planes = in_planes
        self.input_res = 256
        self.num_classes = 3
        self.tf_config = [2, 2]
        self.rgb_tf = SwinTransformer(config=self.tf_config, dim=in_planes//2, drop_path_rate=0.2, input_resolution=self.input_res, input_c=3)
        self.ir_tf = SwinTransformer(config=self.tf_config, dim=in_planes//2, drop_path_rate=0.2, input_resolution=self.input_res, input_c=1)
        self.cross_tf = CrossSwinTransformer(config=self.tf_config, dim=in_planes//2, drop_path_rate=0.2, input_resolution=self.input_res)
        self.pidnethead = PIDNetHead(m=2, n=3, num_classes=self.num_classes, planes=in_planes//2, ppm_planes=96, head_planes=128, augment=True, channels=3)
    
    def find_mode(self, img):
        """
        returns 0 if rgb and 1 if ir
        """
        if(math.isclose(img[1:].sum(), 0, abs_tol=1e-9)):
            return 1
        return 0

    def split_input(self, img_arr):
        B, N, H, W = img_arr.size(0),  img_arr.size(1) // 3, img_arr.size(2), img_arr.size(3)
        img_arr = img_arr.view(B * N, 3, H, W)
        batch_idxs = np.array([[i] * N for i in range(B)]).reshape(-1)
        modes = np.array([self.find_mode(img) for img in img_arr])
        return img_arr, batch_idxs, modes
    
    def forward(self, img_arr):
        img_arr = F.interpolate(img_arr, (self.input_res, self.input_res), mode='bilinear')
        B, N, H, W = img_arr.size(0),  img_arr.size(1) // 3, img_arr.size(2), img_arr.size(3)
        img_arr, batch_idxs, modes = self.split_input(img_arr)

        rgb_idxs, ir_idxs = np.argwhere(modes == 0).reshape(-1, ), np.argwhere(modes == 1).reshape(-1, )
        rev_idxs = np.argsort(np.concatenate([rgb_idxs, ir_idxs]))
        cur_idxs = np.arange(0, B*N, N)
        mask_old = torch.ones(B*N, dtype=torch.bool)
        mask_old[cur_idxs] = False
        
        rgbs = img_arr[rgb_idxs]
        irs = img_arr[ir_idxs][:, :1]

        x_rgb, q_rgb, k_rgb, v_rgb = self.rgb_tf(rgbs)
        x_ir, q_ir, k_ir, v_ir = self.ir_tf(irs)
        q = [torch.cat([q1, q2], dim=1)[:, rev_idxs][:, mask_old]
             for (q1, q2) in zip(q_rgb, q_ir)]
        k = [torch.cat([k1, k2], dim=1)[:, rev_idxs][:, cur_idxs].repeat_interleave(N-1, 1)
             for (k1, k2) in zip(k_rgb, k_ir)]
        v = [torch.cat([v1, v2], dim=1)[:, rev_idxs][:, cur_idxs].repeat_interleave(N-1, 1)
             for (v1, v2) in zip(v_rgb, v_ir)]
        x = img_arr[cur_idxs].repeat_interleave(N-1, 0)
        out = self.cross_tf(x, q, k, v)
        out = out.view(B, N-1, out.size(-3), out.size(-2), self.in_planes).sum(axis=1).permute(0, 3, 1, 2)
        out = self.pidnethead(out)
        return out

    def save_model(self, path, epoch):
        os.makedirs(f'{path}', exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f'Epoch{epoch}.pt'))

if __name__ == '__main__':
    device = 'cpu'
    model = model = AsyncModel(128)
    model.eval()
    model.to(device)    
    input = torch.randn(5, 12, 256, 256).to(device)
    summary(model, input)

    