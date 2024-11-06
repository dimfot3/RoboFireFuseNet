# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import sys
sys.path.insert(0, './models/')
from pidnet_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
import math
from torchsummary import summary
import os
from SwinTransformerv2 import SwinTransformerV2
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

class PIDnetTF(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True, channels=3, head_dim=32, window_size=8, input_resolution=(384, 448)):
        super(PIDnetTF, self).__init__()
        self.augment = augment
        self.channels = channels
        self.head_dim = head_dim
        self.window_size = window_size
        self.pos_param = nn.Parameter(torch.randn(2 if channels > 3 else 1, 1))
        # I Branch
        self.conv1 =  nn.Sequential(
                          nn.Conv2d(channels,planes,kernel_size=1, stride=1),
                      )

        self.relu = nn.ReLU(inplace=True)
        self.tf_config = [2, 2, 18, 2]
        self.tf_emb = 96
        self.tf = SwinTransformerV2(img_size=input_resolution, in_chans=planes, embed_dim=self.tf_emb, window_size=window_size, depths=self.tf_config)
        self.tf.load_state_dict(torch.load('./weights/swinv2_pretrain_w78.pth', map_location='cpu'))
        self.reproj = nn.ModuleList([nn.Conv2d(96 * 2**i, planes * 2**i,kernel_size=1, stride=1, padding=1) for i in range(len(self.tf_config))])
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P Branch
        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes * 2, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
            
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)           

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

 
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
    
    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)
        
        return layer
    
    def imgnet_pretrain(self, path):
        try:
            pretrained_state = torch.load(path, map_location='cpu')['state_dict']
        except:
            pretrained_state = torch.load(path, map_location='cpu')['model_state_dict']
        model_dict = self.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'PIDnet: Loaded {} parameters!'.format(len(pretrained_state))
        self.load_state_dict(model_dict, strict = False)
        print(msg)

    def find_mode(self, img):
        """
        returns 0 if rgb and 1 if ir
        """
        if(math.isclose(img[1:].sum(), 0, abs_tol=1e-9)):
            return 1
        return 0

    def make_input(self, x):
        B, N, H, W = x.size()
        x_new = torch.zeros((B, self.channels, H, W), dtype=x.dtype, device=x.device)
        for b in range(B):
            for n in range(N // 3):
                img = x[b, n*3:(n+1)*3]
                if self.find_mode(img) == 1:
                    x_new[b, -1] = img[0] + self.pos_param[n]
                else:
                    x_new[b, :3] = img + self.pos_param[n]
        return x_new

    def forward(self, x):
        x = self.make_input(x)
        x = self.conv1(x)
        interfeat = self.tf.forward_intermediates(x, intermediates_only=True)
        out1, out2, out3, out4 = [self.reproj[i](feat) for i, feat in enumerate(interfeat)]
        x_ = self.layer3_(out2)
        x_d = self.layer3_d(out2)
        
        width_output = x_d.shape[-1]
        height_output = x_d.shape[-2]

        x_ = self.pag3(x_, self.compression3(out3))
        x_d = x_d + F.interpolate(
                        self.diff3(out3),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        if self.augment:
            temp_p = x_
        
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))
        
        x_ = self.pag4(x_, self.compression4(out4))
        x_d = x_d + F.interpolate(
                        self.diff4(out4),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        if self.augment:
            temp_d = x_d
            
        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
                        self.spp(self.layer5(out4)),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment: 
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_
    
    def save_model(self, path, epoch):
        os.makedirs(f'{path}', exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f'Epoch{epoch}.pt'))

from time import time
if __name__ == '__main__':
    device = 'cpu'
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    model = model = PIDnetTF(m=2, n=3, num_classes=2, planes=32, ppm_planes=96, head_planes=128, augment=False, channels=4, window_size=(7, 8), input_resolution=(448, 512))
    model.eval()
    model.to(device)
    iterations = None
    input = torch.randn(1, 4, 448, 512).to(device)
    t0 = time()
    out = model(input)
    t1 = time()
    print(t1 - t0)