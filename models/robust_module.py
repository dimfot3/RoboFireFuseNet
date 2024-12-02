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
from transformers import Swinv2Config
from Swin2 import Swinv2Model, Swinv2Stage, Swinv2PatchMerging, Swinv2Embeddings, LambdaLayer
from math import gcd
from einops.layers.torch import Rearrange
BatchNorm2d = nn.BatchNorm2d
from pidnet import PIDNet
bn_mom = 0.1
algc = False


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        
        reduced_channels = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),                     
            nn.Linear(reduced_channels, in_channels),  
            nn.Sigmoid()                               
        )
        
        # Convolution to map to output channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_pooled = self.avg_pool(x).view(b, c) 
        max_pooled = self.max_pool(x).view(b, c)  
        
        avg_weights = self.channel_attention(avg_pooled) 
        max_weights = self.channel_attention(max_pooled) 
        
        channel_weights = (avg_weights + max_weights).view(b, c, 1, 1)  
        
        x = x * channel_weights  
        
        x = self.conv(x)
        
        return x

class RobustModule(nn.Module):

    def __init__(self, input_size, input_channels):
        super(RobustModule, self).__init__()
        input_resolution = np.array(input_size)
        # I Branch
        self.rgb_conv0 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.ir_conv0 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        tf_configuration = Swinv2Config(window_size=8, image_size=input_resolution, num_channels=input_channels)
        self.rgb_vit1 = Swinv2Stage(
                                config=tf_configuration,
                                dim=int(tf_configuration.embed_dim),
                                input_resolution=(tf_configuration.image_size[0], tf_configuration.image_size[1]),
                                depth=2,
                                num_heads=tf_configuration.num_heads[0],
                                drop_path=0.05,
                                downsample=None,
                                pretrained_window_size=tf_configuration.pretrained_window_sizes[0])
        self.ir_vit1 = Swinv2Stage(
                                config=tf_configuration,
                                dim=int(tf_configuration.embed_dim),
                                input_resolution=(tf_configuration.image_size[0], tf_configuration.image_size[1]),
                                depth=2,
                                num_heads=tf_configuration.num_heads[0],
                                drop_path=0.05,
                                downsample=None,
                                pretrained_window_size=tf_configuration.pretrained_window_sizes[0])
        self.rgb_vit1_seq = nn.Sequential(Swinv2Embeddings(Swinv2Config(window_size=8, num_channels=input_channels, patch_size=1, embed_dim=96)), \
                                LambdaLayer(lambda xinp: self.rgb_vit1(xinp[0], xinp[1])), \
                                LambdaLayer(lambda xinp: Rearrange('b (h w) c-> b c h w', h=xinp[2][-2]).forward(xinp[0])))
        self.ir_vit1_seq = nn.Sequential(Swinv2Embeddings(Swinv2Config(window_size=8, num_channels=input_channels, patch_size=1, embed_dim=96)), \
                                LambdaLayer(lambda xinp: self.ir_vit1(xinp[0], xinp[1])), \
                                LambdaLayer(lambda xinp: Rearrange('b (h w) c-> b c h w', h=xinp[2][-2]).forward(xinp[0])))
        
        self.rgb_conv1 = nn.Conv2d(int(tf_configuration.embed_dim), input_channels, kernel_size=1)
        self.ir_conv1 = nn.Conv2d(int(tf_configuration.embed_dim), input_channels, kernel_size=1)

        self.fuse_rgb = ChannelAttentionModule(input_channels*2, input_channels, 16)
        self.fuse_ir = ChannelAttentionModule(input_channels*2, input_channels, 16)

        self.rgb_vit2 = Swinv2Stage(
                    config=tf_configuration,
                    dim=int(tf_configuration.embed_dim),
                    input_resolution=(tf_configuration.image_size[0], tf_configuration.image_size[1]),
                    depth=2,
                    num_heads=tf_configuration.num_heads[0],
                    drop_path=0.05,
                    downsample=Swinv2PatchMerging,
                    pretrained_window_size=tf_configuration.pretrained_window_sizes[0],
                )
        self.ir_vit2 = Swinv2Stage(
                    config=tf_configuration,
                    dim=int(tf_configuration.embed_dim),
                    input_resolution=(tf_configuration.image_size[0], tf_configuration.image_size[1]),
                    depth=2,
                    num_heads=tf_configuration.num_heads[0],
                    drop_path=0.05,
                    downsample=Swinv2PatchMerging,
                    pretrained_window_size=tf_configuration.pretrained_window_sizes[0],
                )
        self.rgb_vit2_seq = nn.Sequential(Swinv2Embeddings(Swinv2Config(window_size=8, num_channels=input_channels, patch_size=1, embed_dim=96)), \
                                LambdaLayer(lambda xinp: self.rgb_vit2(xinp[0], xinp[1])), \
                                LambdaLayer(lambda xinp: Rearrange('b (h w) c-> b c h w', h=xinp[2][-2]).forward(xinp[0])))
        self.ir_vit2_seq = nn.Sequential(Swinv2Embeddings(Swinv2Config(window_size=8, num_channels=input_channels, patch_size=1, embed_dim=96)), \
                                LambdaLayer(lambda xinp: self.ir_vit2(xinp[0], xinp[1])), \
                                LambdaLayer(lambda xinp: Rearrange('b (h w) c-> b c h w', h=xinp[2][-2]).forward(xinp[0])))
        
        self.comb_feat = nn.Sequential(nn.Conv2d(int(tf_configuration.embed_dim)*4, int(tf_configuration.embed_dim), kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(int(tf_configuration.embed_dim)), 
                                     nn.Flatten(), 
                                     nn.ReLU(),
                                     nn.Linear(int(tf_configuration.embed_dim)*(input_resolution[0]//2)*(input_resolution[1]//2), 6))
        
        self.out_rgb = nn.Sequential(nn.Conv2d(int(tf_configuration.embed_dim)*2, int(tf_configuration.embed_dim), kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(int(tf_configuration.embed_dim)), 
                                     nn.Flatten(), 
                                     nn.ReLU(),
                                     nn.Linear(int(tf_configuration.embed_dim)*(input_resolution[0]//2)*(input_resolution[1]//2), 6))
        self.out_ir = nn.Sequential(nn.Conv2d(int(tf_configuration.embed_dim)*2, int(tf_configuration.embed_dim), kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(int(tf_configuration.embed_dim)), 
                                     nn.Flatten(), 
                                     nn.ReLU(),
                                     nn.Linear(int(tf_configuration.embed_dim)*(input_resolution[0]//2)*(input_resolution[1]//2), 6))

    def forward(self, rgb_feat=None, ir_feat=None):
        return_list = [rgb_feat, ir_feat]
        if rgb_feat != None:        # recreate ir feature
            ir_feat_new = self.rgb_conv0(rgb_feat)
            ir_feat_new = self.rgb_vit1_seq(ir_feat_new)
            ir_feat_new = self.rgb_conv1(ir_feat_new)
            return_list[1] = ir_feat_new
        if ir_feat != None:         # recreate rgb feature
            rgb_feat_new = self.ir_conv0(ir_feat)
            rgb_feat_new = self.ir_vit1_seq(rgb_feat_new)
            rgb_feat_new = self.ir_conv1(rgb_feat_new)
            return_list[0] = rgb_feat_new
        if (rgb_feat!= None) and (ir_feat != None):         # estimate misalignment
            rgb_fused = self.fuse_rgb(torch.cat((rgb_feat, rgb_feat_new), dim=1))
            rgb_tf = self.rgb_vit2_seq(rgb_fused)
            ir_fused = self.fuse_ir(torch.cat((ir_feat, ir_feat_new), dim=1))
            ir_tf = self.ir_vit2_seq(ir_fused)
            f_tf = self.comb_feat(torch.cat((rgb_tf, ir_tf), dim=1))

            rgb_tf, ir_tf = self.out_rgb(rgb_tf), self.out_ir(ir_tf)
            return_list += [f_tf, rgb_tf, ir_tf]
        return return_list
    
    def save_model(self, path, epoch):
        os.makedirs(f'{path}', exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f'Epoch{epoch}.pt'))

from time import time
if __name__ == '__main__':
    device = 'cpu'
    module = RobustModule((256//8, 256//8), 64)
    rgb, ir = torch.rand((4, 64, 32, 32)), torch.rand((4, 64, 32, 32))
    featss = module(rgb_feat = rgb, ir_feat=ir)
    print(featss[0].shape, featss[1].shape, featss[2:][0].shape)
    
