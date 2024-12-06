# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, './models/')
from torchsummary import summary
import os
from transformers import Swinv2Config
from Swin2 import Swinv2Model, Swinv2Stage, Swinv2PatchMerging, Swinv2Embeddings, LambdaLayer
from einops.layers.torch import Rearrange
from torchvision.models import resnet50

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        """
        Cross-attention module for feature maps.
        Args:
            channels (int): Number of input channels (K).
            num_heads (int): Number of attention heads.
        """
        super(CrossAttention, self).__init__()
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        """
        Perform cross-attention between two feature maps.
        Args:
            x1 (Tensor): First feature map of shape (B, K, H, W).
            x2 (Tensor): Second feature map of shape (B, K, H, W).
        Returns:
            Tensor: Output feature map of shape (B, K, H, W).
        """
        B, K, H, W = x1.shape

        # Linear projections for query, key, and value
        Q = self.query(x1).view(B, self.num_heads, self.head_dim, H * W)
        K_ = self.key(x2).view(B, self.num_heads, self.head_dim, H * W)
        V = self.value(x2).view(B, self.num_heads, self.head_dim, H * W)

        # Transpose dimensions for attention computation
        Q = Q.permute(0, 1, 3, 2)  # (B, num_heads, H*W, channels_per_head)
        K_ = K_.permute(0, 1, 2, 3)  # (B, num_heads, channels_per_head, H*W)
        V = V.permute(0, 1, 3, 2)  # (B, num_heads, H*W, channels_per_head)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K_) / (self.head_dim ** 0.5)  # (B, num_heads, H*W, H*W)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, V)  # (B, num_heads, H*W, channels_per_head)
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous()  # (B, num_heads, channels_per_head, H*W)
        attn_output = attn_output.view(B, K, H, W)  # Merge heads

        # Final projection
        output = self.out_proj(attn_output) + x2
        return output

class ParameterEstimator(nn.Sequential):
    def __init__(self, input_channels, output_dim, activation_fn=nn.ReLU, use_dropout=True, dropout_prob=0.3):
        """
        A sequential model that gradually reduces spatial dimensions and outputs a single value.
        Args:
            input_channels (int): Number of input channels (C).
            output_dim (int): Dimension of the output.
            activation_fn (nn.Module): Activation function to use (default: ReLU).
            use_dropout (bool): Whether to include dropout layers.
            dropout_prob (float): Dropout probability, if use_dropout is True.
        """
        super(ParameterEstimator, self).__init__()

        assert input_channels % 8 == 0, "input_channels must be divisible by 8."

        # Define the building blocks
        self.add_module("conv1", nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, stride=1, padding=1))
        self.add_module("bn1", nn.BatchNorm2d(input_channels // 2))
        self.add_module("activation1", activation_fn())
        if use_dropout:
            self.add_module("dropout1", nn.Dropout2d(dropout_prob))
        self.add_module("avgpool1", nn.AvgPool2d(kernel_size=2, stride=2))  # Downsample spatial dims by 2

        self.add_module("conv2", nn.Conv2d(input_channels // 2, input_channels // 4, kernel_size=3, stride=1, padding=1))
        self.add_module("bn2", nn.BatchNorm2d(input_channels // 4))
        self.add_module("activation2", activation_fn())
        if use_dropout:
            self.add_module("dropout2", nn.Dropout2d(dropout_prob))
        self.add_module("avgpool2", nn.AvgPool2d(kernel_size=2, stride=2))  # Downsample spatial dims by 2

        self.add_module("conv3", nn.Conv2d(input_channels // 4, input_channels // 8, kernel_size=3, stride=1, padding=1))
        self.add_module("bn3", nn.BatchNorm2d(input_channels // 8))
        self.add_module("activation3", activation_fn())
        if use_dropout:
            self.add_module("dropout3", nn.Dropout2d(dropout_prob))
        self.add_module("avgpool3", nn.AvgPool2d(kernel_size=2, stride=2))  # Downsample spatial dims by 2

        self.add_module("final_pool", nn.AdaptiveAvgPool2d(1))  # Final global pooling to 1x1
        self.add_module("flatten", nn.Flatten())  # Flatten to (B, C)
        self.add_module("fc", nn.Linear(input_channels // 8, output_dim))  # Linear layer to output single value
        self.add_module("sigm", nn.Sigmoid())

    def forward(self, x):
        return super(ParameterEstimator, self).forward(x)

class RobustModule(nn.Module):
    def __init__(self, input_channels):
        super(RobustModule, self).__init__()
        resnet = resnet50(pretrained=True)

        # Modify the first convolution layer to accept `input_channels`
        self.conv1_rgb = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_rgb = resnet.bn1
        self.relu_rgb = resnet.relu
        self.layer1_rgb = resnet.layer1  # First residual layer
        self.layer2_rgb = resnet.layer2  # Second residual layer
        self.reduce_channels_rgb = nn.Conv2d(512, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample_rgb = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.conv1_ir = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_ir = resnet.bn1
        self.relu_ir = resnet.relu
        self.layer1_ir = resnet.layer1  # First residual layer
        self.layer2_ir = resnet.layer2  # Second residual layer
        self.reduce_channels_ir = nn.Conv2d(512, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample_ir = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.cross1 = CrossAttention(input_channels, 8)
        self.theta_estimator = ParameterEstimator(input_channels, 1)
        self.prost_cross1 = nn.Sequential(nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1), nn.BatchNorm2d(input_channels), nn.ReLU())
        
        
        self.cross2 = CrossAttention(input_channels, 8)
        self.scale_estimator = ParameterEstimator(input_channels, 1)
        self.prost_cross2 = nn.Sequential(nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1), nn.BatchNorm2d(input_channels), nn.ReLU())

        self.cross3 = CrossAttention(input_channels, 8)
        self.trans_estimator = ParameterEstimator(input_channels, 2)
        self.prost_cross3 = nn.Sequential(nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1), nn.BatchNorm2d(input_channels), nn.ReLU())

    def get_affine_matrix(self, angles=None, translations=None, scales=None, device='cpu'):
        B = None
        if angles is not None:
            B = angles.shape[0]
        elif translations is not None:
            B = translations.shape[0]
        elif scales is not None:
            B = scales.shape[0]
        else:
            raise ValueError("At least one of angles, translations, or scales must be provided to determine batch size.")
        
        if angles is None:
            angles = torch.zeros((B, 1)).to(device)
        
        if translations is None:
            translations = torch.zeros((B, 2)).to(device)
        
        if scales is None:
            scales = torch.ones((B, 1)).to(device)
        
        cos_theta = torch.cos(angles).squeeze(1)  # Shape: (B,)
        sin_theta = torch.sin(angles).squeeze(1)  # Shape: (B,)
        
        scale_factors = scales.squeeze(1)  # Shape: (B,)
        cos_theta *= scale_factors
        sin_theta *= scale_factors
        
        rotation_matrices = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=1),  # First row
            torch.stack([sin_theta, cos_theta], dim=1)   # Second row
        ], dim=1)  # Shape: (B, 2, 2)
        
        affine_matrices = torch.cat([rotation_matrices, translations.unsqueeze(2)], dim=2)  # Shape: (B, 2, 3)
        return affine_matrices
    
    def extend_to_3x3(self, mat):
        B, _, _ = mat.shape
        bottom_row = torch.tensor([0, 0, 1], device=mat.device).view(1, 1, 3).repeat(B, 1, 1)
        return torch.cat([mat, bottom_row], dim=1)  # Shape: (B, 3, 3)
    
    def forward(self, x_rgb, x_ir, tf=None):
        x_rgb_tmp = self.conv1_rgb(x_rgb)
        x_rgb_tmp = self.bn1_rgb(x_rgb_tmp)
        x_rgb_tmp = self.relu_rgb(x_rgb_tmp)
        x_rgb_tmp = self.layer1_rgb(x_rgb_tmp)
        x_rgb_tmp = self.layer2_rgb(x_rgb_tmp)
        x_rgb_tmp = self.reduce_channels_rgb(x_rgb_tmp)
        x_rgb_ir = self.upsample_rgb(x_rgb_tmp)

        # x_ir_tmp = self.conv1_rgb(x_rgb)
        # x_ir_tmp = self.bn1_rgb(x_ir_tmp)
        # x_ir_tmp = self.relu_rgb(x_ir_tmp)
        # x_ir_tmp = self.layer1_rgb(x_ir_tmp)
        # x_ir_tmp = self.layer2_rgb(x_ir_tmp)
        # x_ir_tmp = self.reduce_channels_rgb(x_ir_tmp)
        # x_ir_tf = self.upsample_rgb(x_ir_tmp)

        # x_cross_out1 = self.cross1(x_ir_tf, x_rgb_ir)
        
        # translation = self.trans_estimator(x_cross_out1)
        # translation_mat = self.get_affine_matrix(translations=translation, device=x_rgb.device).view(-1, 2, 3).to(x_rgb.device)
        # # grid = F.affine_grid(translation_mat if tf==None else tf, x_ir.size(), align_corners=False)
        # # x_ir = F.grid_sample(x_ir, grid, align_corners=False)
        # x_cross_out1 = self.prost_cross1(x_cross_out1) + x_ir

        # x_cross_out2 = self.cross2(x_ir_tf, x_cross_out1)
        # # theta = self.theta_estimator(x_cross_out2) * np.pi
        # # theta_mat = self.get_affine_matrix(angles=theta, device=x_rgb.device).view(-1, 2, 3).to(x_rgb.device)
        # # grid = F.affine_grid(theta_mat if tf==None else tf, x_ir.size(), align_corners=False)
        # # x_ir = F.grid_sample(x_ir, grid, align_corners=False)
        # x_cross_out2 = self.prost_cross2(x_cross_out2) + x_ir
       
        # x_cross_out3 = self.cross3(x_ir_tf, x_cross_out2)
        # # scale = 1 - self.scale_estimator(x_cross_out3) / 10
        # # scale_mat = self.get_affine_matrix(scales=scale, device=x_rgb.device).view(-1, 2, 3).to(x_rgb.device)
        # # grid = F.affine_grid(scale_mat if tf==None else tf, x_ir.size(), align_corners=False)
        # # x_ir = F.grid_sample(x_ir, grid, align_corners=False)
        # x_ir = self.prost_cross3(x_cross_out3) + x_ir

        # translation_mat_3x3 = self.extend_to_3x3(translation_mat)
        # # theta_mat_3x3 = self.extend_to_3x3(theta_mat)
        # # scale_mat_3x3 = self.extend_to_3x3(scale_mat)
        # # combined_mat_3x3 = scale_mat_3x3 @ theta_mat_3x3 @ translation_mat_3x3
        # combined_mat_3x3 = translation_mat_3x3
        # final_tf = combined_mat_3x3[:, :2, :]  # Shape: (B, 2, 3)
        return x_ir, x_rgb_ir, None
    
    def save_model(self, path, epoch):
        os.makedirs(f'{path}', exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f'Epoch{epoch}.pt'))

if __name__ == '__main__':
    device = 'cpu'
    module = RobustModule(64)
    rgb, ir = torch.rand((4, 64, 32, 32)), torch.rand((4, 64, 32, 32))
    featss = module(rgb, ir)
    summary(module,(rgb, ir), depth=30, device=device)
    
