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
        self.value1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        
        # Two separate projection layers for the two outputs
        self.out_proj1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        """
        Perform cross-attention between two feature maps.
        Args:
            x1 (Tensor): First feature map of shape (B, K, H, W).
            x2 (Tensor): Second feature map of shape (B, K, H, W).
        Returns:
            Tensor: Two output feature maps of shape (B, K, H, W) and two attention probabilities.
        """
        B, K, H, W = x1.shape

        # Step 1: Linear projections for query, key, and two value maps (V1 and V2)
        Q = self.query(x1).view(B, self.num_heads, self.head_dim, H * W)
        K_ = self.key(x2).view(B, self.num_heads, self.head_dim, H * W) # (B, num_heads, head_dim, H*W)
        V1 = self.value1(x2).view(B, self.num_heads, self.head_dim, H * W)
        V2 = self.value2(x2).view(B, self.num_heads, self.head_dim, H * W)

        # Step 2: Transpose dimensions for attention computation
        Q = Q.permute(0, 1, 3, 2)  # (B, num_heads, H*W, head_dim)
        K_ = K_.permute(0, 1, 2, 3)  # (B, num_heads, head_dim, H*W)
        V1 = V1.permute(0, 1, 3, 2)  # (B, num_heads, H*W, head_dim)
        V2 = V2.permute(0, 1, 3, 2)  # (B, num_heads, H*W, head_dim)

        # Step 3: Scaled dot-product attention
        attn_scores = torch.matmul(Q, K_) / (self.head_dim ** 0.5)  # (B, num_heads, H*W, H*W)
        
        # Attention for V1: traditional softmax along dim=-1 (how queries attend to keys)
        attn_probs1 = torch.softmax(attn_scores, dim=-1)  # (B, num_heads, H*W, H*W)
        attn_probs1 = self.attn_dropout(attn_probs1)
        
        # Attention for V2: use softmax along dim=-2 (how keys attend to queries)
        attn_probs2 = torch.softmax(attn_scores.permute(0, 1, 3, 2), dim=-1)  # (B, num_heads, H*W, H*W)
        # attn_probs2 = attn_probs2.permute(0, 1, 3, 2)  # restore the original shape (B, num_heads, H*W, H*W)
        attn_probs2 = self.attn_dropout(attn_probs2)

        # Step 4: Compute attention-weighted output for V1 and V2
        attn_output1 = torch.matmul(attn_probs1, V1)  # (B, num_heads, H*W, head_dim)
        attn_output2 = torch.matmul(attn_probs2, V2)  # (B, num_heads, H*W, head_dim)
        
        # Step 5: Reshape and combine heads
        attn_output1 = attn_output1.permute(0, 1, 3, 2).contiguous()  # (B, num_heads, head_dim, H*W)
        attn_output2 = attn_output2.permute(0, 1, 3, 2).contiguous()  # (B, num_heads, head_dim, H*W)
        
        attn_output1 = attn_output1.view(B, self.num_heads * self.head_dim, H, W)  # (B, K, H, W)
        attn_output2 = attn_output2.view(B, self.num_heads * self.head_dim, H, W)  # (B, K, H, W)

        # Step 6: Final projection for each output
        output1 = self.out_proj1(attn_output1)  # (B, K, H, W)
        output2 = self.out_proj2(attn_output2)  # (B, K, H, W)

        return output1, output2, attn_probs1, attn_probs2

class AffineEstimator(nn.Module):
    def __init__(self, input_dim=256, h=32, w=32, steps=3):
        super(AffineEstimator, self).__init__()
        self.H, self.W = h, w
        self.conv1 = nn.Conv2d(3 * input_dim, input_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_dim, input_dim//2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(input_dim//2 * self.H * self.W, 128)  # Assuming a 16x16 feature map
        self.fc2 = nn.Linear(128, 4)    # Output: 6 affine parameters (scale, rotation, translation)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.steps = steps

    def generate_affine(self, tx, ty, s, r):
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        a1 = s * cos_r   # [B, 1]
        a2 = -s * sin_r  # [B, 1]
        a3 = tx           # [B, 1]
        a4 = s * sin_r   # [B, 1]
        a5 = s * cos_r   # [B, 1]
        a6 = ty           # [B, 1]
        affine_matrix = torch.cat([a1, a2, a3, a4, a5, a6], dim=1)  # [B, 6]
        affine_matrix = affine_matrix.view(-1, 2, 3)  # [B, 2, 3]
        return affine_matrix

    def forward(self, attn_rgb, att_ir):
        B, num_heads, HW, _ = attn_rgb.size()
        H, W = self.H, self.W
        feat_diff = torch.cat([attn_rgb, att_ir, attn_rgb - att_ir], dim=1)

        x = self.relu(self.conv1(feat_diff))
        x = self.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        tx, ty, s, r = torch.split(x, 1, dim=1) 

        tx = -0.12/self.steps + 2 * (0.12/self.steps) * torch.sigmoid(tx)
        ty = -0.12/self.steps + 2 * (0.12/self.steps) * torch.sigmoid(ty)
        s = 2*(0.1 /self.steps) * torch.sigmoid(s) + 1 - (0.1 /self.steps)
        r = - ((np.pi / 4)/self.steps) + 2 * ((np.pi / 4)/self.steps) * torch.sigmoid(r)
        affine_matrix = self.generate_affine(tx, ty, s, r)
        return affine_matrix

class RobustModule(nn.Module):
    def __init__(self, input_channels, h=32, w=32):
        super(RobustModule, self).__init__()
        self.cross1_rgb_ir = CrossAttention(input_channels, 8)
        self.affine_mod1 = AffineEstimator(input_channels, h, w, steps=3)

    def forward(self, x_rgb_ir, x_ir, tf=None):
        batch_size = x_ir.shape[0]
        total_affine = torch.eye(3, device=x_ir.device).unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: [B, 3, 3]
        for i in range(3):
            our_rgb, out_ir, _, _ = self.cross1_rgb_ir(x_ir, x_rgb_ir)
            affine_mat1 = self.affine_mod1(our_rgb, out_ir) if tf is None else tf

            A = torch.eye(3, device=x_ir.device).unsqueeze(0).repeat(batch_size, 1, 1)
            A[:, :2, :] = affine_mat1  # Put 2x3 affine matrix into 3x3
            total_affine = torch.bmm(total_affine, A)  # Update total affine matrix 

            grid = F.affine_grid(affine_mat1, x_ir.size(), align_corners=False)
            x_ir = F.grid_sample(x_ir, grid, mode='nearest', align_corners=False, padding_mode='zeros') if tf is None else x_ir
        return x_ir, x_rgb_ir, affine_mat1
    
    def save_model(self, path, epoch):
        os.makedirs(f'{path}', exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f'Epoch{epoch}.pt'))

if __name__ == '__main__':
    device = 'cpu'
    module = RobustModule(64)
    rgb, ir = torch.rand((4, 64, 32, 32)), torch.rand((4, 64, 32, 32))
    featss = module(rgb, ir)
    summary(module,(rgb, ir), depth=30, device=device)
    
