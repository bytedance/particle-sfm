# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Transformer-OANet based encoder-decoder network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.network.oanet import OANBlock

class pt_transformer(nn.Module):
    # point trajectory transformer
    def __init__(self, window_size, in_out_channels=16, stride=2):
        super(pt_transformer, self).__init__()
        self.L = window_size
        self.stride = stride
        self.in_out_channels = in_out_channels
        self.input_fc1 = nn.Conv2d(10, 16, (1,1))
        self.fc2 = nn.Conv2d(16, in_out_channels, (1,1))
        self.transformer_model = nn.Transformer(d_model=in_out_channels, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=64, dropout=0.1, activation='relu')
        
    def project(self, traj_pad):
        x = F.relu(self.input_fc1(traj_pad))
        x = F.relu(self.fc2(x))
        return x
    
    def extract_feature(self, traj_pad, pad_mask):
        # input traj data: [B, C, N, L], normalized (x,y) coord, pad_mask: [B, 1, N, L]
        # output features: [B, N, C]
        self.L = traj_pad.shape[-1]
        input_traj = traj_pad.permute(3,0,2,1).reshape(self.L, -1, self.in_out_channels) # [T, N, E]
        input_pad_mask = (pad_mask.reshape(-1, self.L) > 0.5) # [N, T]
        output_feat = self.transformer_model(input_traj, input_traj, \
                src_key_padding_mask=input_pad_mask, tgt_key_padding_mask=input_pad_mask) # [T, N, E]
        output = output_feat.reshape(self.L, traj_pad.shape[0], traj_pad.shape[2], -1) # [L, B, N, E]
        global_output = torch.max(output, dim=0, keepdim=False)[0] # [B, N, E]
        return global_output

    def forward(self, traj_pad, pad_mask):
        # image: [B, 3, H, W, L], depth: [B, 1, H, W, L]
        # input traj data: [B, 2, N, L], normalized (x,y) coord, pad_mask: [B, 1, N, L]
        # Return: geo_feature: [B, C, N]
        traj_pad_proj = self.project(traj_pad)
        traj_feature = self.extract_feature(traj_pad_proj, pad_mask)
        return traj_feature.permute(0,2,1)

class traj_oa_depth(nn.Module):
    # Trajectory classification model with transformer encoder and OANet decoder
    def __init__(self, window_size, input_hw):
        super(traj_oa_depth, self).__init__()
        self.window_size = window_size
        self.input_hw = input_hw
        self.image_grid()
        self.joint_encoder = pt_transformer(window_size)
        self.decoder = OANBlock(net_channels=128, input_channel=self.joint_encoder.in_out_channels, depth=8, clusters=100)
    
    def image_grid(self):
        h, w = self.input_hw[0], self.input_hw[1]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h,w))
        self.xy = np.stack([xx,yy,ones], axis=-1)
        self.xy_t = torch.from_numpy(self.xy).reshape(-1,3).permute(1,0).unsqueeze(0).float().cuda()
        fx, fy = (h + w) / 2.0, (h + w) / 2.0
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])
        self.K_inv = np.linalg.inv(K)
        self.K_inv_t = torch.from_numpy(self.K_inv).unsqueeze(0).float().cuda()
    
    def depth_project(self, depth):
        # depth: [B, 1, H, W, L]
        b, _, h, w, l = depth.shape
        depth_b = depth.permute(0,4,1,2,3).reshape(b*l, 1, h*w)
        point_3d = depth_b * self.K_inv_t.bmm(self.xy_t)
        point_3d_img = point_3d.reshape(b,l,3,h,w).permute(0,2,3,4,1)
        return point_3d_img
    
    def gather_point(self, traj, points):
        b, _, h, w, l = points.shape
        # traj: [B, 2, N, L], points: [B, 3, H, W, L]
        points_src = points.permute(0,4,1,2,3).reshape(b*l,3,h*w)
        traj = traj.permute(0,3,1,2).reshape(b*l, 2, -1)
        traj_1dim = (traj[:,1,:] * h).to(torch.int) * w + (traj[:,0,:] * w).to(torch.int)
        traj_1dim = traj_1dim.unsqueeze(1).repeat(1,3,1).to(torch.int64).clamp(0, h*w-1)
        traj_points = torch.gather(points_src, dim=-1, index=traj_1dim) # [b*l, 3, N]
        traj_3d = traj_points.reshape(b,l,3,-1).permute(0,2,3,1)
        return traj_3d

    def augment_traj(self, depth, traj, mask):
        # point_3d: [B, 3, H, W, L]
        # return: traj_aug: [B, 10, N, L]
        L = traj.shape[-1]
        point_3d = self.depth_project(depth)
        traj_3d = self.gather_point(traj, point_3d)
        motion_2d = torch.zeros_like(traj).to(traj.device)
        motion_2d[:,:,:,:-1] = (traj[:,:,:,1:] - traj[:,:,:,:-1]) * (1.0 - mask[:,:,:,1:])
        motion_3d = torch.zeros_like(traj_3d).to(traj.device)
        motion_3d[:,:,:,:-1] = (traj_3d[:,:,:,1:] - traj_3d[:,:,:,:-1]) * (1.0 - mask[:,:,:,1:])
        traj_aug = torch.cat([traj, motion_2d, traj_3d, motion_3d], dim=1)
        return traj_aug
    
    def forward(self, batch):
        # First extract the image features
        # img: [B, 3, H, W, L], traj: [B, 2, N, L], depth: [B, 1, H, W, L]
        # mask: [B, 1, N]
        depths, trajs, masks = batch["depth"], batch["traj"], batch["mask"]
        aug_trajs = self.augment_traj(depths, trajs, masks)
        feat = self.joint_encoder(aug_trajs, masks)
        mask = self.decoder(feat.unsqueeze(-1))
        mask = torch.sigmoid(mask)
        return mask