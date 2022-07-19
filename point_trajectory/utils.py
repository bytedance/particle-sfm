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

import os
import numpy as np
import glob
import torch
import torch.nn.functional as F
import cv2

TAG_FLOAT = 202021.25

def load_flows(dir):
    flows = []
    flow_names = glob.glob(dir + "/*.flo")
    for name in sorted(flow_names):
        flow = read_flo(name)
        flows.append(flow)
    return flows

def load_images(dir):
    images = []
    image_names = glob.glob(dir + "/*.png") + glob.glob(dir + "/*.jpg")
    for name in sorted(image_names):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

def get_oob_mask(flow_1_2):
    B, _, H, W = flow_1_2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([1, 2, H, W])
    coord[0, 0] = ww
    coord[0, 1] = hh
    coord = coord.repeat(B,1,1,1).to(flow_1_2.device)
    target_range = coord + flow_1_2
    m1 = (target_range[:,0,:,:] < 0) + (target_range[:,0,:,:] > W - 1)
    m2 = (target_range[:,1,:,:] < 0) + (target_range[:,1,:,:] > H - 1)
    return (m1 + m2).float()


def backward_flow_warp(im2, flow_1_2):
    B, _, H, W = im2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([1, 2, H, W])
    coord[0, 0, :, :] = ww
    coord[0, 1, :, :] = hh
    coord = coord.repeat(B,1,1,1).to(flow_1_2.device)
    sample_grids = coord + flow_1_2
    sample_grids[:,0,:,:] /= (W - 1) / 2
    sample_grids[:,1,:,:] /= (H - 1) / 2
    sample_grids -= 1
    out = F.grid_sample(im2, sample_grids.permute(0,2,3,1), align_corners=True)
    return out

def get_occ_mask(flow, flow_b, thres):
    warp_flow_1_2 = backward_flow_warp(flow_b, flow)  # using latter to sample former
    err_1 = torch.norm(warp_flow_1_2 + flow, dim=1)
    mask_1 = (err_1 > thres)
    oob_mask_1 = get_oob_mask(flow)
    mask_1 = torch.clamp(mask_1 + oob_mask_1, 0, 1)
    mask = (mask_1 > 0)
    return mask, err_1

def flow_check(flows, flows_b, thres):
    error_maps, occ_maps = [], []
    for f, f_b in zip(flows, flows_b):
        f_t = torch.from_numpy(f).permute(2,0,1).unsqueeze(0).float()
        f_b_t = torch.from_numpy(f_b).permute(2,0,1).unsqueeze(0).float()
        occ_map, err_map = get_occ_mask(f_t, f_b_t, thres)
        occ_map = occ_map.squeeze().numpy()
        err_map = err_map.squeeze().numpy()
        
        error_maps.append(err_map)
        occ_maps.append(occ_map)
    return error_maps, occ_maps

def gradient(img):
    # compute the image gradient
    dx, dy = np.zeros_like(img), np.zeros_like(img)
    dx[:,:-1,:] = np.abs(img[:,:-1,:] - img[:,1:,:])
    dy[:-1,:,:] = np.abs(img[:-1,:,:] - img[1:,:,:])
    dx, dy = np.mean(dx, -1), np.mean(dy, -1)
    return dx, dy

def gradient_map(images):
    grads = []
    for img in images:
        grad = np.zeros_like(img)[:,:,:1]
        grads.append(grad)
    return grads

color_bank = [(255,0,0), (0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(100,100,0),(0,100,100),(100,0,100),(100,100,100)]
def draw_traj(imgs, traj_xys, traj_times):
    L, h, w, _ = imgs.shape
    N = len(traj_xys)
    vis_imgs = [imgs[i][:,:,::-1].copy() for i in range(L)]
    #select_idx = np.random.randint(0, N-1, 1000)
    #for i in select_idx:
    for i in range(N):
        color = color_bank[i%len(color_bank)]
        single_traj = traj_xys[i]
        single_time = traj_times[i]
        leng = len(single_traj)
        if leng < 3:
            continue
        for j in range(leng):
            x, y = single_traj[j]
            x, y = int(x), int(y)
            time = int(single_time[j])
            cv2.circle(vis_imgs[time], center=(x,y), radius=2, color=color, thickness=2)
    return vis_imgs
