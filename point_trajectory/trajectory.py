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

import os, sys
import torch
import numpy as np
import scipy.ndimage

from .utils import gradient
from .optimize.build import particlesfm

def grid_sample(data, xy):
    # sample flow/feature value by xy indices
    # data: [C, H, W] of torch.tensor, xy: [N, 2], return: [N, C]
    data = data.unsqueeze(0)
    xy = torch.from_numpy(xy).float().to(data.device)
    xy = xy.unsqueeze(0).unsqueeze(0)
    H, W = data.shape[2], data.shape[3]
    xy[:,:,:,0] /= ((W - 1) / 2)
    xy[:,:,:,1] /= ((H - 1) / 2)
    xy -= 1
    out = torch.nn.functional.grid_sample(data, xy, align_corners=True)
    out = out.squeeze(0).squeeze(1).permute(1,0).cpu().numpy()
    return out

def motion_boundary(flow, thres=0.02):
    f_dx, f_dy = gradient(flow)
    motion = np.sqrt(f_dx ** 2 + f_dy ** 2)
    mask = (motion > thres * np.linalg.norm(flow, ord=2, axis=-1))
    return mask

def step_forward(xys, flow, occ_map, mb_mask):
    # Step and check
    H, W = occ_map.shape[0], occ_map.shape[1]
    occ_map_t = torch.from_numpy(occ_map).unsqueeze(0).float()
    occ_cond = grid_sample(occ_map_t, np.array(xys))
    occ_cond = (occ_cond > 0.1)
    mb_map_t = torch.from_numpy(mb_mask).unsqueeze(0).float()
    mb_cond = grid_sample(mb_map_t, np.array(xys))
    mb_cond = (mb_cond > 0.1)

    next_xys = np.array(xys) + np.array(flow)
    valid_cond = (next_xys[:,0] > 0) * (next_xys[:,0] < W - 1) * \
        (next_xys[:,1] > 0) * (next_xys[:,1] < H - 1)
    # for sintel, we don't use this
    # for some in-the-wilds, we use motion boundary
    #flags = valid_cond * (1.0 - np.squeeze(occ_cond)) * (1.0 - np.squeeze(mb_cond))
    flags = valid_cond * (1.0 - np.squeeze(occ_cond))
    return next_xys, flags

# python implementation of the Trajectory class
class Trajectory(object):
    def __init__(self, start_time, start_xy, buffer_size=0):
        self.buffer_size = buffer_size
        self.times = []
        self.xys = []
        self.buffer_xys = []
        self.extend(start_time, start_xy)

    def extend(self, time, xy):
        self.times.append(time)
        self.buffer_xys.append(xy)
        if len(self.buffer_xys) > self.buffer_size:
            self.xys.append(self.buffer_xys[0])
            self.buffer_xys.pop(0)

    def length(self):
        return len(self.xys) + len(self.buffer_xys)

    def get_tail_location(self):
        if self.length() == 0:
            raise ValueError("Error!The trajectory is empty!")
        if len(self.buffer_xys) == 0:
            return self.xys[-1]
        else:
            return self.buffer_xys[-1]

    def clear_buffer(self):
        self.xys.extend(self.buffer_xys)
        self.buffer_xys = []

    def set_buffer_xy(self, index, xy):
        self.buffer_xys[index] = xy

class IncrementalTrajectorySet(object):
    def __init__(self, total_length, img_h, img_w, sample_ratio, buffer_size=0):
        self.total_length = total_length
        self.ratio = sample_ratio
        self.h, self.w = img_h, img_w
        self.buffer_size = buffer_size
        self.active_trajs = []
        self.full_trajs = []

        self.all_candidates = self.generate_all_candidates()
        self.sample_candidates = np.reshape(np.copy(self.all_candidates), (-1,2))

    def generate_all_candidates(self):
        x, y = np.arange(0, self.w), np.arange(0, self.h)
        xx, yy = np.meshgrid(x, y)
        xys = np.stack([xx, yy], -1)
        s_xys = xys[::self.ratio, ::self.ratio, :]
        return s_xys

    def new_traj_all(self, start_times, start_xys):
        for time, xy in zip(start_times, start_xys):
            t = particlesfm.Trajectory(time, xy, buffer_size=self.buffer_size)
            self.active_trajs.append(t)

    def get_cur_pos(self):
        # Get all the current traj positions
        cur_pos = []
        for i in range(len(self.active_trajs)):
            cur_pos.append(self.active_trajs[i].get_tail_location())
        return np.array(cur_pos)

    def extend_all(self, next_xys, next_time, flags, thres=2):
        # Extend all the trajs
        assert len(next_xys) == len(self.active_trajs)
        assert len(flags) == len(next_xys)

        # also check the next sample candidates
        occupied_map = np.zeros((self.h, self.w, 1))

        self.new_active_trajs = []
        for i in range(len(flags)):
            next_xy, flag = next_xys[i], flags[i]
            if not flag:
                self.active_trajs[i].clear_buffer()
                self.full_trajs.append(self.active_trajs[i])
            else:
                occupied_map[int(next_xy[1]), int(next_xy[0])] = 1
                self.active_trajs[i].extend(next_time, next_xy)
                self.new_active_trajs.append(self.active_trajs[i])
        self.active_trajs = self.new_active_trajs

        # generate the next sample candidates
        occupied_map_trans = scipy.ndimage.morphology.distance_transform_edt(1.0 - occupied_map)
        sample_map = (occupied_map_trans > self.ratio)[::self.ratio, ::self.ratio, 0]
        self.sample_candidates = np.copy(self.all_candidates[sample_map])

    def clear_active(self):
        for traj in self.active_trajs:
            traj.clear_buffer()
            self.full_trajs.append(traj)
        self.active_trajs = []

    # only support sized-3 buffer ([0, 1, 2])
    def optimize_buffer(self, flow01_map, flow12_map, flow02_map, occ02_map, next_time, upper_flow=20.0):
        # check the traj buffer and fuse those who are ready
        xys_012, idxs = [], []
        h, w = flow01_map.shape[0], flow01_map.shape[1]
        for i in range(len(self.active_trajs)):
            if len(self.active_trajs[i].buffer_xys) == 3:
                buffer_xys = np.stack(self.active_trajs[i].buffer_xys, 0)
                xys_012.append(buffer_xys)
                idxs.append(i)
        xys_012 = np.stack(xys_012, 0) # [N, 3, 2]

        # sample flows
        flow01_map_t = torch.from_numpy(flow01_map).permute(2,0,1).float()
        flow02_map_t = torch.from_numpy(flow02_map).permute(2,0,1).float()
        flow01 = grid_sample(flow01_map_t, xys_012[:,0,:])
        flow02 = grid_sample(flow02_map_t, xys_012[:,0,:])
        occ02_map_t = torch.from_numpy(occ02_map).unsqueeze(0).float()
        occ02 = grid_sample(occ02_map_t, xys_012[:,0,:])
        loss02_scale = (1.0 - occ02) * (np.linalg.norm(flow02, axis=-1, keepdims=True) < upper_flow)

        # optimization
        xys_1_ref = xys_012[:,0,:] + flow01
        xys_2_ref = xys_012[:,0,:] + flow02
        new_xys12 = particlesfm.optimize_location(\
                np.reshape(xys_012[:,1:,:], (xys_012.shape[0], 4)), \
                xys_1_ref, xys_2_ref, loss02_scale, flow12_map, xys_012.shape[0], w, h)
        new_xys12 = np.reshape(new_xys12, (new_xys12.shape[0], 2, 2))

        # update the buffer
        for i in range(len(idxs)):
            new_xy12 = new_xys12[i]
            idx = idxs[i]
            self.active_trajs[idx].set_buffer_xy(1, new_xy12[0])
            self.active_trajs[idx].set_buffer_xy(2, new_xy12[1])

