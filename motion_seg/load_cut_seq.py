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
import numpy as np
import glob
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.dataset.data_utils import normalize_point_traj, resize_point_traj, sample_point_traj_raw

def load_cut_seq(img_dir, depth_dir, traj_dir, window, input_size, traj_max_num):
    """Load the input data and cut it into batches according to the window size
    """
    # images
    imgs = []
    image_names = sorted(os.listdir(img_dir))
    for img_name in image_names:
        img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_name)), cv2.COLOR_BGR2RGB)
        raw_hw = img.shape[0:2]
        img_resize = cv2.resize(img, (input_size[1], input_size[0]))
        imgs.append(img_resize)

    # midas depths
    depths = []
    depth_names = sorted(glob.glob(os.path.join(depth_dir + "/*.png")))
    for depth_name in depth_names:
        depth = cv2.imread(depth_name, -1) / 65535.0
        depth_resize = cv2.resize(depth, (input_size[1], input_size[0]))
        depths.append(depth_resize)

    # initiate a TrajectorySet object
    trajectories = np.load(os.path.join(traj_dir, "track.npy"), allow_pickle=True).item()
    trajectories.build_invert_indexes()

    # test if the window is larger than the length of the whole sequence
    length = len(imgs)
    if window >= length:
        output = trajectories.sample_inside_window(np.arange(length).tolist(), max_num_tracks=traj_max_num)
        full_traj = np.concatenate([output["locations"][0][:,:,None], output["locations"][1][:,:,None]], 2)
        nor_traj = resize_point_traj(full_traj, raw_hw, input_size)
        nor_traj = normalize_point_traj(nor_traj, input_size)
        full_mask = (1 - output["masks"]).astype(float)[:,:,None]
        sample_idx = output["traj_ids"]
        return [np.stack(imgs, 0)], [np.stack(depths, 0)], [full_traj], [nor_traj], [full_mask], [np.arange(length)], [sample_idx]

    # cut into windows
    img_batchs, depth_batchs, raw_traj_batchs, traj_batchs, mask_batchs = [], [], [], [], []
    time_idx_batchs, sample_idx_batchs = [], []
    num = int(np.ceil(1.0*length / window))
    for i in range(num):
        if i != num - 1:
            idx = np.array(np.arange(i*window, (i+1)*window))
            cur_imgs = imgs[i*window:(i+1)*window]
            cur_depths = depths[i*window:(i+1)*window]
        else:
            idx = np.array(np.arange(length-window, length))
            cur_imgs = imgs[length-window:length]
            cur_depths = depths[length-window:length]

        output = trajectories.sample_inside_window(idx.tolist(), min_length=3, max_num_tracks=traj_max_num)
        s_traj_raw = np.concatenate([output["locations"][0][:,:,None], output["locations"][1][:,:,None]], 2)
        s_traj = resize_point_traj(s_traj_raw, raw_hw, input_size)
        s_traj = normalize_point_traj(s_traj, input_size)
        s_mask = (1 - output["masks"]).astype(float)[:,:,None]
        s_idx = np.array(output["traj_ids"])

        # sample to control the maximum number (hyper-parameter) of trajectories
        img_batchs.append(cur_imgs)
        depth_batchs.append(cur_depths)
        raw_traj_batchs.append(s_traj_raw)
        traj_batchs.append(s_traj)
        mask_batchs.append(s_mask)
        time_idx_batchs.append(idx)
        sample_idx_batchs.append(s_idx)
    return img_batchs, depth_batchs, raw_traj_batchs, traj_batchs, mask_batchs, time_idx_batchs, sample_idx_batchs


