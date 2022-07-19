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

"""Pytorch dataset class for loading processed Flyingthings3D
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import glob
from core.dataset.data_utils import read_flow_png, read_seg_gt_png, resize_flow, read_point_traj, resize_point_traj, normalize_point_traj

class flyingthings3d_seq(Dataset):
    def __init__(self, root, transform, split, gap, load_flow, input_size, length=10, pt_thres_min_len=3):
        self.root = root
        self.transform = transform
        self.split = split
        self.gap = gap
        self.load_flow = load_flow
        self.input_size = input_size
        self.length = length
        self.pt_thres_min_len = pt_thres_min_len
        self.glob_sequences()
    
    def glob_sequences(self):
        # Example: rgb: /root/frame_finalpass/TRAIN/A/seq/left/0006.png
        # flow_gt: /root/optical_flow/TRAIN/A/seq/into_future/left/OpticalFlowIntoFuture_0010_L.png
        # motion_gt: /root/motion_labels/TRAIN/A/seq/left/0006.png
        # We organize the train/test data in the sequence manner: load one sequence at once as a sample
        self.seqs = []
        self.seq_imgs, self.seq_flows, self.seq_gt_segs = [], [], []
        self.point_traj, self.pad_mask, self.traj_label, self.depth = [], [], [], []
        split_C = 'TRAIN' if self.split == 'train' else 'TEST'
        seq_cates = ["A", "B", "C"]
        for seq_cate in seq_cates:
            cur_dir = os.path.join(self.root, 'motion_labels', split_C, seq_cate)
            seq_names = glob.glob(cur_dir + '/*')
            seq_left = [os.path.join(cur_dir, seq_name, 'left') for seq_name in seq_names]
            seq_right = [os.path.join(cur_dir, seq_name, 'right') for seq_name in seq_names]
            self.seqs += seq_left
            self.seqs += seq_right
        for seq in self.seqs:
            cate, seq_name, hand = seq.split('/')[-3:]
            num = len(os.listdir(seq))
            if num != 10:
                continue
            # img
            img_seq = seq.replace('motion_labels', 'frames_finalpass')
            imgs = [os.path.join(img_seq, img) for img in sorted(os.listdir(img_seq))]
            self.seq_imgs.append(imgs)
            # gt flow
            flow_seq = os.path.join(self.root, 'optical_flow', split_C, cate, seq_name, 'into_future', hand)
            if not os.path.exists(flow_seq):
                flow_seq = flow_seq.replace('into_future', 'into_past')
            flows = [os.path.join(flow_seq, flow) for flow in sorted(os.listdir(flow_seq))]
            self.seq_flows.append(flows)
            # gt motion seg
            segs = [os.path.join(seq, gt) for gt in sorted(os.listdir(seq))]
            self.seq_gt_segs.append(segs)
            # relative depth
            depth_seq = seq.replace('motion_labels', 'midas_depth')
            depths = [os.path.join(depth_seq, img) for img in sorted(os.listdir(depth_seq))]
            self.depth.append(depths)
            # point trajectory data
            pt_seq = seq.replace('motion_labels', 'point_traj')
            pt = os.path.join(pt_seq, 'pt.npz')
            mask = os.path.join(pt_seq, 'pad_mask.npz')
            traj_label = os.path.join(pt_seq, 'traj_label.npy')
            self.point_traj.append(pt)
            self.pad_mask.append(mask)
            self.traj_label.append(traj_label)


    def __len__(self):
        return len(self.seq_imgs)
    
    def __getitem__(self, idx):
        imgs_name, flows_name, gts_name = self.seq_imgs[idx], self.seq_flows[idx], self.seq_gt_segs[idx]
        depths_name = self.depth[idx]
        assert len(imgs_name) == self.length
        imgs, flows, gts = [], [], []
        depths = []
        for i in range(self.length):
            # load and resize
            img = cv2.cvtColor(cv2.imread(imgs_name[i]), cv2.COLOR_BGR2RGB)
            raw_hw = img.shape[:2]
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
            flow = read_flow_png(flows_name[i])
            flow = resize_flow(flow, self.input_size, to_rgb=True).astype(np.float32)
            gt = read_seg_gt_png(gts_name[i])
            gt = cv2.resize(gt, (self.input_size[1], self.input_size[0]), cv2.INTER_NEAREST).astype(np.float32)
            if len(depths_name) == self.length:
                depth = cv2.imread(depths_name[i], -1) / 65535.0
                depth = cv2.resize(depth, (self.input_size[1], self.input_size[0]))
            else:
                depth = np.zeros_like(gt)
            # transform
            if self.transform is not None:
                img = self.transform(img)
                flow = self.transform(flow)
                gt = self.transform(gt)
                depth = self.transform(depth)
            # append
            imgs.append(img)
            flows.append(flow)
            gts.append(gt)
            depths.append(depth)
        
        imgs, flows, gts = torch.stack(imgs, -1), torch.stack(flows, -1), torch.stack(gts, -1)
        depths = torch.stack(depths, -1)
        
        # load the point trajectory
        point_traj, mask, label = read_point_traj(self.point_traj[idx], self.pad_mask[idx], \
            self.traj_label[idx], max_num=100000) # max_num: 100000
        point_traj = resize_point_traj(point_traj, raw_hw, self.input_size)
        point_traj = normalize_point_traj(point_traj, self.input_size)

        if self.transform is not None:
            point_traj = self.transform(point_traj)
            mask = self.transform(mask)
        label = torch.from_numpy(label)

        batch = {}
        batch["imgs"] = imgs
        batch["flows"] = flows
        batch["gts"] = gts
        batch["point_traj"] = point_traj
        batch["mask"] = mask
        batch["label"] = label
        batch["depths"] = depths
        
        return batch

        

