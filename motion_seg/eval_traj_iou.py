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

"""Evaluate the motion segmentation metric with trajectories
"""
import argparse
import os
import cv2
import glob
import numpy as np
import torch
import sklearn.metrics

def load_trajs(dir):
    trajectories = np.load(os.path.join(dir, "track.npy"), allow_pickle=True).item().as_dict()
    trajs, labels, times = {}, {}, {}
    for key in trajectories:
        trajs[key] = trajectories[key]["locations"]
        labels[key] = trajectories[key]["labels"]
        times[key] = trajectories[key]["frame_ids"]
    return trajs, times, labels

def load_images(dir):
    images = []
    image_names = glob.glob(dir + "/*.png") + glob.glob(dir + "/*.jpg")
    for name in sorted(image_names):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

def load_masks(dir):
    masks = []
    names = glob.glob(dir + "/*.png")
    for name in sorted(names):
        mask = 1.0 - cv2.imread(name)[:,:,0] / 255.0
        masks.append(mask)
    return masks

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

def seg_metrics(labels, gt_labels):
    # labels: [N, 1]
    # metrics: IoU, "sklearn defined" Precision, Recall, F1-score
    pred, gt = np.array(labels > 0.5), np.array(gt_labels > 0.5)
    intersection = np.sum(pred * gt)
    union = np.sum(1.0 - (1.0 - pred) * (1.0 - gt))
    iou = intersection / (union + 1e-6)
    precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(\
        y_true=gt, y_pred=pred, average='binary', zero_division=0)
    return [iou, precision, recall, f1_score]


def per_img_traj_metrics(imgs, gtmasks, trajs, times, labels):
    # evaluate the motion seg metrics with trajectories
    num = len(imgs)
    mean_metrics = []
    img_xys, img_labels = {}, {}
    # accumulate all the trajectory pixels
    for key in trajs.keys():
        time = times[key]
        for i in range(len(time)):
            xy = trajs[key][i]
            l = labels[key][i]
            if time[i] in img_xys.keys():
                img_xys[time[i]].append(xy)
                img_labels[time[i]].append(l)
            else:
                img_xys[time[i]] = [xy]
                img_labels[time[i]] = [l]

    for i in range(num-1):
        gtmask = gtmasks[i]
        # skip the frame if only contain too small masks
        if np.sum(gtmask) < 10:
            continue
        # get the traj locations on this image
        xys, traj_labels = img_xys[i], img_labels[i]
        xys = np.stack(xys, 0)
        traj_labels = np.expand_dims(np.stack(traj_labels, 0), -1)
        # get the groundtruth masks on the traj locations
        gtmask_t = torch.from_numpy(gtmask).unsqueeze(0).float()
        gt_traj_labels = grid_sample(gtmask_t, xys)
        # metrics
        traj_metrics = seg_metrics(traj_labels, gt_traj_labels)
        mean_metrics.append(traj_metrics)
    if len(mean_metrics) == 0:
        return None
    mean_metrics = np.stack(mean_metrics, 0)
    return mean_metrics

def main(args):
    seqs = sorted(os.listdir(args.root_dir))
    metrics_list = []
    for seq in seqs:
        if seq == 'sleeping_1' or seq == 'sleeping_2' or seq == 'mountain_1':
            continue
        if seq == 'ambush_7' or seq == 'bamboo_1' or seq == 'bamboo_2' \
            or seq == 'bandage_1' or seq == 'bandage_2' or seq == 'shaman_2':
            continue
        img_dir = os.path.join(args.root_dir, seq, "images")
        traj_dir = os.path.join(args.root_dir, seq, "trajectories_labeled")
        gt_dir = os.path.join(args.gt_dir, seq)

        # load data
        imgs = load_images(img_dir)
        gtmasks = load_masks(gt_dir)
        trajs, times, labels = load_trajs(traj_dir)
        # calculate metrics
        seq_metrics = per_img_traj_metrics(\
                imgs, gtmasks, trajs, times, labels)
        print(seq)
        print(np.mean(seq_metrics, 0))
        metrics_list.append(seq_metrics)
    metrics_list = np.concatenate(metrics_list, 0)
    print("Mean metrics: ")
    print(np.mean(metrics_list, 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir")
    parser.add_argument("--gt_dir")
    args = parser.parse_args()
    main(args)

