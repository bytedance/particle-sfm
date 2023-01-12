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

"""Inference over raw trajectory data (coming from our point traj module) to predict per-trajectory motion label
"""
import os
import sys
import numpy as np
import cv2
import argparse
import torch
import torchvision
from tqdm.contrib import tzip

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.utils.utils import load_config_file, draw_traj_cls
from core.network.traj_oa_depth import traj_oa_depth
from .load_cut_seq import load_cut_seq


def main_motion_segmentation(
    image_dir,
    depth_dir,
    traj_dir,
    output_traj_dir,
    config_file=None,
    window_size=10,
    traj_max_num=100000,
    skip_exists=False,
):
    # TODO: actually images are only used for visualization in this function
    video_name = os.path.join(output_traj_dir, "motion_seg.mp4")
    if skip_exists and os.path.exists(video_name) and os.path.exists(output_traj_dir):
        return
    if not os.path.exists(output_traj_dir):
        os.makedirs(output_traj_dir)
    if config_file is None:
        config_file = "configs/example_test.yaml"
        curpath = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(curpath, config_file)

    cfg = load_config_file(config_file)
    if cfg.model_name == "traj_oa_depth":
        model = traj_oa_depth(window_size, cfg.resolution).cuda()
    else:
        raise NotImplementedError
    if cfg.resume_path:
        curpath = os.path.dirname(os.path.abspath(__file__))
        resume_path = os.path.join(curpath, cfg.resume_path)
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Load model from {}".format(resume_path))
    else:
        raise NotImplementedError

    model.eval()
    # load and cut data
    transforms = torchvision.transforms.ToTensor()
    (
        img_batch,
        depth_batch,
        raw_traj_batch,
        traj_batch,
        mask_batch,
        time_idx_batch,
        sample_idx_batch,
    ) = load_cut_seq(
        image_dir, depth_dir, traj_dir, window_size, cfg.resolution, traj_max_num
    )
    # model forward
    vis_imgs = []
    all_trajs, traj_labels, traj_times = {}, {}, {}
    with torch.no_grad():
        # loop over all batchs
        for imgs, depths, raw_traj, traj, mask, time_idx, sample_idx in tzip(
            img_batch,
            depth_batch,
            raw_traj_batch,
            traj_batch,
            mask_batch,
            time_idx_batch,
            sample_idx_batch,
        ):
            depths_t = []
            for depth in depths:
                depth_t = transforms(depth).float().cuda()
                depths_t.append(depth_t)
            depths_t = torch.stack(depths_t, -1).unsqueeze(0).float().cuda()
            traj_t = transforms(traj).unsqueeze(0).float().cuda()
            mask_t = transforms(mask).unsqueeze(0).float().cuda()
            batch = {"depth": depths_t, "traj": traj_t, "mask": mask_t}
            pred = model(batch)
            pred = (pred[0, 0] > 0.5).detach().cpu().numpy()
            # draw the seg results of the current batch
            traj_denor = np.copy(traj)
            traj_denor[:, :, 0] *= cfg.resolution[1]
            traj_denor[:, :, 1] *= cfg.resolution[0]
            vis = draw_traj_cls(
                np.stack(imgs, 0),
                traj_denor,
                mask,
                pred,
                np.zeros((traj_denor.shape[0])),
            )
            print(f"vis = {vis.shape}")
            for i in range(len(imgs)):
                vis_imgs.append(
                    vis[:, i * cfg.resolution[1] : (i + 1) * cfg.resolution[1], :]
                )
            # save the sampled traj, motion seg
            pred = np.tile(np.expand_dims(pred, -1), (1, raw_traj.shape[1]))
            mask = mask[:, :, 0]
            for i in range(raw_traj.shape[0]):
                # remove padding
                valid_mask = mask[i] == 0.0
                valid_traj = raw_traj[i][valid_mask]
                valid_label = pred[i][valid_mask]
                valid_time = time_idx[valid_mask]
                # merge the same trajectory across all batchs
                key = sample_idx[i]
                if key not in all_trajs:
                    all_trajs[key] = np.copy(valid_traj)
                    traj_labels[key] = np.copy(valid_label)
                    traj_times[key] = valid_time
                else:
                    cur_time = traj_times[key]
                    non_overlap_idx = np.array(
                        [time not in cur_time for time in valid_time]
                    )
                    non_overlap_traj = valid_traj[non_overlap_idx]
                    non_overlap_label = valid_label[non_overlap_idx]
                    non_overlap_time = valid_time[non_overlap_idx]
                    all_trajs[key] = np.concatenate(
                        [all_trajs[key], non_overlap_traj], 0
                    )
                    traj_labels[key] = np.concatenate(
                        [traj_labels[key], non_overlap_label], 0
                    )
                    traj_times[key] = np.concatenate(
                        [traj_times[key], non_overlap_time], 0
                    )

    # draw the full prediction results
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        video_name, fourcc, 5.0, (vis_imgs[0].shape[1], vis_imgs[0].shape[0])
    )
    print(f"vis (final) = {len(vis_imgs)}")
    for vis_img in vis_imgs:
        video_writer.write(vis_img.astype(np.uint8))
    video_writer.release()

    # save the new traj, label, and time
    trajectories = {}
    for traj_id in all_trajs:
        traj = {}
        traj["locations"] = all_trajs[traj_id]
        traj["labels"] = traj_labels[traj_id]
        traj["frame_ids"] = traj_times[traj_id]
        trajectories[traj_id] = traj
    np.save(os.path.join(output_traj_dir, "track.npy"), trajectories)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="path to the image folder")
    parser.add_argument("--depth_dir", help="path to depth folder")
    parser.add_argument("--traj_dir", help="path to the input trajectory")
    parser.add_argument(
        "--output_traj_dir",
        type=str,
        default="none",
        help="path to write output trajectory",
    )
    parser.add_argument("--config_file", help="path to the config file")
    parser.add_argument(
        "--window_size", type=int, help="sliding window size to load the data"
    )
    parser.add_argument("--traj_max_num", type=int, help="The maximum number of trajs")
    args = parser.parse_args()
    if args.output_traj_dir == "none":
        args.output_traj_dir = args.traj_dir + "_labeled"

    main_motion_segmentation(
        args.image_dir,
        args.depth_dir,
        args.traj_dir,
        args.output_traj_dir,
        window_size=args.window_size,
        traj_max_num=args.traj_max_num,
    )
