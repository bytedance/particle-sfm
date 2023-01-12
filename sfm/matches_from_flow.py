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
from tqdm import tqdm


class imageMatchData:
    def __init__(self, image_id):
        self.image_id = image_id
        self.keypoints = []
        self.match_pairs = {}

    def insert_keypoint(self, kp1):
        # kp1: [x,y]
        self.keypoints.append(kp1)

    def insert_match(self, tgt_img_id, kp_ind1, kp_ind2):
        # kp_ind1: the keypoint index in current image
        # kp_ind2: the keypoint index in target matched image (refered by tgt_img_id)
        match_key = str(self.image_id) + "-" + str(tgt_img_id)
        if match_key in self.match_pairs.keys():
            self.match_pairs[match_key].append([kp_ind1, kp_ind2])
        else:
            self.match_pairs[match_key] = [[kp_ind1, kp_ind2]]

    def rename_matches(self, image_names):
        # rename the matching pairs, replace the sorted id to image names
        old_keys = np.copy(list(self.match_pairs.keys()))
        for key in old_keys:
            self_id, tgt_img_id = key.split("-")
            self_id, tgt_img_id = int(self_id), int(tgt_img_id)
            assert self_id == self.image_id
            new_key = image_names[self_id] + "-" + image_names[tgt_img_id]
            self.match_pairs[new_key] = self.match_pairs.pop(key)


def traj_to_matches(img_dir, traj_dir, match_list_file, remove_dynamic=True):
    # some hyper-parameters
    sample_k = 20

    # load trajectories
    trajectories = np.load(
        os.path.join(traj_dir, "track.npy"), allow_pickle=True
    ).item()
    if type(trajectories) != dict:
        trajectories = trajectories.as_dict()

    # initialize each image
    image_names = sorted(os.listdir(img_dir))
    image_datas = []
    for i in range(len(image_names)):
        image_datas.append(imageMatchData(image_id=i))

    # loop through each trajectory
    for key in tqdm(trajectories):
        traj = trajectories[key]
        locations, labels, frame_ids = (
            np.array(traj["locations"]),
            np.array(traj["labels"]),
            np.array(traj["frame_ids"]),
        )
        traj_len = locations.shape[0]
        assert traj_len == labels.shape[0]
        assert traj_len == frame_ids.shape[0]
        # loop through the trajectory to get all the keypoint ids in each image
        img_ids, kp_inds = [], []
        for j in range(traj_len):
            if remove_dynamic:
                # remove dynamic keypoints
                if labels[j] == 1:
                    continue
            mx, my = locations[j]
            img_id = frame_ids[j]
            mx, my, img_id = float(mx), float(my), int(img_id)
            image_datas[img_id].insert_keypoint([mx, my])
            ind = len(image_datas[img_id].keypoints) - 1
            kp_inds.append(ind)
            img_ids.append(img_id)
        # loop through the trajectory images to sample matches (N*K)
        for j in range(len(img_ids)):
            if len(img_ids) <= sample_k:
                # then insert every other image as matching pairs
                for k in range(len(img_ids)):
                    if k == j:
                        continue
                    image_datas[img_ids[j]].insert_match(
                        img_ids[k], kp_inds[j], kp_inds[k]
                    )
            else:
                # then uniformly sample K images among this traj
                stride = len(img_ids) // sample_k
                for k in range(sample_k):
                    tgt_img_traj_ind = k * stride
                    if tgt_img_traj_ind == j:
                        continue
                    image_datas[img_ids[j]].insert_match(
                        img_ids[tgt_img_traj_ind], kp_inds[j], kp_inds[tgt_img_traj_ind]
                    )

    # Read the images and project the trajectory-based image-id to image name
    # The colmap does not necessarily follow the sorted image name as ids
    colmap_datas = {}
    for i, img_name in enumerate(image_names):
        image_datas[i].rename_matches(image_names)
        colmap_datas[img_name] = image_datas[i]

    # write the pair txt file and filter out the very few matched pairs
    pair_txt = open(match_list_file, "w")
    for img_name, data in colmap_datas.items():
        for m in data.match_pairs.keys():
            name0, name1 = m.split("-")
            pair_txt.write(name0 + " " + name1 + "\n")
    pair_txt.close()
    return colmap_datas
