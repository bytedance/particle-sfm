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

"""Track the point trajectories and use path consistency to correct
"""
import os
import numpy as np
import argparse
from .utils import flow_check, load_flows, load_images
from .track import track
from .track_optimize import track_optimize
from .optimize.build import particlesfm

import pdb


def main_connect_point_trajectories(
    flow_dir,
    traj_dir,
    sample_ratio=2,
    flow_check_thres=1.0,
    traj_min_len=3,
    skip_path_consistency=False,
    skip_exists=False,
):
    # initiate traj_dir
    os.makedirs(traj_dir, exist_ok=True)
    output_npy_fname = os.path.join(traj_dir, "track.npy")
    if skip_exists:
        if os.path.exists(output_npy_fname):
            return

    # load data
    flow_f_dir = os.path.join(flow_dir, "flow_f")
    flow_b_dir = os.path.join(flow_dir, "flow_b")
    flows_f, flows_b = load_flows(flow_f_dir), load_flows(flow_b_dir)
    error_maps, occ_maps = flow_check(flows_f, flows_b, thres=flow_check_thres)
    n_images = len(flows_f) + 1

    # load stride-2 optical flow
    if not skip_path_consistency:
        flow_f2_dir = os.path.join(flow_dir, "flow_f2")
        flow_b2_dir = os.path.join(flow_dir, "flow_b2")
        flows_f2, flows_b2 = load_flows(flow_f2_dir), load_flows(flow_b2_dir)
        error_maps_s2, occ_maps_s2 = flow_check(
            flows_f2, flows_b2, thres=flow_check_thres
        )

    # start connecting tracks into point trajectories
    if skip_path_consistency:
        trajs = track(flows_f, occ_maps, sample_ratio)
    else:
        trajs = track_optimize(flows_f, flows_f2, occ_maps, occ_maps_s2, sample_ratio)

    # save the outputs
    dict_trajs = {}
    print(f"found {len(trajs)} trajs")
    for idx, traj in enumerate(trajs):

        if traj.length() < traj_min_len:
            continue
        dict_trajs[idx] = traj

    trajectories = particlesfm.TrajectorySet(dict_trajs)
    np.save(output_npy_fname, trajectories)


def main(args):
    main_connect_point_trajectories(
        args.flow_dir,
        args.traj_dir,
        sample_ratio=args.sample_ratio,
        flow_check_thres=args.flow_check_thres,
        traj_min_len=args.traj_min_len,
        skip_path_consistency=args.skip_path_consistency,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Connecting and optimizing point trajectories from pairwise flows"
    )
    parser.add_argument("--flow_dir", help="path to the folder of optical flows")
    parser.add_argument("--traj_dir", help="trajectory output")
    parser.add_argument(
        "--sample_ratio", type=int, default=2, help="sample ratio of trajectories"
    )
    parser.add_argument(
        "--traj_min_len", type=int, default=3, help="minimum length of the trajectories"
    )
    parser.add_argument(
        "--flow_check_thres",
        type=float,
        default=1.0,
        help="flow consistency check threshold",
    )
    parser.add_argument(
        "--skip_path_consistency",
        action="store_true",
        help="whether to skip the path consistency optimization or not",
    )
    args = parser.parse_args()
    main(args)
