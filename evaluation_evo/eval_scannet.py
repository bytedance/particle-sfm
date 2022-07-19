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
from os.path import dirname
import numpy as np
import argparse
import glob
from scipy.spatial.transform import Rotation
import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation
from evo.core.metrics import Unit
from scipy.spatial.transform import Rotation

def eval_one_seq(args, gt_dir, input_dir):
    # pre-process the colmap-format poses to tum format poses
    pose_names = sorted(glob.glob(input_dir + "/*.txt"))
    poses, tum_poses = [], []
    for name in pose_names:
        time = int(os.path.basename(name).split(".")[0].split("_")[-1])
        pose = np.loadtxt(name)
        # world2cam -> cam2world
        if pose.shape[0] == 3:
            pose = np.concatenate([pose, np.array([[0,0,0,1]])], 0)
        cam2world = np.linalg.inv(pose)
        R, t = cam2world[:3,:3], cam2world[:3,3]
        rot = Rotation.from_matrix(R)
        quad = rot.as_quat() # xyzw
        quad_wxyz = np.array([quad[3], quad[0], quad[1], quad[2]])
        pose_t = np.concatenate([[time], t, quad_wxyz], 0) # [time, tx, ty, tz, qw, qx, qy, qz]
        tum_pose_t = np.concatenate([[time], t, quad], 0)
        poses.append(pose_t)
        tum_poses.append(tum_pose_t)
    poses = np.stack(poses, 0)
    tum_poses = np.stack(tum_poses, 0)
    np.savetxt("./Ours.txt", tum_poses)

    traj_est = PoseTrajectory3D(
            positions_xyz=poses[:,1:4],
            orientations_quat_wxyz=poses[:,4:],
            timestamps=poses[:,0:1])

    # pre-process the colmap-format poses to tum format poses
    gt_pose_lists = sorted(glob.glob(os.path.join(gt_dir, '*.txt')))
    tstamps = [float(x.split('/')[-1][:-4].split("_")[-1]) for x in gt_pose_lists]
    gt_poses = [np.loadtxt(f) for f in gt_pose_lists]
    xyzs, wxyzs = [], []
    tum_gt_poses = []
    for gt_pose in gt_poses:
        xyz = gt_pose[:3,-1]
        xyzs.append(xyz)
        R = Rotation.from_matrix(gt_pose[:3,:3])
        xyzw = R.as_quat() # scalar-last for scipy
        wxyz = np.array([xyzw[-1], xyzw[0], xyzw[1], xyzw[2]])
        wxyzs.append(wxyz)
        tum_gt_pose = np.concatenate([xyz, xyzw], 0)
        tum_gt_poses.append(tum_gt_pose)
    tum_gt_poses = np.stack(tum_gt_poses, 0)
    tum_gt_poses[:,:3] = tum_gt_poses[:,:3] - np.mean(tum_gt_poses[:,:3], 0, keepdims=True)
    tt = np.expand_dims(np.stack(tstamps, 0), -1)
    tum_gt_poses = np.concatenate([tt, tum_gt_poses], -1)
    np.savetxt("Groundtruth.txt", tum_gt_poses)
    # plot the trajectory
    save_name = os.path.join(os.path.dirname(os.path.dirname(input_dir)), "plot.png")
    command = "evo_traj tum ./Ours.txt --ref=./Groundtruth.txt -p --plot_mode=xyz -a -s --save_plot " + save_name + \
        " --no_warnings"
    os.system(command)

    traj_ref = PoseTrajectory3D(
        positions_xyz=np.stack(xyzs, 0),
        orientations_quat_wxyz=np.stack(wxyzs, 0),
        timestamps=np.array(tstamps)
    )
    if len(poses) < 0.8 * len(tstamps):
        return None, None, None
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    try:
        result = main_ape.ape(traj_ref, traj_est, est_name='traj',
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

        print("Seq " + seq)
        print(result)
        ate = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_est, est_name='traj',
            pose_relation=PoseRelation.rotation_angle_deg, align=True, correct_scale=True,
            delta=1.0, delta_unit=Unit.frames, rel_delta_tol=0.1)

        print("Seq " + seq)
        print(result)
        rpe_rot = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_est, est_name='traj',
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True,
            delta=1.0, delta_unit=Unit.frames, rel_delta_tol=0.1)

        print("Seq " + seq)
        print(result)
        rpe_trans = result.stats['rmse']

    except:
        print("Seq " + seq + " not valid")
        ate = None
        rpe_trans, rpe_rot = None, None
    return ate, rpe_trans, rpe_rot


if __name__ == "__main__":
    # evaluate the single sequence with groundtruth pose
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help='input dir of colmap poses')
    parser.add_argument("--gt_dir", type=str, help='dir of groundtruth poses')
    parser.add_argument("--output_dir", type=str, default="./", help='output dir of tum format pose txt')
    parser.add_argument("--error_summary_dir", type=str, default="./", help='output error summary')
    parser.add_argument("--dataset", help='input dataset')
    args = parser.parse_args()
    if args.dataset != 'scannet':
        raise NotImplementedError
    # loop over the whole dataset
    seqs = sorted(os.listdir(args.input_dir))
    ates = []
    result_summary = open(os.path.join(args.error_summary_dir, "errors_ate.txt"), "w")
    result_summary.write("ATE (m), RPE trans (m), RPE Rot (deg)\n")
    for seq in seqs:
        if not os.path.isdir(os.path.join(args.input_dir, seq)):
            continue
        input_seq = os.path.join(args.input_dir, seq, "colmap_outputs_converted/poses/")
        gt_seq = os.path.join(args.gt_dir, seq)
        output_seq = os.path.join(args.output_dir, seq, "colmap_outputs_converted")
        if not os.path.isdir(input_seq):
            result_summary.write(seq + " None" + "\n")
            continue

        # evaluate the ate and rpe with respect to gt
        ate, rpe_trans, rpe_rot = eval_one_seq(args, gt_seq, input_seq)
        if ate is not None:
            result_summary.write(seq + " {:.06f} {:.06f} {:.06f}\n".format(ate, rpe_trans, rpe_rot))
        else:
            result_summary.write(seq + " None" + "\n")
    result_summary.close()
