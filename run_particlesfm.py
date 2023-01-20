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

"""Run the whole pipeline of trajectory-based video sfm from images
images -> optical flow -> point trajectories -> motion seg -> global mapper
"""
import os
import argparse
import shutil


def connect_point_trajectory(
    args, image_dir, output_dir, skip_exists=False, keep_intermediate=False
):
    # set directories in the workspace
    flow_dir = os.path.join(output_dir, "optical_flows")
    traj_dir = os.path.join(output_dir, "trajectories")

    # optical flow (RAFT)
    from third_party.RAFT import (
        compute_raft_custom_folder,
        compute_raft_custom_folder_stride2,
    )

    print("[ParticleSFM] Running pairwise optical flow inference......")
    compute_raft_custom_folder(image_dir, flow_dir, skip_exists=skip_exists)
    if not args.skip_path_consistency:
        print("[ParticleSfM] Running pairwise optical flow inference (stride 2)......")
        compute_raft_custom_folder_stride2(image_dir, flow_dir, skip_exists=skip_exists)

    # point trajectory (saved in workspace_dir / point_trajectories)
    from point_trajectory import main_connect_point_trajectories

    print(
        "[ParticleSfM] Connecting (optimization {0}) point trajectories from optical flows.......".format(
            "disabled" if args.skip_path_consistency else "enabled"
        )
    )
    main_connect_point_trajectories(
        flow_dir,
        traj_dir,
        sample_ratio=args.sample_ratio,
        flow_check_thres=args.flow_check_thres,
        skip_path_consistency=args.skip_path_consistency,
        skip_exists=skip_exists,
    )

    if not keep_intermediate:
        # remove optical flows
        shutil.rmtree(os.path.join(output_dir, "optical_flows"))
    return traj_dir


def motion_segmentation(
    args, image_dir, output_dir, traj_dir, skip_exists=False, keep_intermediate=False
):
    # set directories in the workspace
    depth_dir = os.path.join(output_dir, "midas_depth")
    labeled_traj_dir = traj_dir + "_labeled"

    # monocular depth (MiDaS)
    print("[ParticleSfM] Running per-frame monocular depth estimation........")
    from third_party.MiDaS import run_midas

    os.environ["MKL_THREADING_LAYER"] = "GNU"
    run_midas(image_dir, depth_dir, skip_exists=skip_exists)

    # point trajectory based motion segmentation
    print("[ParticleSfM] Running point trajectory based motion segmentation........")
    from motion_seg import main_motion_segmentation

    main_motion_segmentation(
        image_dir,
        depth_dir,
        traj_dir,
        labeled_traj_dir,
        window_size=args.window_size,
        traj_max_num=args.traj_max_num,
        skip_exists=skip_exists,
    )
    if os.path.isfile(os.path.join(output_dir, "motion_seg.mp4")):
        os.remove(os.path.join(output_dir, "motion_seg.mp4"))
    shutil.move(os.path.join(labeled_traj_dir, "motion_seg.mp4"), output_dir)

    if not keep_intermediate:
        # remove original point trajectories
        shutil.rmtree(depth_dir)
        shutil.rmtree(traj_dir)
    return labeled_traj_dir


def sfm_reconstruction(
    args, image_dir, output_dir, traj_dir, skip_exists=False, keep_intermediate=False
):
    # set directories in the workspace
    sfm_dir = os.path.join(output_dir, "sfm")

    # sfm reconstruction
    from sfm import (
        main_global_sfm,
        main_incremental_sfm,
        write_depth_pose_from_colmap_format,
    )

    if not args.incremental_sfm:
        print("[ParticleSfM] Running global structure-from-motion........")
        main_global_sfm(
            sfm_dir,
            image_dir,
            traj_dir,
            remove_dynamic=(not args.assume_static),
            skip_exists=skip_exists,
        )
    else:
        print(
            "[ParticleSfM] Running incremental structure-from-motion with COLMAP........"
        )
        main_incremental_sfm(
            sfm_dir,
            image_dir,
            traj_dir,
            remove_dynamic=(not args.assume_static),
            skip_exists=skip_exists,
        )

    # # write depth and pose files from COLMAP format
    write_depth_pose_from_colmap_format(
        sfm_dir, os.path.join(output_dir, "colmap_outputs_converted")
    )

    if not keep_intermediate:
        # remove labeled point trajectories
        shutil.rmtree(traj_dir)


def particlesfm(
    args,
    image_dir,
    output_dir,
    skip_exists=False,
    keep_intermediate=False,
):
    """
    Inputs:
    - img_dir: str - The folder containing input images
    - output_dir: str - The workspace directory
    """
    # if not os.path.exists(image_dir):
    #     raise ValueError(
    #         "Error! The input image directory {0} is not found.".format(image_dir)
    #     )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # connect point trajectory
    traj_dir = connect_point_trajectory(
        args,
        image_dir,
        output_dir,
        skip_exists=skip_exists,
        keep_intermediate=keep_intermediate,
    )

    """
    # motion segmentation
    if not args.assume_static:
        traj_dir = motion_segmentation(
            args,
            image_dir,
            output_dir,
            traj_dir,
            skip_exists=skip_exists,
            keep_intermediate=keep_intermediate,
        )

    # sfm reconstruction
    if not args.skip_sfm:
        sfm_reconstruction(
            args,
            image_dir,
            output_dir,
            traj_dir,
            skip_exists=skip_exists,
            keep_intermediate=keep_intermediate,
        )
    """


def parse_args():
    parser = argparse.ArgumentParser(
        "Dense point trajectory based colmap reconstruction for videos"
    )
    # point trajectory
    parser.add_argument(
        "--flow_check_thres",
        type=float,
        default=1.0,
        help="the forward-backward flow consistency check threshold",
    )
    parser.add_argument(
        "--sample_ratio",
        type=int,
        default=2,
        help="the sampling ratio for point trajectories",
    )
    parser.add_argument(
        "--traj_min_len",
        type=int,
        default=3,
        help="the minimum length for point trajectories",
    )
    # motion segmentation
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="the window size for trajectory motion segmentation",
    )
    parser.add_argument(
        "--traj_max_num",
        type=int,
        default=100000,
        help="the maximum number of trajs inside a window",
    )
    # sfm
    parser.add_argument(
        "--incremental_sfm",
        action="store_true",
        help="whether to use incremental sfm or not",
    )
    # pipeline control
    parser.add_argument(
        "--skip_path_consistency",
        action="store_true",
        help="whether to skip the path consistency optimization or not",
    )
    parser.add_argument(
        "--assume_static",
        action="store_true",
        help="whether to skip the motion segmentation or not",
    )
    parser.add_argument(
        "--skip_sfm",
        action="store_true",
        help="whether to skip structure-from-motion or not",
    )
    parser.add_argument(
        "--skip_exists", action="store_true", help="whether to skip exists"
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="whether to keep intermediate files such as flows, monocular depths, etc.",
    )

    # input by sequence directory
    # python run_particlesfm.py --image_dir ${PATH_TO_SEQ_FOLDER} --output_dir ${OUTPUT_WORKSPACE}
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        default="none",
        help="path to the sequence folder containing images",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="none", help="workspace for output"
    )

    # input by workspace
    # python run_particlesfm.py --workspace_dir ${WORKSPACE_DIR}
    parser.add_argument(
        "--workspace_dir", type=str, default="none", help="input workspace"
    )
    parser.add_argument(
        "--image_folder", type=str, default="images", help="image folder"
    )  # also used in the folder option

    # input by folder containing multiple workspaces
    # python run_particlesfm.py --root_dir ${ROOT_DIR}
    # multiple sequences should be with the structure below:
    # - ROOT_DIR
    #    - XXX (sequence 1)
    #        - images
    #            - xxxxxx.png
    #    - XXX (sequence 2)
    parser.add_argument(
        "--root_dir",
        type=str,
        default="none",
        help="path to to the folder containing workspaces",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if (
        args.image_dir != "none" and args.output_dir != "none"
    ):  # input by sequence directory
        particlesfm(
            args,
            args.image_dir,
            args.output_dir,
            skip_exists=args.skip_exists,
            keep_intermediate=args.keep_intermediate,
        )
    elif args.workspace_dir != "none":
        image_dir = os.path.join(args.workspace_dir, args.image_folder)
        particlesfm(
            args,
            image_dir,
            args.workspace_dir,
            skip_exists=args.skip_exists,
            keep_intermediate=args.keep_intermediate,
        )
    elif args.root_dir != "none":
        if not os.path.exists(args.root_dir):
            raise ValueError(
                "Error! The input folder {0} is not found.".format(args.root_dir)
            )
        seq_names = sorted(os.listdir(args.root_dir))
        print(
            "A total of {0} sequences found in {1}.".format(
                len(seq_names), args.root_dir
            )
        )
        for seq_name in seq_names:
            workspace_dir = os.path.join(args.root_dir, seq_name)
            image_dir = os.path.join(workspace_dir, args.image_folder)
            particlesfm(
                args,
                image_dir,
                workspace_dir,
                skip_exists=args.skip_exists,
                keep_intermediate=args.keep_intermediate,
            )
    else:
        raise ValueError("Error! No input provided.")
