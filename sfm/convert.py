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
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from colmap_utils.read_write_model import qvec2rotmat, read_model

def gray2rgb(im, cmap):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=98, cmap='gray'):
    vinds = depth > 0
    depth = 1. / (depth + 1)
    z1 = np.percentile(depth[vinds], pc)
    z2 = np.percentile(depth[vinds], 100-pc)

    depth = (depth - z2) / (z1 - z2)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    return depth

def save_depth_pose(output_dir, cameras, images, points3D):
    # Save the sparse depth image and camera pose (world-to-cam) from colmap outputs
    depth_dir = os.path.join(output_dir, "depths")
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    pose_dir = os.path.join(output_dir, "poses")
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)
    intrinsic_dir = os.path.join(output_dir, "intrinsics")
    if not os.path.exists(intrinsic_dir):
        os.makedirs(intrinsic_dir)
    for key in tqdm(images.keys()):
        image_name = images[key].name
        # get camera intrinsics
        camera_id = images[key].camera_id
        camera = cameras[camera_id]
        h, w, params = camera.height, camera.width, camera.params
        if camera.model == 'SIMPLE_PINHOLE':
            f, cx, cy = params
        elif camera.model == 'SIMPLE_RADIAL':
            f, cx, cy, d = params
        else:
            raise NotImplementedError
        K = np.array([[f, 0, cx], [0, f, cy], [0,0,1]])
        np.savetxt(os.path.join(intrinsic_dir, os.path.splitext(image_name)[0]+'.txt'), K)
        # world-to-cam quaternion and translation
        qvec, tvec = images[key].qvec, images[key].tvec
        R, t = qvec2rotmat(qvec), np.expand_dims(tvec, -1)
        # acquire 3d points
        xys = images[key].xys
        points3D_ids = images[key].point3D_ids
        points_3d, valid_xys = [], []
        for i in range(len(points3D_ids)):
            idx = points3D_ids[i]
            if idx == -1:
                continue
            points_3d.append((points3D[idx].xyz))
            valid_xys.append(xys[i])

        points_3d = np.transpose(np.array(points_3d))
        # project onto image
        cam_points = np.matmul(R, points_3d) + t
        project_depth = np.transpose(np.matmul(K, cam_points))[:,-1]
        xy_int = np.round(np.array(valid_xys)).astype(np.int32)
        xy_int[:,0] = np.clip(xy_int[:,0], 0, w-1)
        xy_int[:,1] = np.clip(xy_int[:,1], 0, h-1)
        depth = np.zeros(shape=(h,w))
        depth[xy_int[:,1], xy_int[:,0]] = project_depth
        # save depth and pose
        np.save(os.path.join(depth_dir, os.path.splitext(image_name)[0]+'.npy'), depth)
        plt.imsave(os.path.join(depth_dir, os.path.splitext(image_name)[0]+'.png'), normalize_depth_for_display(depth, cmap='binary'))
        np.savetxt(os.path.join(pose_dir, os.path.splitext(image_name)[0]+'.txt'), np.concatenate([R, t], -1))

def write_depth_pose_from_colmap_format(input_dir, output_dir):
    model = read_model(input_dir)
    if model is None:
        return
    else:
        cameras, images, points3D = model
    save_depth_pose(output_dir, cameras, images, points3D)

def main():
    parser = argparse.ArgumentParser(description="Read and write COLMAP binary and text models")
    parser.add_argument("--input_dir", help="path to input model folder")
    parser.add_argument("--output_dir",
                        help="path to output model folder")
    args = parser.parse_args()

    model = read_model(path=args.input_dir)
    if model is None:
        return
    else:
        cameras, images, points3D = model

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))

    if args.output_dir is not None:
        print(args.output_dir)
        save_depth_pose(args.output_dir, cameras, images, points3D)

if __name__ == "__main__":
    main()

