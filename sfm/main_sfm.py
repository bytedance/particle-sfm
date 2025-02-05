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

import argparse
import subprocess
import multiprocessing
import logging
from pathlib import Path
import shutil
import pprint
import time

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from colmap_utils.read_write_model import read_cameras_binary
from import_feature_matches import *

def build_database(sfm_dir, image_dir, traj_dir, colmap_path="colmap", single_camera=True, remove_dynamic=True, skip_geometric_verification=False, skip_exists=False):
    sfm_dir, image_dir = Path(sfm_dir), Path(image_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    database_path = sfm_dir / 'database.db'
    pair_txt_path = sfm_dir / "image_match_pairs.txt"
    if skip_exists:
        if os.path.exists(database_path) and os.path.exists(pair_txt_path):
            return database_path, pair_txt_path
    if os.path.exists(database_path):
        os.remove(database_path)
    if os.path.exists(pair_txt_path):
        os.remove(pair_txt_path)

    create_empty_db(database_path)
    import_images(colmap_path, sfm_dir, image_dir, database_path, single_camera)
    image_ids = get_image_ids(database_path)
    import_keypoints_matches(image_ids, image_dir, database_path, pair_txt_path, traj_dir, skip_geometric_verification, remove_dynamic=remove_dynamic)
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database_path, pair_txt_path)
    return database_path, pair_txt_path

def compute_model_stats(model_path, colmap_path="colmap"):
    models = list(model_path.iterdir())
    if len(models) == 0:
        logging.error('Could not reconstruct any model!')
        return None

    largest_model = None
    largest_model_num_images = 0
    for model in models:
        if not os.path.exists(model / "cameras.bin"):
            continue
        num_images = len(read_cameras_binary(str(model / 'cameras.bin')))
        if num_images > largest_model_num_images:
            largest_model = model
            largest_model_num_images = num_images
    if largest_model_num_images == 0:
        print("Error! No model found in {0}.".format(model_path))
        return None
    logging.info(f'Largest model is #{largest_model.name} '
                 f'with {largest_model_num_images} images.')

    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer',
         '--path', str(largest_model)])
    stats_raw = stats_raw.decode().split("\n")
    stats = dict()
    for stat in stats_raw:
        if stat.startswith("Registered images"):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith("Points"):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])
    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        shutil.copy(str(largest_model / filename), str(model_path))
    return stats

def main_incremental_sfm(sfm_dir, image_dir, traj_dir, colmap_path="colmap", single_camera=True, remove_dynamic=True, skip_geometric_verification=False, min_num_matches=None, skip_exists=False):
    """
    Incremental structure-from-motion with COLMAP
    """
    database_path, pair_txt_path = build_database(sfm_dir, image_dir, traj_dir, colmap_path=colmap_path, single_camera=single_camera, remove_dynamic=remove_dynamic, skip_geometric_verification=skip_geometric_verification, skip_exists=skip_exists)
    model_path = Path(sfm_dir) / 'model'
    model_path.mkdir(exist_ok=True, parents=True)

    # run colmap incremental mapper
    cmd = [
    str(colmap_path), 'mapper',
    '--database_path', str(database_path),
    '--image_path', str(image_dir),
    '--output_path', str(model_path),
    '--Mapper.num_threads', str(min(multiprocessing.cpu_count(), 64)),
    '--random_seed', str(100)]
    if min_num_matches:
        cmd += ['--Mapper.min_num_matches', str(min_num_matches)]
    cmd += ['--Mapper.multiple_models', str(0),
        '--Mapper.ba_refine_principal_point', str(0),
        '--Mapper.ba_refine_extra_params', str(0)]
    print(' '.join(cmd))
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    print('incremental sfm: {}s'.format(end - start))

    # analyze model
    stats = compute_model_stats(model_path)
    if stats is not None:
        print(stats)

def main_global_sfm(sfm_dir, image_dir, traj_dir, gcolmap_path=None, colmap_path="colmap", single_camera=True, remove_dynamic=True, skip_geometric_verification=False, min_num_matches=None, skip_exists=False):
    """
    Global structure-from-motion for videos
    """
    if gcolmap_path is None:
        curpath = os.path.dirname(os.path.abspath(__file__))
        gcolmap_path = os.path.join(curpath, "gmapper/build/src/exe/gcolmap")
    database_path, pair_txt_path = build_database(sfm_dir, image_dir, traj_dir, colmap_path=colmap_path, single_camera=single_camera, remove_dynamic=remove_dynamic, skip_geometric_verification=skip_geometric_verification, skip_exists=skip_exists)
    model_path = Path(sfm_dir) / 'model'
    model_path.mkdir(exist_ok=True, parents=True)

    # run global structure-from-motion
    cmd = [
    str(gcolmap_path), 'global_mapper',
    '--database_path', str(database_path),
    '--image_path', str(image_dir),
    '--output_path', str(model_path),
    '--GlobalMapper.num_threads', str(min(multiprocessing.cpu_count(), 64)),
    '--random_seed', str(100)]
    if min_num_matches:
        cmd += ['--GlobalMapper.min_num_matches', str(min_num_matches)]
    cmd += ['--GlobalMapper.ba_refine_principal_point', str(0),
        '--GlobalMapper.ba_refine_extra_params', str(0)]
    print(' '.join(cmd))
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    print('global sfm: {}s'.format(end - start))

    # analyze model
    stats = compute_model_stats(model_path)
    if stats is not None:
        print(stats)

def main_global_sfm_glomap(sfm_dir, image_dir, traj_dir, gcolmap_path=None, glomap_path="glomap", colmap_path="colmap", single_camera=True, remove_dynamic=True, skip_geometric_verification=False, min_num_matches=None, skip_exists=False):
    """
    Global structure-from-motion for videos using GLOMAP
    """
    database_path, pair_txt_path = build_database(sfm_dir, image_dir, traj_dir, colmap_path=colmap_path, single_camera=single_camera, remove_dynamic=remove_dynamic, skip_geometric_verification=skip_geometric_verification, skip_exists=skip_exists)
    model_path = Path(sfm_dir) / 'model'
    model_path.mkdir(exist_ok=True, parents=True)

    # run global structure-from-motion
    cmd = [
        str(glomap_path), 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--output_path', str(model_path)]
    print(' '.join(cmd))
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    print('global sfm with GLOMAP: {}s'.format(end - start))

    # analyze model
    stats = compute_model_stats(model_path)
    if stats is not None:
        print(stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--traj_dir', type=Path, required=True)
    parser.add_argument('--colmap_path', type=Path, default='colmap')
    parser.add_argument('--gcolmap_path', type=Path, default='./gmapper/build/src/exe/gcolmap')

    parser.add_argument('--incremental_sfm', action='store_true')
    parser.add_argument('--remove_dynamic', type=bool, default=True)
    parser.add_argument('--skip_geometric_verification', action='store_true')
    args = parser.parse_args()

    if not args.incremental_sfm:
        # use global SfM by default
        main_global_sfm(args.sfm_dir, args.image_dir, args.traj_dir, remove_dynamic=remove_dynamic, skip_geometric_verification=skip_geometric_verification)
    else:
        main_incremental_sfm(args.sfm_dir, args.image_dir, args.traj_dir, remove_dynamic=remove_dynamic, skip_geometric_verification=skip_geometric_verification)


