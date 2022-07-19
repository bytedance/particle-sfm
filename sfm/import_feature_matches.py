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

"""Modified from https://github.com/cvg/Hierarchical-Localization.
"""
import shutil
import subprocess
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from colmap_utils.database import COLMAPDatabase
from matches_from_flow import traj_to_matches

def create_empty_db(database_path):
    print('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(colmap_path, sfm_dir, image_dir, database_path,
                  single_camera=False):
    print('Importing images into the database...')
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')

    # We need to create dummy features for COLMAP to import images with EXIF
    dummy_dir = sfm_dir / 'dummy_features'
    dummy_dir.mkdir()
    for i in images:
        with open(str(dummy_dir / (i.name + '.txt')), 'w') as f:
            f.write('0 128')

    cmd = [
        str(colmap_path), 'feature_importer',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--import_path', str(dummy_dir),
        '--ImageReader.single_camera',
        str(int(single_camera)),
        '--ImageReader.camera_model', 'SIMPLE_PINHOLE']
    subprocess.run(cmd, check=True)

    db = COLMAPDatabase.connect(database_path)
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.commit()
    db.close()
    shutil.rmtree(str(dummy_dir))


def get_image_ids(database_path):
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images

def import_keypoints_matches(image_ids, image_dir, database_path, match_list_file, \
        traj_dir, skip_geometric_verification=False, remove_dynamic=True):
    colmap_feat_match_data = traj_to_matches(image_dir, traj_dir, match_list_file, remove_dynamic=remove_dynamic)
    print('Importing keypoints and matches into the database...')
    db = COLMAPDatabase.connect(database_path)
    for image_name, image_id in image_ids.items():
        keypoints = np.array(colmap_feat_match_data[image_name].keypoints)
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)
    db.commit()
    print('Importing matches into the database...')

    matched = set()
    for image_name, image_id in image_ids.items():
        matches = colmap_feat_match_data[image_name].match_pairs
        for pair, match in matches.items():
            # get the image name and then id
            name0, name1 = pair.split('-')
            id0, id1 = image_ids[name0], image_ids[name1]
            if len({(id0, id1), (id1, id0)} & matched) > 0:
                continue
            match = np.array(match)
            db.add_matches(id0, id1, match)
            matched |= {(id0, id1), (id1, id0)}
            if skip_geometric_verification:
                db.add_two_view_geometry(id0, id1, match)
    db.commit()
    db.close()
    print('Imported.')

def geometric_verification(colmap_path, database_path, pairs_path):
    print('Performing geometric verification of the matches...')
    cmd = [
        str(colmap_path), 'matches_importer',
        '--database_path', str(database_path),
        '--match_list_path', str(pairs_path),
        '--match_type', 'pairs',
        '--SiftMatching.max_num_trials', str(20000),
        '--SiftMatching.min_inlier_ratio', str(0.1),
        '--SiftMatching.min_num_inlier', str(15),
        '--SiftMatching.max_error', str(4)]
    subprocess.run(cmd, check=True)
