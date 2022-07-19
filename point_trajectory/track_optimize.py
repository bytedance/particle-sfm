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

import torch
import numpy as np
from tqdm import tqdm

from .trajectory import Trajectory, IncrementalTrajectorySet
from .trajectory import grid_sample, motion_boundary, step_forward

def track_optimize(flows, flows_f2, occ_maps, occ_maps_s2, sample_ratio):
    """
    Sequentially track and optimize point trajectories
    """
    n_flows = len(flows)
    h, w = flows[0].shape[:2]
    trajs = IncrementalTrajectorySet(n_flows + 1, h, w, sample_ratio, buffer_size=3)
    for frame_id in tqdm(range(n_flows)):
        # Generate new trajectories if needed
        points = trajs.sample_candidates
        times = (np.ones(points.shape[0]) * frame_id).astype(int)
        trajs.new_traj_all(times, points)

        # Propagate all the trajectories by flow
        cur_xys = trajs.get_cur_pos()
        flow_t = torch.from_numpy(flows[frame_id]).permute(2,0,1).float()
        flow_sample = grid_sample(flow_t, cur_xys)

        # mask
        mb_mask = motion_boundary(flows[frame_id])

        # step
        next_xys, valid_flags = step_forward(cur_xys, flow_sample, occ_maps[frame_id], mb_mask)
        trajs.extend_all(next_xys, frame_id+1, valid_flags)
        # optimize
        if frame_id + 1 >= 2: # buffer should be full
            trajs.optimize_buffer(flows[frame_id-1], flows[frame_id], flows_f2[frame_id-1], occ_maps_s2[frame_id-1], frame_id+1)
    # finish
    trajs.clear_active()
    return trajs.full_trajs

