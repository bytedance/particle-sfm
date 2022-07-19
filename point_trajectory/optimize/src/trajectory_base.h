// ParticleSfM
// Copyright (C) 2022  ByteDance Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef OPTIMIZE_TRAJECTORY_BASE_H_
#define OPTIMIZE_TRAJECTORY_BASE_H_

#include <Eigen/Core>
#include <vector>
#include <deque>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace particlesfm {

using V2D = Eigen::Vector2d;

class Trajectory {
public:
    Trajectory() {}
    Trajectory(int time, V2D xy, int buffer_size = 0);
    Trajectory(double time, V2D xy, int buffer_size = 0): Trajectory(int(time), xy, buffer_size) {};
    Trajectory(const std::vector<int>& times, const std::vector<V2D>& xys, const std::vector<bool>& labels=std::vector<bool>());
    Trajectory(py::dict dict);
    py::dict as_dict() const;

    std::vector<int> times;
    std::vector<bool> labels;
    std::vector<V2D> xys;
    std::deque<V2D> buffer_xys;

    void extend(int time, V2D xy);
    void clear_buffer();
    void set_buffer_xy(int index, V2D xy);
    void set_label(int index, bool label) { labels[index] = label; }
    void set_labels(const std::vector<bool>& input_labels) { labels = input_labels; }

    int length() const;
    V2D get_tail_location() const;

private:
    int buffer_size = 0;
};

class TrajectorySet {
public:
    TrajectorySet() {};
    TrajectorySet(const std::map<int, Trajectory>& trajs_): trajs(trajs_) {}
    TrajectorySet(std::map<int, py::dict> dict);
    std::map<int, py::dict> as_dict() const;

    std::map<int, Trajectory> trajs;

    void insert(int traj_id, Trajectory traj);
    void build_invert_indexes();
    py::dict sample_inside_window(const std::vector<int>& frame_ids, int min_length = 3, int max_num_tracks = 100000) const;

private:
    std::map<int, std::map<int, size_t>> invert_maps; // key: frame_id, value: set of trajectory ids and the corresponding indexes of frame_id
};

}  // namespace particlesfm 

#endif

