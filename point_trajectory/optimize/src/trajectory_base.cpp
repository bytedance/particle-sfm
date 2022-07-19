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

#include "trajectory_base.h"

namespace particlesfm {

Trajectory::Trajectory(int time, V2D xy, int buffer_size_) {
    buffer_size = buffer_size_;
    extend(time, xy);
}


Trajectory::Trajectory(const std::vector<int>& times_, const std::vector<V2D>& xys_, const std::vector<bool>& labels_) {
    times = times_;
    xys = xys_;
    if (labels_.empty()) {
        labels.resize(times.size());
        std::fill(labels.begin(), labels.end(), false);
    }
    else
        labels = labels_;
}

Trajectory::Trajectory(py::dict dict) {
    if (dict.contains("frame_ids"))
        times = dict["frame_ids"].cast<std::vector<int>>();
    if (dict.contains("locations"))
        xys = dict["locations"].cast<std::vector<V2D>>();
    if (dict.contains("labels"))
        labels = dict["labels"].cast<std::vector<bool>>();
}

py::dict Trajectory::as_dict() const{
    py::dict output;
    output["frame_ids"] = times;
    output["locations"] = xys;
    output["labels"] = labels;
    return output;
}

void Trajectory::extend(int time, V2D xy) {
    times.push_back(time);
    labels.push_back(false);
    if (buffer_size == 0) {
        xys.push_back(xy);
        return;
    }
    buffer_xys.push_back(xy);
    if (buffer_xys.size() > buffer_size) {
        xys.push_back(buffer_xys[0]);
        buffer_xys.pop_front();
    }
}

void Trajectory::clear_buffer() {
    for (auto it = buffer_xys.begin(); it != buffer_xys.end(); ++it) {
        xys.push_back(*it);
    }
    buffer_xys.clear();
}

void Trajectory::set_buffer_xy(int index, V2D xy) {
    if (index >= buffer_xys.size())
        throw std::runtime_error("Error! Index out of bound for the buffer.");
    buffer_xys[index] = xy;
}

int Trajectory::length() const {
    return int(xys.size()) + int(buffer_xys.size());
}

V2D Trajectory::get_tail_location() const {
    if (length() == 0)
        throw std::runtime_error("Error! The trajectory is empty!");
    if (buffer_xys.empty())
        return xys.back();
    else
        return buffer_xys.back();
}

std::map<int, py::dict> TrajectorySet::as_dict() const {
    std::map<int, py::dict> output;
    for (auto it = trajs.begin(); it != trajs.end(); ++it) {
        output[it->first] = it->second.as_dict();
    }
    return output;
}

TrajectorySet::TrajectorySet(std::map<int, py::dict> input) {
    for (auto it = input.begin(); it != input.end(); ++it) {
        trajs.insert(std::make_pair(it->first, Trajectory(it->second)));
    }
}

void TrajectorySet::insert(int traj_id, Trajectory traj) {
    if (trajs.find(traj_id) != trajs.end())
        throw std::runtime_error("Error! The trajectory id already exists!");
    trajs.insert(std::make_pair(traj_id, traj));
}

void TrajectorySet::build_invert_indexes() {
    for (auto it = trajs.begin(); it != trajs.end(); ++it) {
        auto traj = it->second;
        for (size_t i = 0; i < traj.length(); ++i) {
            int frame_id = traj.times[i];
            if (invert_maps.find(frame_id) == invert_maps.end())
                invert_maps.insert(std::make_pair(frame_id, std::map<int, size_t>()));
            invert_maps[frame_id].insert(std::make_pair(it->first, i));
        }
    }
}

py::dict TrajectorySet::sample_inside_window(const std::vector<int>& frame_ids, int min_length, int max_num_tracks) const {
    // collect all the valid trajectories
    if (invert_maps.empty())
        throw std::runtime_error("Error! The inverted index maps have not been built!");
    std::map<int, int> m_traj_ids_counter; // traj counter
    for (size_t i = 0; i < frame_ids.size(); ++i) {
        int frame_id = frame_ids[i];
        if (invert_maps.find(frame_id) == invert_maps.end()) 
            continue;
        for (auto it = invert_maps.at(frame_id).begin(); it != invert_maps.at(frame_id).end(); ++it) {
            int traj_id = it->first;
            if (m_traj_ids_counter.find(traj_id) == m_traj_ids_counter.end())
                m_traj_ids_counter.insert(std::make_pair(traj_id, 0));
            m_traj_ids_counter[traj_id]++;
        }
    }
    std::vector<int> traj_ids;
    for (auto it = m_traj_ids_counter.begin(); it != m_traj_ids_counter.end(); ++it) {
        if (it->second < min_length)
            continue;
        traj_ids.push_back(it->first);
    }

    // shrink if needed
    if (traj_ids.size() > max_num_tracks) {
        std::random_shuffle(traj_ids.begin(), traj_ids.end());
        traj_ids.resize(max_num_tracks);
    }


    // build padded arrays and masks
    int K = traj_ids.size();
    int L = frame_ids.size();
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> locations = std::make_pair(Eigen::MatrixXd::Zero(K, L), Eigen::MatrixXd::Zero(K, L));
    Eigen::MatrixXi masks(K, L);
    for (size_t i = 0; i < K; ++i) {
        int traj_id = traj_ids[i];
        for (size_t j = 0; j < L; ++j) {
            int frame_id = frame_ids[j];
            bool mask = true;
            if (invert_maps.find(frame_id) == invert_maps.end())
                mask = false;
            else
                mask = (invert_maps.at(frame_id).find(traj_id) != invert_maps.at(frame_id).end());
            masks(i, j) = mask;
            if (mask) {
                int index = invert_maps.at(frame_id).at(traj_id); 
                locations.first(i, j) = trajs.at(traj_id).xys[index][0];
                locations.second(i, j) = trajs.at(traj_id).xys[index][1];
            }
        }
    }

    py::dict output;
    output["locations"] = locations;
    output["masks"] = masks;
    output["traj_ids"] = traj_ids;
    return output;
}

} // namespace particlesfm

