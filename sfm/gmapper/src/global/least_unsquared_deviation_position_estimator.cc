// Copyright (C) 2015 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include "global/least_unsquared_deviation_position_estimator.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <ceres/rotation.h>

#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <colmap/base/pose.h>
#include <colmap/util/threading.h>
#include <theia/math/constrained_l1_solver.h>
#include <theia/util/map_util.h>

#include "base/reconstruction.h"
#include "global/triplet_util.h"

namespace colmap {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;

Vector3d GetRotatedTranslation(const Vector4d& rotation,
                               const Vector3d& translation) {
  return QuaternionRotatePoint(InvertQuaternion(rotation), translation);
}

}  // namespace

bool LeastUnsquaredDeviationPositionEstimator::Options::Check() const {
  CHECK_OPTION_GE(num_threads, 0);
  CHECK_OPTION_GT(max_num_iterations, 0);
  CHECK_OPTION_GT(max_num_reweighted_iterations, 0);
  CHECK_OPTION_GT(max_num_points, 0);
  CHECK_OPTION_GE(min_tri_angle, 0.0);
  return true;
}

LeastUnsquaredDeviationPositionEstimator::
    LeastUnsquaredDeviationPositionEstimator(
        const Options& options, const Reconstruction& reconstruction)
    : options_(options), reconstruction_(reconstruction) {
  CHECK(options_.Check());
}

Eigen::Vector2d LeastUnsquaredDeviationPositionEstimator::GetNormalizedFeature(const Image& view,
                                                      const point3D_t track_id) {
  const Camera& camera = reconstruction_.Camera(view.CameraId());
  const Eigen::Vector2d& feature = view.Point2D(
      point_map_.at(std::make_pair(view.ImageId(), track_id))).XY();
  return camera.ImageToWorld(feature);
}

bool LeastUnsquaredDeviationPositionEstimator::EstimatePositions(
    const std::vector<ImagePair>& view_pairs,
    const std::unordered_map<image_t, Vector4d>& orientations,
    std::unordered_map<image_t, Vector3d>* positions) {
  CHECK_NOTNULL(positions)->clear();
  view_pairs_ = &view_pairs;

  InitializeIndexMapping(view_pairs, orientations);
  const int num_views = view_id_to_index_.size();
  const int num_view_pairs = view_id_pair_to_index_.size();

  if (options_.use_scale_constraints) {
    view_pair_map_.clear();
    for (size_t i = 0; i < view_pairs.size(); ++i) {
      const auto& pair = view_pairs[i];
      CHECK_LT(pair.image_id1, pair.image_id2);
      view_pair_map_.emplace(std::make_pair(pair.image_id1, pair.image_id2), i);
    }

    point_map_.clear();
    for (const auto& point : reconstruction_.Points3D()) {
      point3D_t track_id = point.first;
      for (const auto& track : point.second.Track().Elements()) {
        point_map_.emplace(std::make_pair(track.image_id, track_id), track.point2D_idx);
      }
    }

    // Extract triplets from the view pairs. As of now, we only consider the
    // largest connected triplet in the viewing graph.
    VLOG(2) << "Extracting triplets from the viewing graph.";
    triplets_ = GetLargetConnectedTripletGraph(view_pairs);

    VLOG(2) << "Determining baseline ratios within each triplet...";
    // Baselines where (x, y, z) corresponds to the baseline of the first,
    // second, and third view pair in the triplet.
    std::unique_ptr<ThreadPool> pool(new ThreadPool(options_.num_threads));
    baselines_.resize(triplets_.size());
    weights_.resize(triplets_.size());
    for (int i = 0; i < triplets_.size(); i++) {
      pool->AddTask(&LeastUnsquaredDeviationPositionEstimator::ComputeBaselineRatioForTriplet,
                this,
                triplets_[i],
                &baselines_[i],
                &weights_[i]);
    }
    pool->Wait();
  }

  VLOG(2) << "Building the constraint matrix...";
  // Set up the linear system.
  SetupConstraintMatrix(view_pairs, orientations);
  Eigen::VectorXd solution;
  solution.setZero(constraint_matrix_.cols());

  // Create the lower bound constraint enforcing that all scales are > 1.
  Eigen::SparseMatrix<double> geq_mat(num_view_pairs,
                                      constraint_matrix_.cols());
  for (int i = 0; i < num_view_pairs; i++) {
    geq_mat.insert(i, 3 * (num_views - 1) + i) = 1.0;
  }
  Eigen::VectorXd geq_vec(num_view_pairs);
  geq_vec.setConstant(1.0);

  Eigen::VectorXd b(constraint_matrix_.rows());
  b.setZero();

  VLOG(2) << "Solving for positions...";
  // Solve for camera positions by solving a constrained L1 problem to enforce
  // all relative translations scales > 1.
  theia::ConstrainedL1Solver::Options l1_options;
  theia::ConstrainedL1Solver solver(
      l1_options, constraint_matrix_, b, geq_mat, geq_vec);
  solver.Solve(&solution);

  // Set the estimated positions.
  for (const auto& view_id_index : view_id_to_index_) {
    const int index = view_id_index.second;
    const image_t view_id = view_id_index.first;
    if (index == kConstantViewIndex) {
      (*positions)[view_id] = Eigen::Vector3d::Zero();
    } else {
      (*positions)[view_id] = solution.segment<3>(index);
    }
  }

  return true;
}

void LeastUnsquaredDeviationPositionEstimator::InitializeIndexMapping(
    const std::vector<ImagePair>& view_pairs,
    const std::unordered_map<image_t, Vector4d>& orientations) {
  std::unordered_set<image_t> views;
  for (const auto& view_pair : view_pairs) {
    if (theia::ContainsKey(orientations, view_pair.image_id1) &&
        theia::ContainsKey(orientations, view_pair.image_id2)) {
      views.insert(view_pair.image_id1);
      views.insert(view_pair.image_id2);
    }
  }

  // Create a mapping from the view id to the index of the linear system.
  int index = kConstantViewIndex;
  view_id_to_index_.reserve(views.size());
  for (const image_t view_id : views) {
    view_id_to_index_[view_id] = index;
    index += 3;
  }

  // Create a mapping from the view id pair to the index of the linear system.
  view_id_pair_to_index_.reserve(view_pairs.size());
  for (const auto& view_pair : view_pairs) {
    if (theia::ContainsKey(view_id_to_index_, view_pair.image_id1) &&
        theia::ContainsKey(view_id_to_index_, view_pair.image_id2)) {
      auto pair = std::make_pair(view_pair.image_id1, view_pair.image_id2);
      CHECK(!theia::ContainsKey(view_id_pair_to_index_, pair));
      view_id_pair_to_index_[std::make_pair(view_pair.image_id1, view_pair.image_id2)] = index;
      ++index;
    }
  }
}

void LeastUnsquaredDeviationPositionEstimator::ComputeBaselineRatioForTriplet(
    const ImageIdTriplet& triplet, Vector3d* baseline, double* weight) {
  baseline->setZero();
  *weight = 0;

  const Image& view1 = reconstruction_.Image(std::get<0>(triplet));
  const Image& view2 = reconstruction_.Image(std::get<1>(triplet));
  const Image& view3 = reconstruction_.Image(std::get<2>(triplet));

  // Find common tracks.
  const std::vector<image_t> triplet_view_ids = {
      std::get<0>(triplet), std::get<1>(triplet), std::get<2>(triplet)};
  std::vector<point3D_t> common_tracks = FindCommonTracksInViews(reconstruction_, triplet_view_ids);

  // Normalize all features.
  std::vector<Eigen::Vector2d> feature1, feature2, feature3;
  feature1.reserve(common_tracks.size());
  feature2.reserve(common_tracks.size());
  feature3.reserve(common_tracks.size());
  for (const auto& track_id : common_tracks) {
    feature1.emplace_back(GetNormalizedFeature(view1, track_id));
    feature2.emplace_back(GetNormalizedFeature(view2, track_id));
    feature3.emplace_back(GetNormalizedFeature(view3, track_id));
  }

  // Get the baseline ratios.
  ImageTriplet view_triplet;
  view_triplet.view_ids[0] = std::get<0>(triplet);
  view_triplet.view_ids[1] = std::get<1>(triplet);
  view_triplet.view_ids[2] = std::get<2>(triplet);
  view_triplet.info_one_two = (*view_pairs_)[theia::FindOrDieNoPrint(
      view_pair_map_,
      std::make_pair(view_triplet.view_ids[0], view_triplet.view_ids[1]))];
  view_triplet.info_one_three = (*view_pairs_)[theia::FindOrDieNoPrint(
      view_pair_map_,
      std::make_pair(view_triplet.view_ids[0], view_triplet.view_ids[2]))];
  view_triplet.info_two_three = (*view_pairs_)[theia::FindOrDieNoPrint(
      view_pair_map_,
      std::make_pair(view_triplet.view_ids[1], view_triplet.view_ids[2]))];

  int point_number = ComputeTripletBaselineRatios(
      view_triplet, options_.min_tri_angle, feature1, feature2, feature3, baseline);
  *weight = std::min(static_cast<double>(point_number) / options_.max_num_points, 1.0);
}

void LeastUnsquaredDeviationPositionEstimator::SetupConstraintMatrix(
    const std::vector<ImagePair>& view_pairs,
    const std::unordered_map<image_t, Vector4d>& orientations) {
  // Add the camera to camera constraints.
  std::vector<Eigen::Triplet<double> > triplet_list;
  triplet_list.reserve(9 * view_pairs.size() + 6 * baselines_.size());
  int row = 0;
  for (const auto& view_pair : view_pairs) {
    if (!theia::ContainsKey(view_id_to_index_, view_pair.image_id1) ||
        !theia::ContainsKey(view_id_to_index_, view_pair.image_id2)) {
      continue;
    }

    const int view1_index = theia::FindOrDie(view_id_to_index_, view_pair.image_id1);
    const int view2_index = theia::FindOrDie(view_id_to_index_, view_pair.image_id2);
    const int scale_index =
        theia::FindOrDieNoPrint(view_id_pair_to_index_, std::make_pair(view_pair.image_id1, view_pair.image_id2));

    // Rotate the relative translation so that it is aligned to the global
    // orientation frame.
    const Vector3d translation_direction =
        GetRotatedTranslation(theia::FindOrDie(orientations, view_pair.image_id2),
                              view_pair.tvec);

    // Add the constraint for view 1 in the minimization:
    //   position1 - position2 - scale_1_2 * translation_direction.
    if (view1_index != kConstantViewIndex) {
      triplet_list.emplace_back(row + 0, view1_index + 0, 1.0);
      triplet_list.emplace_back(row + 1, view1_index + 1, 1.0);
      triplet_list.emplace_back(row + 2, view1_index + 2, 1.0);
    }

    // Add the constraint for view 2 in the minimization:
    //   position1 - position2 - scale_1_2 * translation_direction.
    if (view2_index != kConstantViewIndex) {
      triplet_list.emplace_back(row + 0, view2_index + 0, -1.0);
      triplet_list.emplace_back(row + 1, view2_index + 1, -1.0);
      triplet_list.emplace_back(row + 2, view2_index + 2, -1.0);
    }

    // Add the constraint for scale in the minimization:
    //   position1 - position2 - scale_1_2 * translation_direction.
    triplet_list.emplace_back(row + 0, scale_index, -translation_direction[0]);
    triplet_list.emplace_back(row + 1, scale_index, -translation_direction[1]);
    triplet_list.emplace_back(row + 2, scale_index, -translation_direction[2]);

    row += 3;
  }

  if (options_.use_scale_constraints) {
    int scale_count = 0;
    for (size_t i = 0; i < triplets_.size(); ++i) {
      const double w = weights_[i];
      if (w > 0.0) {
        scale_count += 1;
        const image_t i1 = std::get<0>(triplets_[i]);
        const image_t i2 = std::get<1>(triplets_[i]);
        const image_t i3 = std::get<2>(triplets_[i]);
        const int scale12_index = theia::FindOrDieNoPrint(view_id_pair_to_index_, std::make_pair(i1, i2));
        const int scale13_index = theia::FindOrDieNoPrint(view_id_pair_to_index_, std::make_pair(i1, i3));
        const int scale23_index = theia::FindOrDieNoPrint(view_id_pair_to_index_, std::make_pair(i2, i3));
        const double b12 = baselines_[i](0);
        const double b13 = baselines_[i](1);
        const double b23 = baselines_[i](2);
        // Scale 12 and 13
        triplet_list.emplace_back(row + 0, scale12_index, w * b13 / b12);
        triplet_list.emplace_back(row + 0, scale13_index, -w);
        // Scale 12 and 23
        triplet_list.emplace_back(row + 1, scale12_index, w * b23 / b12);
        triplet_list.emplace_back(row + 1, scale23_index, -w);
        // Scale 13 and 23
        triplet_list.emplace_back(row + 2, scale13_index, w * b23 / b13);
        triplet_list.emplace_back(row + 2, scale23_index, -w);
        row += 3;
      }
    }
    LOG(INFO) << "Added " << scale_count << " scale constraints.";
  }

  constraint_matrix_.resize(
      row,
      3 * (view_id_to_index_.size() - 1) + view_pairs.size());
  constraint_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());

  VLOG(2) << view_pairs.size() << " camera to camera constraints were added "
                                  "to the position estimation problem.";
}

}
