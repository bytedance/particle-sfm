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

#include "global/linear_position_estimator.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <algorithm>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <colmap/base/pose.h>
#include <colmap/util/threading.h>
#include <theia/math/matrix/spectra_linear_operator.h>
#include <theia/util/map_util.h>

#include "spectra/include/SymEigsShiftSolver.h"
#include "base/reconstruction.h"
#include "global/triplet_util.h"
#include "global/image_triplet.h"

namespace colmap {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;

namespace {

// Adds the constraint from the triplet to the symmetric matrix. Our standard
// constraint matrix A is a 3M x 3N matrix with M triplet constraints and N
// cameras. We seek to construct A^t * A directly. For each triplet constraint
// in our matrix A (i.e. a 3-row block), we can compute the corresponding
// entries in A^t * A with the following summation:
//
//   A^t * A += Row(i)^t * Row(i)
//
// for each triplet constraint i.
void AddTripletConstraintToSymmetricMatrix(
    const std::vector<Matrix3d>& constraints,
    const std::vector<int>& view_indices,
    std::unordered_map<std::pair<int, int>, double>* sparse_matrix_entries) {
  // Construct Row(i)^t * Row(i). If we denote the row as a block matrix:
  //
  //   Row(i) = [A | B | C]
  //
  // then we have:
  //
  //   Row(i)^t * Row(i) = [A | B | C]^t * [A | B | C]
  //                     = [ A^t * A  |  A^t * B  |  A^t * C]
  //                       [ B^t * A  |  B^t * B  |  B^t * C]
  //                       [ C^t * A  |  C^t * B  |  C^t * C]
  //
  // Since A^t * A is symmetric, we only store the upper triangular portion.
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // Skip any block entries that correspond to the lower triangular portion
      // of the matrix.
      if (view_indices[i] > view_indices[j]) {
        continue;
      }

      // Compute the A^t * B, etc. matrix.
      const Eigen::Matrix3d symmetric_constraint =
          constraints[i].transpose() * constraints[j];

      // Add to the 3x3 block corresponding to (i, j)
      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
          const std::pair<int, int> row_col(view_indices[i] + r,
                                            view_indices[j] + c);
          (*sparse_matrix_entries)[row_col] += symmetric_constraint(r, c);
        }
      }
    }
  }
}

// Returns true if the vector R1 * (c2 - c1) is in the same direction as t_12.
bool VectorsAreSameDirection(const Vector3d& position1,
                             const Vector3d& position2,
                             const Vector4d& rotation2,
                             const Vector3d& relative_position21) {
  const Vector3d global_relative_position =
      (position1 - position2).normalized();
  Vector3d rotated_relative_position = QuaternionRotatePoint(rotation2, global_relative_position);
  return rotated_relative_position.dot(relative_position21) > 0;
}

}  // namespace

bool LinearPositionEstimator::Options::Check() const {
  CHECK_OPTION_GE(num_threads, 0);
  CHECK_OPTION_GT(max_power_iterations, 0);
  CHECK_OPTION_GT(eigensolver_threshold, 0.0);
  CHECK_OPTION_GE(min_tri_angle, 0.0);
  return true;
}

LinearPositionEstimator::LinearPositionEstimator(
    const Options& options, const Reconstruction& reconstruction)
    : options_(options), reconstruction_(reconstruction) {
  CHECK(options_.Check());
}

bool LinearPositionEstimator::EstimatePositions(
    const std::vector<ImagePair>& view_pairs,
    const std::unordered_map<image_t, Vector4d>& orientations,
    std::unordered_map<image_t, Vector3d>* positions) {
  CHECK_NOTNULL(positions)->clear();
  view_pairs_ = &view_pairs;
  orientations_ = &orientations;

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
  for (int i = 0; i < triplets_.size(); i++) {
    AddTripletConstraint(triplets_[i]);
    pool->AddTask(&LinearPositionEstimator::ComputeBaselineRatioForTriplet,
              this,
              triplets_[i],
              &baselines_[i]);
  }
  pool->Wait();

  VLOG(2) << "Building the constraint matrix...";
  // Create the linear system based on triplet constraints.
  Eigen::SparseMatrix<double> constraint_matrix;
  CreateLinearSystem(&constraint_matrix);

  // Solve for positions by examining the smallest eigenvalues. Since we have
  // set one position constant at the origin, we only need to solve for the
  // eigenvector corresponding to the smallest eigenvalue. This can be done
  // efficiently with inverse power iterations.
  VLOG(2) << "Solving for positions from the sparse eigenvalue problem...";
  theia::SparseSymShiftSolveLLT op(constraint_matrix);
  Spectra::SymEigsShiftSolver<double, Spectra::LARGEST_MAGN, theia::SparseSymShiftSolveLLT>
          eigs(&op, 1, 6, 0.0);
  eigs.init();
  eigs.compute();

  // Compute with power iterations.
  const Eigen::VectorXd solution = eigs.eigenvectors().col(0);

  // Add the solutions to the output. Set the position with an index of -1 to
  // be at the origin.
  for (const auto& view_index : linear_system_index_) {
    if (view_index.second < 0) {
      (*positions)[view_index.first].setZero();
    } else {
      (*positions)[view_index.first] =
          solution.segment<3>(view_index.second * 3);
    }
  }

  // Flip the sign of the positions if necessary.
  FlipSignOfPositionsIfNecessary(positions);

  return true;
}

// An alternative interface is to instead add triplets one by one to linear
// estimator. This allows for adding redundant observations of triplets, which
// may be useful if there are multiple estimates of the data.
void LinearPositionEstimator::AddTripletConstraint(
    const ImageIdTriplet& view_triplet) {
  num_triplets_for_view_[std::get<0>(view_triplet)] += 1;
  num_triplets_for_view_[std::get<1>(view_triplet)] += 1;
  num_triplets_for_view_[std::get<2>(view_triplet)] += 1;

  // Determine the order of the views in the linear system. We subtract 1 from
  // the linear system index so that the first position added to the system
  // will be set constant (index of -1 is intentionally not evaluated later).
  theia::InsertIfNotPresent(&linear_system_index_,
                     std::get<0>(view_triplet),
                     linear_system_index_.size() - 1);
  theia::InsertIfNotPresent(&linear_system_index_,
                     std::get<1>(view_triplet),
                     linear_system_index_.size() - 1);
  theia::InsertIfNotPresent(&linear_system_index_,
                     std::get<2>(view_triplet),
                     linear_system_index_.size() - 1);
}

void LinearPositionEstimator::ComputeBaselineRatioForTriplet(
    const ImageIdTriplet& triplet, Vector3d* baseline) {
  baseline->setZero();

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

  ComputeTripletBaselineRatios(
      view_triplet, options_.min_tri_angle, feature1, feature2, feature3, baseline);
}

// Sets up the linear system with the constraints that each triplet adds.
void LinearPositionEstimator::CreateLinearSystem(
    Eigen::SparseMatrix<double>* constraint_matrix) {
  const int num_views = num_triplets_for_view_.size();

  std::unordered_map<std::pair<int, int>, double> sparse_matrix_entries;
  sparse_matrix_entries.reserve(27 * num_triplets_for_view_.size());
  int valid_triplet = 0;
  for (int i = 0; i < triplets_.size(); i++) {
    if (baselines_[i].x() < 0.9) {
      continue;
    }
    const image_t& view_id1 = std::get<0>(triplets_[i]);
    const image_t& view_id2 = std::get<1>(triplets_[i]);
    const image_t& view_id3 = std::get<2>(triplets_[i]);
    AddTripletConstraintToSparseMatrix(
        view_id1, view_id2, view_id3, baselines_[i], &sparse_matrix_entries);
    valid_triplet += 1;
  }
  LOG(INFO) << "triplet: " << valid_triplet << "/" << triplets_.size();

  // Set the sparse matrix from the container of the accumulated entries.
  std::vector<Eigen::Triplet<double> > triplet_list;
  triplet_list.reserve(sparse_matrix_entries.size());
  for (const auto& sparse_matrix_entry : sparse_matrix_entries) {
    // Skip this entry if the indices are invalid. This only occurs when we
    // encounter a constraint with the constant camera (which has a view index
    // of -1).
    if (sparse_matrix_entry.first.first < 0 ||
        sparse_matrix_entry.first.second < 0) {
      continue;
    }
    triplet_list.emplace_back(sparse_matrix_entry.first.first,
                              sparse_matrix_entry.first.second,
                              sparse_matrix_entry.second);
  }

  // We construct the constraint matrix A^t * A directly, which is an
  // N - 1 x N - 1 matrix where N is the number of cameras (and 3 entries per
  // camera, corresponding to the camera position entries).
  constraint_matrix->resize((num_views - 1) * 3, (num_views - 1) * 3);
  constraint_matrix->setFromTriplets(triplet_list.begin(), triplet_list.end());
}

void LinearPositionEstimator::ComputeRotatedRelativeTranslationRotations(
    const image_t view_id0,
    const image_t view_id1,
    const image_t view_id2,
    Eigen::Matrix3d* r012,
    Eigen::Matrix3d* r201,
    Eigen::Matrix3d* r120) {
  // Relative camera positions.
  const Eigen::Vector4d& orientation1_aa =
      theia::FindOrDieNoPrint(*orientations_, view_id1);
  const Eigen::Vector4d& orientation2_aa =
      theia::FindOrDieNoPrint(*orientations_, view_id2);
  const Matrix3d orientation1 = QuaternionToRotationMatrix(orientation1_aa);
  const Matrix3d orientation2 = QuaternionToRotationMatrix(orientation2_aa);
  const Vector3d t01 =
      orientation1.transpose() *
      (*view_pairs_)[theia::FindOrDieNoPrint(view_pair_map_, std::make_pair(view_id0, view_id1))].tvec;
  const Vector3d t02 =
      orientation2.transpose() *
      (*view_pairs_)[theia::FindOrDieNoPrint(view_pair_map_, std::make_pair(view_id0, view_id2))].tvec;
  const Vector3d t12 =
      orientation2.transpose() *
      (*view_pairs_)[theia::FindOrDieNoPrint(view_pair_map_, std::make_pair(view_id1, view_id2))].tvec;

  // Rotations between the translation vectors.
  *r012 = Eigen::Quaterniond::FromTwoVectors(t12, -t01).toRotationMatrix();
  *r201 = Eigen::Quaterniond::FromTwoVectors(t01, t02).toRotationMatrix();
  *r120 = Eigen::Quaterniond::FromTwoVectors(-t02, -t12).toRotationMatrix();
}

// Adds a triplet constraint to the linear system. The weight of the constraint
// (w), the global orientations, baseline (ratios), and view triplet information
// are needed to form the constraint.
void LinearPositionEstimator::AddTripletConstraintToSparseMatrix(
    const image_t view_id0,
    const image_t view_id1,
    const image_t view_id2,
    const Eigen::Vector3d& baselines,
    std::unordered_map<std::pair<int, int>, double>* sparse_matrix_entries) {
  // Weight each term by the inverse of the # of triplet that the nodes
  // participate in.
  const double w =
      1.0 / std::sqrt(std::min({num_triplets_for_view_[view_id0],
                                num_triplets_for_view_[view_id1],
                                num_triplets_for_view_[view_id2]}));

  // Get the index of each camera in the sparse matrix.
  const std::vector<int> view_indices = {
      static_cast<int>(3 * theia::FindOrDie(linear_system_index_, view_id0)),
      static_cast<int>(3 * theia::FindOrDie(linear_system_index_, view_id1)),
      static_cast<int>(3 * theia::FindOrDie(linear_system_index_, view_id2))};

  // Compute the rotations between relative translations.
  Eigen::Matrix3d r012, r201, r120;
  ComputeRotatedRelativeTranslationRotations(
      view_id0, view_id1, view_id2, &r012, &r201, &r120);

  // Baselines ratios.
  const double s_012 = baselines[0] / baselines[2];
  const double s_201 = baselines[1] / baselines[0];
  const double s_120 = baselines[2] / baselines[1];

  // Assume that t01 is perfect and solve for c2.
  std::vector<Eigen::Matrix3d> constraints(3);
  constraints[0] =
      (-s_201 * r201 + r012.transpose() / s_012 + Matrix3d::Identity()) * w;
  constraints[1] =
      (s_201 * r201 - r012.transpose() / s_012 + Matrix3d::Identity()) * w;
  constraints[2] = -2.0 * w * Matrix3d::Identity();
  AddTripletConstraintToSymmetricMatrix(
      constraints, view_indices, sparse_matrix_entries);

  // Assume t02 is perfect and solve for c1.
  constraints[0] =
      (-r201.transpose() / s_201 + s_120 * r120 + Matrix3d::Identity()) * w;
  constraints[1] = -2.0 * w * Matrix3d::Identity();
  constraints[2] =
      (r201.transpose() / s_201 - s_120 * r120 + Matrix3d::Identity()) * w;
  AddTripletConstraintToSymmetricMatrix(
      constraints, view_indices, sparse_matrix_entries);

  // Assume t12 is perfect and solve for c0.
  constraints[0] = -2.0 * w * Matrix3d::Identity();
  constraints[1] =
      (-s_012 * r012 + r120.transpose() / s_120 + Matrix3d::Identity()) * w;
  constraints[2] =
      (s_012 * r012 - r120.transpose() / s_120 + Matrix3d::Identity()) * w;
  AddTripletConstraintToSymmetricMatrix(
      constraints, view_indices, sparse_matrix_entries);
}

Eigen::Vector2d LinearPositionEstimator::GetNormalizedFeature(const Image& view,
                                                      const point3D_t track_id) {
  const Camera& camera = reconstruction_.Camera(view.CameraId());
  const Eigen::Vector2d& feature = view.Point2D(
      point_map_.at(std::make_pair(view.ImageId(), track_id))).XY();
  return camera.ImageToWorld(feature);
}

void LinearPositionEstimator::FlipSignOfPositionsIfNecessary(
    std::unordered_map<image_t, Vector3d>* positions) {
  // If this value is below zero, then we should flip the sign.
  int correct_sign_votes = 0;
  for (const auto& view_pair : *view_pairs_) {
    // Only count the votes for edges where both positions were successfully
    // estimated.
    const Vector3d* position1 = theia::FindOrNull(*positions, view_pair.image_id1);
    const Vector3d* position2 = theia::FindOrNull(*positions, view_pair.image_id2);
    if (position1 == nullptr || position2 == nullptr) {
      continue;
    }

    // Check the relative translation of views 1 and 2 in the triplet.
    if (VectorsAreSameDirection(
            *position1,
            *position2,
            theia::FindOrDieNoPrint(*orientations_, view_pair.image_id2),
            view_pair.tvec)) {
      correct_sign_votes += 1;
    } else {
      correct_sign_votes -= 1;
    }
  }

  // If the sign of the votes is below zero, we must flip the sign of all
  // position estimates.
  if (correct_sign_votes < 0) {
    const int num_correct_votes =
        (view_pairs_->size() + correct_sign_votes) / 2;
    VLOG(2) << "Sign of the positions was incorrect: " << num_correct_votes
            << " of " << view_pairs_->size()
            << " relative translations had the correct sign. "
               "Flipping the sign of the camera positions.";
    for (auto& position : *positions) {
      position.second *= -1.0;
    }
  }
}

}  // namespace colmap
