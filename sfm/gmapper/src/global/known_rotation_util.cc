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

#include "global/known_rotation_util.h"

#include <ceres/rotation.h>
#include <Eigen/Core>
#include <glog/logging.h>
#include <algorithm>
#include <vector>

#include <colmap/util/threading.h>

#include <colmap/base/pose.h>
#include "base/correspondence_graph.h"
#include "base/reconstruction.h"

namespace colmap {
namespace {

// Creates the constraint matrix such that ||A * t|| is minimized, where A is
// R_i * f_i x R_j * f_j. Given known rotations, we can solve for the
// relative translation from this constraint matrix.
void CreateConstraintMatrix(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const Eigen::Vector4d& rotation1,
    const Eigen::Vector4d& rotation2,
    Eigen::MatrixXd* constraint_matrix) {
  constraint_matrix->resize(3, points1.size());

  Eigen::Matrix3d rotation_matrix1 = QuaternionToRotationMatrix(rotation1);
  Eigen::Matrix3d rotation_matrix2 = QuaternionToRotationMatrix(rotation2);

  for (int i = 0; i < points1.size(); i++) {
    const Eigen::Vector3d rotated_feature1 =
        rotation_matrix1.transpose() *
        points1[i].homogeneous();
    const Eigen::Vector3d rotated_feature2 =
        rotation_matrix2.transpose() *
        points2[i].homogeneous();

    constraint_matrix->col(i) =
        rotated_feature1.cross(rotated_feature2).transpose() *
        rotation_matrix2.transpose();
  }
}

// Determines if the majority of the points are in front of the cameras. This is
// useful for determining the sign of the relative position. Returns true if
// more than 50% of correspondences are in front of both cameras and false
// otherwise.
bool MajorityOfPointsInFrontOfCameras(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const Eigen::Vector4d& rotation1,
    const Eigen::Vector4d& rotation2,
    const Eigen::Vector3d& relative_position) {
  // Compose the relative rotation.
  Eigen::Matrix3d rotation_matrix1 = QuaternionToRotationMatrix(rotation1);
  Eigen::Matrix3d rotation_matrix2 = QuaternionToRotationMatrix(rotation2);
  const Eigen::Matrix3d relative_rotation_matrix =
      rotation_matrix2 * rotation_matrix1.transpose();

  // Tests all points for cheirality.
  std::vector<Eigen::Vector3d> points3D;
  CheckCheirality(relative_rotation_matrix, relative_position,
      points1, points2, &points3D);

  return points3D.size() > (points1.size() / 2);
}

}  // namespace

// Given known camera rotations and feature correspondences, this method solves
// for the relative translation that optimizes the epipolar error
// f_i * E * f_j^t = 0.
bool OptimizeRelativePositionWithKnownRotation(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const Eigen::Vector4d& rotation1,
    const Eigen::Vector4d& rotation2,
    Eigen::Vector3d* relative_position) {
  CHECK_NOTNULL(relative_position);
  CHECK_EQ(points1.size(), points2.size());

  // Set the initial relative position to random. This helps avoid a bad local
  // minima that is achieved from poor initialization.
  relative_position->setRandom();

  // Constants used for the IRLS solving.
  const double eps = 1e-5;
  const int kMaxIterations = 100;
  const int kMaxInnerIterations = 10;
  const double kMinWeight = 1e-7;

  // Create the constraint matrix from the known correspondences and rotations.
  Eigen::MatrixXd constraint_matrix;
  CreateConstraintMatrix(points1,
                         points2,
                         rotation1,
                         rotation2,
                         &constraint_matrix);

  // Initialize the weighting terms for each correspondence.
  Eigen::VectorXd weights(points1.size());
  weights.setConstant(1.0);

  // Solve for the relative positions using a robust IRLS.
  double cost = 0;
  int num_inner_iterations = 0;
  for (int i = 0;
       i < kMaxIterations && num_inner_iterations < kMaxInnerIterations;
       i++) {
    // Limit the minimum weight at kMinWeight.
    weights = (weights.array() < kMinWeight).select(kMinWeight, weights);

    // Apply the weights to the constraint matrix.
    const Eigen::Matrix3d lhs = constraint_matrix *
                                weights.asDiagonal().inverse() *
                                constraint_matrix.transpose();

    // Solve for the relative position which is the null vector of the weighted
    // constraints.
    const Eigen::Vector3d new_relative_position =
        lhs.jacobiSvd(Eigen::ComputeFullU).matrixU().rightCols<1>();

    // Update the weights based on the current errors.
    weights =
        (new_relative_position.transpose() * constraint_matrix).array().abs();

    // Compute the new cost.
    const double new_cost = weights.sum();

    // Check for convergence.
    const double delta = std::max(std::abs(cost - new_cost),
                                  1 - new_relative_position.squaredNorm());

    // If we have good convergence, attempt an inner iteration.
    if (delta <= eps) {
      ++num_inner_iterations;
    } else {
      num_inner_iterations = 0;
    }

    cost = new_cost;
    *relative_position = new_relative_position;
  }

  // The position solver above does not consider the sign of the relative
  // position. We can determine the sign by choosing the sign that puts the most
  // points in front of the camera.
  if (!MajorityOfPointsInFrontOfCameras(points1,
                                        points2,
                                        rotation1,
                                        rotation2,
                                        *relative_position)) {
    *relative_position *= -1.0;
  }

  return true;
}

void BatchOptimizeRelativePositionWithKnownRotation(
    const int num_threads,
    const Reconstruction& reconstruction,
    const std::unordered_map<image_t, Eigen::Vector4d>& orientations,
    std::vector<ImagePair>* pairs) {
  const CorrespondenceGraph& correspondence_graph =
      reconstruction.CorrespondenceGraph();
  std::unique_ptr<ThreadPool> pool(new ThreadPool(num_threads));
  for (auto& pair : *pairs) {
    FeatureMatches corrs = correspondence_graph.FindCorrespondencesBetweenImages(pair.image_id1, pair.image_id2);
    const Image& image1 = reconstruction.Image(pair.image_id1);
    const Camera& camera1 = reconstruction.Camera(image1.CameraId());

    const Image& image2 = reconstruction.Image(pair.image_id2);
    const Camera& camera2 = reconstruction.Camera(image2.CameraId());

    std::vector<Eigen::Vector2d> points1;
    points1.reserve(corrs.size());
    std::vector<Eigen::Vector2d> points2;
    points2.reserve(corrs.size());
    for (const auto& corr : corrs) {
      const Eigen::Vector2d point1_N =
          camera1.ImageToWorld(image1.Point2D(corr.point2D_idx1).XY());
      points1.push_back(point1_N);
      const Eigen::Vector2d point2_N =
          camera2.ImageToWorld(image2.Point2D(corr.point2D_idx2).XY());
      points2.push_back(point2_N);
    }

    pool->AddTask(OptimizeRelativePositionWithKnownRotation,
        points1, points2, orientations.at(pair.image_id1),
        orientations.at(pair.image_id2), &pair.tvec);
  }
  pool->Wait();
}

}
