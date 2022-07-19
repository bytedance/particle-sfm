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

#ifndef COLMAP_SRC_GLOBAL_LINEAR_POSITION_ESTIMATOR_H_
#define COLMAP_SRC_GLOBAL_LINEAR_POSITION_ESTIMATOR_H_

// Modified from Theia SfM
// [LINK] https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/global_pose_estimation/linear_position_estimator.h

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unordered_map>
#include <vector>

#include "global/triplet_util.h"
#include "global/position_estimator.h"
#include <theia/util/random.h>
#include <theia/util/hash.h>

namespace colmap {

class Image;
class Reconstruction;

// Estimates the camera position of views given global orientations and view
// triplets. The constraints formed by each triplet are used to create a sparse
// linear system to solve for the positions. This implementation closely follows
// "A Global Linear Method for Camera Pose Registration" by Jiang et al, ICCV
// 2013. Please see the paper for more details on the mathematics.
class LinearPositionEstimator : public PositionEstimator {
 public:
  struct Options {
    int num_threads = 0;

    // Maximum number of inverse power iterations to perform while extracting
    // the eigenvector corresponding to the smallest eigenvalue.
    int max_power_iterations = 1000;
    // The threshold at which to the iterative eigensolver method is considered
    // to be converged.
    double eigensolver_threshold = 1e-8;

    double min_tri_angle = 6;

    bool Check() const;
  };

  LinearPositionEstimator(const Options& options,
                          const Reconstruction& reconstruction);

  // Estimate the positions given view pairs and global orientation estimates.
  bool EstimatePositions(
      const std::vector<ImagePair>& view_pairs,
      const std::unordered_map<image_t, Eigen::Vector4d>& orientations,
      std::unordered_map<image_t, Eigen::Vector3d>* positions);

 private:
  // Returns the features as a unit-norm pixel ray after camera intrinsics
  // (i.e. focal length an principal point) have been removed.
  Eigen::Vector2d GetNormalizedFeature(const Image& view, const point3D_t track_id);

  // Computes the relative baselines between three views in a triplet. The
  // baseline is estimated from the depths of triangulated 3D points. The
  // relative positions of the triplets are then scaled to account for the
  // baseline ratios.
  void ComputeBaselineRatioForTriplet(const ImageIdTriplet& triplet,
                                      Eigen::Vector3d* baseline);

  // Store the triplet.
  void AddTripletConstraint(const ImageIdTriplet& view_triplet);

  // Sets up the linear system with the constraints that each triplet adds.
  void CreateLinearSystem(Eigen::SparseMatrix<double>* constraint_matrix);

  void AddTripletConstraintToSparseMatrix(
      const image_t view_id1,
      const image_t view_id2,
      const image_t view_id3,
      const Eigen::Vector3d& baseline,
      std::unordered_map<std::pair<int, int>, double>* sparse_matrix_entries);

  // A helper method to compute the relative rotations between translation
  // directions.
  void ComputeRotatedRelativeTranslationRotations(const image_t view_id0,
                                                  const image_t view_id1,
                                                  const image_t view_id2,
                                                  Eigen::Matrix3d* r012,
                                                  Eigen::Matrix3d* r201,
                                                  Eigen::Matrix3d* r120);
  // Positions are estimated from an eigenvector that is unit-norm with an
  // ambiguous sign. To ensure that the sign of the camera positions is correct,
  // we measure the relative translations from estimated camera positions and
  // compare that to the relative positions. If the sign is incorrect, we flip
  // the sign of all camera positions.
  void FlipSignOfPositionsIfNecessary(
      std::unordered_map<image_t, Eigen::Vector3d>* positions);

  const Options options_;
  const Reconstruction& reconstruction_;
  const std::vector<ImagePair>* view_pairs_;
  std::unordered_map<std::pair<image_t, image_t>, size_t> view_pair_map_;
  std::unordered_map<std::pair<image_t, point3D_t>, point2D_t> point_map_;
  const std::unordered_map<image_t, Eigen::Vector4d>* orientations_;

  std::vector<ImageIdTriplet> triplets_;
  std::vector<Eigen::Vector3d> baselines_;

  // We keep one of the positions as constant to remove the ambiguity of the
  // origin of the linear system.
  static const int kConstantPositionIndex = -1;

  std::unordered_map<image_t, int> num_triplets_for_view_;
  std::unordered_map<image_t, int> linear_system_index_;
};

}  // namespace colmap

#endif
