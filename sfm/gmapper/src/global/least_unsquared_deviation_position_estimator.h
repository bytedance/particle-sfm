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

#ifndef COLMAP_SRC_GLOBAL_LEAST_UNSQUARED_DEVIATION_POSITION_ESTIMATOR_H_
#define COLMAP_SRC_GLOBAL_LEAST_UNSQUARED_DEVIATION_POSITION_ESTIMATOR_H_

// Modified from Theia SfM
// [LINK] https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/global_pose_estimation/least_unsquared_deviation_position_estimator.h

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unordered_map>

#include "global/position_estimator.h"
#include "global/triplet_util.h"
#include <theia/util/hash.h>

namespace colmap {

class Reconstruction;

// Estimates the camera position of views given pairwise relative poses and the
// absolute orientations of cameras. Positions are estimated using a least
// unsquared deviations solver -- essentially an L1 solver that is wrapped in an
// Iteratively Reweighted Least Squares (IRLS) formulation. This method was
// proposed in "Robust Camera Location Estimation by Convex Programming" by
// Ozyesil and Singer (CVPR 2015). Please cite this paper when using this
// method.
class LeastUnsquaredDeviationPositionEstimator : public PositionEstimator {
 public:
  struct Options {
    // Options for ADMM QP solver.
    int max_num_iterations = 400;

    // Maximum number of reweighted iterations.
    int max_num_reweighted_iterations = 10;

    // A measurement for convergence criterion.
    double convergence_criterion = 1e-4;

    int num_threads = 0;

    bool use_scale_constraints = false;

    int max_num_points = 500;

    double min_tri_angle = 6;

    bool Check() const;
  };

  LeastUnsquaredDeviationPositionEstimator(
      const Options& options, const Reconstruction& reconstruction);

  // Returns true if the optimization was a success, false if there was a
  // failure.
  bool EstimatePositions(
      const std::vector<ImagePair>& view_pairs,
      const std::unordered_map<image_t, Eigen::Vector4d>& orientations,
      std::unordered_map<image_t, Eigen::Vector3d>* positions);

 private:
  void InitializeIndexMapping(
      const std::vector<ImagePair>& view_pairs,
      const std::unordered_map<image_t, Eigen::Vector4d>& orientations);

  Eigen::Vector2d GetNormalizedFeature(const Image& view, const point3D_t track_id);

  // Computes the relative baselines between three views in a triplet. The
  // baseline is estimated from the depths of triangulated 3D points. The
  // relative positions of the triplets are then scaled to account for the
  // baseline ratios.
  void ComputeBaselineRatioForTriplet(const ImageIdTriplet& triplet,
                                      Eigen::Vector3d* baseline,
                                      double* weight);

  // Creates camera to camera constraints from relative translations.
  void SetupConstraintMatrix(
      const std::vector<ImagePair>& view_pairs,
      const std::unordered_map<image_t, Eigen::Vector4d>& orientations);

  const Options options_;
  const Reconstruction& reconstruction_;
  const std::vector<ImagePair>* view_pairs_;

  std::unordered_map<std::pair<image_t, image_t>, int> view_id_pair_to_index_;
  std::unordered_map<image_t, int> view_id_to_index_;
  std::unordered_map<std::pair<image_t, image_t>, size_t> view_pair_map_;
  std::unordered_map<std::pair<image_t, point3D_t>, point2D_t> point_map_;
  static const int kConstantViewIndex = -3;

  Eigen::SparseMatrix<double> constraint_matrix_;
  std::vector<ImageIdTriplet> triplets_;
  std::vector<Eigen::Vector3d> baselines_;
  std::vector<double> weights_;

  friend class EstimatePositionsLeastUnsquaredDeviationTest;
};

}

#endif
