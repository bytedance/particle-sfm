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

#ifndef COLMAP_SRC_SFM_GLOBAL_MAPPER_H_
#define COLMAP_SRC_SFM_GLOBAL_MAPPER_H_

#include <unordered_map>
#include "base/database_cache.h"
#include "base/reconstruction.h"
#include "optim/bundle_adjustment.h"
#include "global/linear_position_estimator.h"
#include "global/nonlinear_position_estimator.h"
#include "global/least_unsquared_deviation_position_estimator.h"
#include "global/filter_util.h"
#include "global/robust_rotation_estimator.h"
#include "sfm/incremental_triangulator.h"

namespace colmap {

struct PositionEstimatorOptions {
 public:
  std::string method = "lud";
  NonlinearPositionEstimator::Options nonlinear_options;
  LinearPositionEstimator::Options linear_options;
  LeastUnsquaredDeviationPositionEstimator::Options lud_options;

  bool Check() const;
};

class GlobalMapper {
 public:
  struct Options {
    // Thresholds for bogus camera parameters. Images with bogus camera
    // parameters are filtered and ignored in triangulation.
    double min_focal_length_ratio = 0.1;  // Opening angle of ~130deg
    double max_focal_length_ratio = 10;   // Opening angle of ~5deg
    double max_extra_param = 1;

    // Maximum reprojection error in pixels for observations.
    double filter_max_reproj_error = 4.0;

    // Minimum triangulation angle in degrees for stable 3D points.
    double filter_min_tri_angle = 1.5;

    // If reconstruction is provided as input, fix the existing image poses.
    bool fix_existing_images = false;

    // Number of threads.
    int num_threads = -1;

    // If true, filter the pairwise translation estimates to remove potentially
    // bad relative poses. Removing potential outliers can increase the
    // performance of position estimation.
    bool filter_with_1dsfm = false;

    // Before the camera positions are estimated, it is wise to remove any
    // relative translations estimates that are low quality. See
    // theia/sfm/filter_view_pairs_from_relative_translation.h
    int translation_filter_num_iterations = 48;
    double translation_filter_tolerance = 0.1;

    bool Check() const;
  };

  struct LocalBundleAdjustmentReport {
    size_t num_merged_observations = 0;
    size_t num_completed_observations = 0;
    size_t num_filtered_observations = 0;
    size_t num_adjusted_observations = 0;
  };

  explicit GlobalMapper(DatabaseCache* database_cache);

  // Prepare the mapper for a new reconstruction, which might have existing
  // registered images (in which case `RegisterNextImage` must be called) or
  // which is empty (in which case `RegisterInitialImagePair` must be called).
  void BeginReconstruction(Reconstruction* reconstruction);

  // Cleanup the mapper after the current reconstruction is done. If the
  // model is discarded, the number of total and shared registered images will
  // be updated accordingly.
  void EndReconstruction(const bool discard);

  size_t TriangulateAllPoints(const IncrementalTriangulator::Options& tri_options);

  // Triangulate observations of image.
  size_t TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                          const image_t image_id);

  // Retriangulate image pairs that should have common observations according to
  // the scene graph but don't due to drift, etc. To handle drift, the employed
  // reprojection error thresholds should be relatively large. If the thresholds
  // are too large, non-robust bundle adjustment will break down; if the
  // thresholds are too small, we cannot fix drift effectively.
  size_t Retriangulate(const IncrementalTriangulator::Options& tri_options);

  // Complete tracks by transitively following the scene graph correspondences.
  // This is especially effective after bundle adjustment, since many cameras
  // and point locations might have improved. Completion of tracks enables
  // better subsequent registration of new images.
  size_t CompleteTracks(const IncrementalTriangulator::Options& tri_options);

  // Merge tracks by using scene graph correspondences. Similar to
  // `CompleteTracks`, this is effective after bundle adjustment and improves
  // the redundancy in subsequent bundle adjustments.
  size_t MergeTracks(const IncrementalTriangulator::Options& tri_options);

  // Global bundle adjustment using Ceres Solver.
  bool AdjustGlobalBundle(const Options& options,
                          const BundleAdjustmentOptions& ba_options);

  // Filter images and point observations.
  size_t FilterImages(const Options& options);
  size_t FilterPoints(const Options& options);

  void AddModifiedPoint3D(point3D_t point3D_id);

  // Get changed 3D points, since the last call to `ClearModifiedPoints3D`.
  const std::unordered_set<point3D_t>& GetModifiedPoints3D();

  // Clear the collection of changed 3D points.
  void ClearModifiedPoints3D();

  const Reconstruction& GetReconstruction() const;

  // Number of images that are registered in at least on reconstruction.
  size_t NumTotalRegImages() const;

  // Number of shared images between current reconstruction and all other
  // previous reconstructions.
  size_t NumSharedRegImages() const;

  bool EstimateGlobalRotations(const Options& options, const RobustRotationEstimator::Options& rotation_options);

  void SetGlobalRotations(double rotation_filter_max_degrees);

  void OptimizePairwiseTranslations(const Options& options);

  bool EstimatePositions(const Options& options, const PositionEstimatorOptions& position_options);

  void FillZeroPositions();

  void RegisterAllImages();

  void UpdateAllTranslations();

  void RecoverAllPoses();

 private:
  LocalBundleAdjustmentReport AdjustLocalBundle(
      const Options& options, const BundleAdjustmentOptions& ba_options,
      const IncrementalTriangulator::Options& tri_options,
      const image_t image_id, const std::vector<image_t>& local_bundle, const std::unordered_set<point3D_t>& point3D_ids);

  // Register / De-register image in current reconstruction and update
  // the number of shared images between all reconstructions.
  void RegisterImageEvent(const image_t image_id);
  void DeRegisterImageEvent(const image_t image_id);

  image_t PropagateKeyframe(image_t image_id, int delta, const std::unordered_map<image_t, std::unordered_map<image_t, size_t>>& pair_mapping);

  // Class that holds all necessary data from database in memory.
  DatabaseCache* database_cache_;

  // Class that holds data of the reconstruction.
  Reconstruction* reconstruction_;

  // Number of images that are registered in at least on reconstruction.
  size_t num_total_reg_images_;

  // Number of shared images between current reconstruction and all other
  // previous reconstructions.
  size_t num_shared_reg_images_;

  // Images that were registered before beginning the reconstruction.
  // This image list will be non-empty, if the reconstruction is continued from
  // an existing reconstruction.
  std::unordered_set<image_t> existing_image_ids_;

  // Images that have been filtered in current reconstruction.
  std::unordered_set<image_t> filtered_images_;

  // The number of registered images per camera. This information is used
  // to avoid duplicate refinement of camera parameters and degradation of
  // already refined camera parameters in local bundle adjustment when multiple
  // images share intrinsics.
  std::unordered_map<camera_t, size_t> num_reg_images_per_camera_;

  // The number of reconstructions in which images are registered.
  std::unordered_map<image_t, size_t> num_registrations_;

  // Class that is responsible for incremental triangulation.
  std::unique_ptr<IncrementalTriangulator> triangulator_;

  std::vector<ImagePair> pairs_;
  std::unordered_map<image_t, Eigen::Vector4d> orientations_;
  std::unordered_map<image_t, Eigen::Vector3d> positions_;
};

}

#endif
