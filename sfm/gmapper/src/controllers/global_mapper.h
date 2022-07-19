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

#ifndef COLMAP_SRC_CONTROLLERS_GLOBAL_MAPPER_H_
#define COLMAP_SRC_CONTROLLERS_GLOBAL_MAPPER_H_

#include <string>
#include <unordered_set>
#include <colmap/base/image.h>
#include <colmap/util/threading.h>
#include "base/reconstruction_manager.h"
#include "global/robust_rotation_estimator.h"
#include "sfm/global_mapper.h"
#include "sfm/incremental_triangulator.h"

namespace colmap {

class DatabaseCache;

struct GlobalMapperOptions {
 public:
  // The minimum number of matches for inlier matches to be considered.
  int min_num_matches = 15;

  // Whether to ignore the inlier matches of watermark image pairs.
  bool ignore_watermarks = false;

  // The number of threads to use during reconstruction.
  int num_threads = -1;

  // Whether to extract colors for reconstructed points.
  bool extract_colors = true;

  // tracks
  int min_track_length = 2;
  int max_track_length = std::numeric_limits<int>::max();

  // Thresholds for filtering images with degenerate intrinsics.
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  double max_extra_param = 1.0;

  // Which intrinsic parameters to optimize during the reconstruction.
  bool ba_refine_focal_length = true;
  bool ba_refine_principal_point = false;
  bool ba_refine_extra_params = true;
  bool ba_fix_prior_rotation = false;

  // The minimum number of residuals per bundle adjustment problem to
  // enable multi-threading solving of the problems.
  int ba_min_num_residuals_for_multi_threading = 50000;

  // The maximum number of global bundle adjustment iterations.
  int ba_global_max_num_iterations = 50;
  int ba_global_max_refinements = 5;
  double ba_global_max_refinement_change = 0.0005;

  // If reconstruction is provided as input, fix the existing image poses.
  bool fix_existing_images = false;

  std::string camera_path = "";

  GlobalMapper::Options Mapper() const;
  IncrementalTriangulator::Options Triangulation() const;
  BundleAdjustmentOptions GlobalBundleAdjustment() const;
  RobustRotationEstimator::Options Rotation() const;
  PositionEstimatorOptions Position() const;
  bool Check() const;

  IncrementalTriangulator::Options triangulation;
  GlobalMapper::Options mapper;
  RobustRotationEstimator::Options rotation;
  PositionEstimatorOptions position;
};

class GlobalMapperController : public Thread {
 public:
  GlobalMapperController(const GlobalMapperOptions* options,
                         const std::string& image_path,
                         const std::string& database_path,
                         ReconstructionManager* reconstruction_manager);

 protected:
  void Run() override;
  bool LoadDatabase();
  virtual void Reconstruct(const GlobalMapper::Options& mapper_options);

  const GlobalMapperOptions* options_;
  const std::string image_path_;
  const std::string database_path_;
  ReconstructionManager* reconstruction_manager_;
  DatabaseCache database_cache_;
};

// Globally filter points and images in mapper.
size_t FilterPoints(const GlobalMapperOptions& options,
                    GlobalMapper* mapper);
size_t FilterImages(const GlobalMapperOptions& options,
                    GlobalMapper* mapper);

// Globally complete and merge tracks in mapper.
size_t CompleteAndMergeTracks(const GlobalMapperOptions& options,
                              GlobalMapper* mapper);

void AdjustGlobalBundle(const GlobalMapperOptions& options,
                        GlobalMapper* mapper,
                        bool force_update_rotation = false);

void IterativeGlobalRefinement(const GlobalMapperOptions& options,
                               GlobalMapper* mapper,
                               bool force_update_rotation = false);

size_t TriangulateImage(const GlobalMapperOptions& options,
                        const Image& image, GlobalMapper* mapper);

void ExtractColors(const std::string& image_path, const image_t image_id,
                   Reconstruction* reconstruction);

bool LoadDatabaseToCache(const GlobalMapperOptions& options,
    const std::string& database_path,
    const std::unordered_set<std::string>& image_names,
    bool relative_pose,
    DatabaseCache* database_cache);

}

#endif
