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

#include "sfm/global_mapper.h"
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <math.h>

#include <colmap/base/pose.h>
#include <colmap/base/projection.h>
#include <colmap/util/math.h>
#include <colmap/util/misc.h>

#include "global/robust_rotation_estimator.h"
#include "global/linear_position_estimator.h"
#include "global/nonlinear_position_estimator.h"
#include "global/least_unsquared_deviation_position_estimator.h"
#include "global/filter_util.h"
#include "global/known_rotation_util.h"

namespace colmap {

bool PositionEstimatorOptions::Check() const {
  CHECK_OPTION(method == "nonlinear" || method == "linear" || method == "lud");
  CHECK_OPTION(nonlinear_options.Check());
  CHECK_OPTION(linear_options.Check());
  CHECK_OPTION(lud_options.Check());
  return true;
}

bool GlobalMapper::Options::Check() const {
  CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GE(max_extra_param, 0.0);
  CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
  CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
  CHECK_OPTION_GT(translation_filter_num_iterations, 0);
  CHECK_OPTION_GT(translation_filter_tolerance, 0.0);
  return true;
}

GlobalMapper::GlobalMapper(DatabaseCache* database_cache):
  database_cache_(database_cache), reconstruction_(nullptr),
  num_total_reg_images_(0), num_shared_reg_images_(0),
  triangulator_(nullptr) {
}


image_t GlobalMapper::PropagateKeyframe(image_t image_id, int delta,
    const std::unordered_map<image_t, std::unordered_map<image_t, size_t>>& pair_mapping) {
  image_t current_image_id = image_id;
  const auto& current_pair_map = pair_mapping.at(image_id);
  const auto& image = reconstruction_->Image(image_id);
  const double min_ratio = 0.6;
  const double min_angle_cos = 0.95;
  const double max_tri_angle = 10 * M_PI / 180;
  while (current_pair_map.count(current_image_id + delta) > 0) {
    current_image_id += delta;
    const auto& pair = pairs_[current_pair_map.at(current_image_id)];
    const double ratio = static_cast<double>(pair.num_correspondences) / image.NumPoints2D();
    const double angle_cos = (QuaternionRotatePoint(pair.qvec, Eigen::Vector3d(0, 0, 1))).dot(Eigen::Vector3d(0, 0, 1));
    LOG(INFO) << image_id << " -> " << current_image_id << ": r " << ratio << " cos " << angle_cos << " tri " << pair.tri_angle;
    if (ratio <= min_ratio || angle_cos <= min_angle_cos || pair.tri_angle >= max_tri_angle) {
      break;
    }
  }

  if (current_image_id == image_id) {
    current_image_id = kInvalidImageId;
  }
  return current_image_id;
}

bool GlobalMapper::EstimateGlobalRotations(const Options& options, const RobustRotationEstimator::Options& rotation_options) {
  return ::colmap::EstimateGlobalRotations(rotation_options,
      &pairs_, &orientations_);
}

void GlobalMapper::SetGlobalRotations(double rotation_filter_max_degrees) {
  for (const auto& image : reconstruction_->Images()) {
    if (image.second.HasQvecPrior()) {
      orientations_[image.first] = image.second.QvecPrior();
    }
  }

  FilterViewPairsFromOrientation(orientations_, rotation_filter_max_degrees,
      &pairs_);
  RemoveDisconnectedViewPairs(pairs_);
}

void GlobalMapper::OptimizePairwiseTranslations(const Options& options) {
  BatchOptimizeRelativePositionWithKnownRotation(options.num_threads,
      *reconstruction_, orientations_, &pairs_);
}

bool GlobalMapper::EstimatePositions(const Options& options, const PositionEstimatorOptions& position_options) {
  std::unique_ptr<PositionEstimator> position_estimator;
  if (position_options.method == "linear") {
    position_estimator.reset(new LinearPositionEstimator(position_options.linear_options, *reconstruction_));
  } else if (position_options.method == "nonlinear") {
    position_estimator.reset(new NonlinearPositionEstimator(position_options.nonlinear_options, *reconstruction_));
  } else if (position_options.method == "lud") {
    position_estimator.reset(new LeastUnsquaredDeviationPositionEstimator(position_options.lud_options, *reconstruction_));
  }
  CHECK(position_estimator);

  if (options.filter_with_1dsfm) {
    FilterViewPairsFromRelativeTranslationOptions filter_options;
    filter_options.num_iterations = options.translation_filter_num_iterations;
    filter_options.translation_projection_tolerance = options.translation_filter_tolerance;
    FilterViewPairsFromRelativeTranslation(filter_options, orientations_, &pairs_);
    RemoveDisconnectedViewPairs(pairs_);
  }

  bool result = position_estimator->EstimatePositions(pairs_, orientations_, &positions_);
  return result;
}

void GlobalMapper::FillZeroPositions() {
  for (const auto& orientation : orientations_) {
    positions_.emplace(orientation.first, Eigen::Vector3d::Zero());
  }
}

void GlobalMapper::RegisterAllImages() {
  if (num_total_reg_images_ > 0) {
    LOG(INFO) << "Already registered. Ignore the calling";
    return;
  }

  for (const auto& position : positions_) {
    image_t image_id = position.first;
    Image& image = reconstruction_->Image(image_id);
    auto iter = orientations_.find(image_id);
    if (iter == orientations_.end()) {
      LOG(WARNING) << "No rotation for image " << image_id;
      continue;
    }
    reconstruction_->RegisterImage(image_id);
    RegisterImageEvent(image_id);
    image.Qvec() = iter->second;
    image.Tvec() = -QuaternionRotatePoint(iter->second, position.second);
  }
  LOG(INFO) << "Registered " << num_total_reg_images_ << " images.";
}

void GlobalMapper::UpdateAllTranslations() {
  for (const auto& position : positions_) {
    image_t image_id = position.first;
    Image& image = reconstruction_->Image(image_id);
    image.Tvec() = -QuaternionRotatePoint(image.Qvec(), position.second);
  }
}

void GlobalMapper::RecoverAllPoses() {
  for (const auto& position : positions_) {
    image_t image_id = position.first;
    Image& image = reconstruction_->Image(image_id);
    auto iter = orientations_.find(image_id);
    if (iter == orientations_.end()) {
      LOG(WARNING) << "No rotation for image " << image_id;
      continue;
    }
    if (!image.IsRegistered()) {
      reconstruction_->RegisterImage(image_id);
      RegisterImageEvent(image_id);
    }
    image.Qvec() = iter->second;
    image.Tvec() = -QuaternionRotatePoint(iter->second, position.second);
  }
}

void GlobalMapper::BeginReconstruction(Reconstruction* reconstruction) {
  CHECK(reconstruction_ == nullptr);

  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  reconstruction_->SetUp();

  triangulator_.reset(new IncrementalTriangulator(
      &reconstruction_->CorrespondenceGraph(), reconstruction));

  for (const image_t image_id : reconstruction_->RegImageIds()) {
    RegisterImageEvent(image_id);
  }

  existing_image_ids_ =
      std::unordered_set<image_t>(reconstruction_->RegImageIds().begin(),
                                  reconstruction_->RegImageIds().end());
  filtered_images_.clear();
  num_shared_reg_images_ = 0;
  num_reg_images_per_camera_.clear();

  pairs_ = reconstruction_->CorrespondenceGraph().AllPairs();
  orientations_.clear();
  positions_.clear();
}

void GlobalMapper::EndReconstruction(const bool discard) {
  CHECK_NOTNULL(reconstruction_);

  if (discard) {
    for (const image_t image_id : reconstruction_->RegImageIds()) {
      DeRegisterImageEvent(image_id);
    }
  }

  pairs_.clear();
  orientations_.clear();
  positions_.clear();

  reconstruction_->TearDown();
  reconstruction_ = nullptr;
  triangulator_.reset();
}

size_t GlobalMapper::TriangulateAllPoints(const IncrementalTriangulator::Options& tri_options) {
  std::vector<image_t> image_ids(reconstruction_->RegImageIds().begin(), reconstruction_->RegImageIds().end());
  std::sort(image_ids.begin(), image_ids.end());
  size_t total_points = 0;
  for (image_t image_id : image_ids) {
    const auto& image = reconstruction_->Image(image_id);
    const size_t num_existing_points3D = image.NumPoints3D();
    LOG(INFO) << "Image " << image_id << " sees " << num_existing_points3D << " / "
              << image.NumObservations() << " points";
    triangulator_->TriangulateImage(tri_options, image_id);
    LOG(INFO) << "  => Triangulated "
              << (image.NumPoints3D() - num_existing_points3D) << " points";
    total_points += (image.NumPoints3D() - num_existing_points3D);
  }
  return total_points;
}

size_t GlobalMapper::TriangulateImage(
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->TriangulateImage(tri_options, image_id);
}

size_t GlobalMapper::Retriangulate(const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->Retriangulate(tri_options);
}

size_t GlobalMapper::CompleteTracks(const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->CompleteAllTracks(tri_options);
}

size_t GlobalMapper::MergeTracks(const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->MergeAllTracks(tri_options);
}

const Reconstruction& GlobalMapper::GetReconstruction() const {
  CHECK_NOTNULL(reconstruction_);
  return *reconstruction_;
}

size_t GlobalMapper::NumTotalRegImages() const {
  return num_total_reg_images_;
}

size_t GlobalMapper::NumSharedRegImages() const {
  return num_shared_reg_images_;
}

GlobalMapper::LocalBundleAdjustmentReport GlobalMapper::AdjustLocalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options, const image_t image_id,
    const std::vector<image_t>& local_bundle,
    const std::unordered_set<point3D_t>& point3D_ids) {
  LocalBundleAdjustmentReport report;

  // Do the bundle adjustment only if there is any connected images.
  if (local_bundle.size() > 0) {
    BundleAdjustmentConfig ba_config;
    ba_config.AddImage(image_id);
    for (const image_t local_image_id : local_bundle) {
      ba_config.AddImage(local_image_id);
    }

    // Fix the existing images, if option specified.
    if (options.fix_existing_images) {
      for (const image_t local_image_id : local_bundle) {
        if (existing_image_ids_.count(local_image_id)) {
          ba_config.SetConstantPose(local_image_id);
        }
      }
    }

    // Determine which cameras to fix, when not all the registered images
    // are within the current local bundle.
    std::unordered_map<camera_t, size_t> num_images_per_camera;
    for (const image_t image_id : ba_config.Images()) {
      const Image& image = reconstruction_->Image(image_id);
      num_images_per_camera[image.CameraId()] += 1;
    }

    for (const auto& camera_id_and_num_images_pair : num_images_per_camera) {
      const size_t num_reg_images_for_camera =
          num_reg_images_per_camera_.at(camera_id_and_num_images_pair.first);
      if (camera_id_and_num_images_pair.second < num_reg_images_for_camera) {
        ba_config.SetConstantCamera(camera_id_and_num_images_pair.first);
      }
    }

    // Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment.
    if (local_bundle.size() == 1) {
      ba_config.SetConstantPose(local_bundle[0]);
      ba_config.SetConstantTvec(image_id, {0});
    } else if (local_bundle.size() > 1) {
      const image_t image_id1 = local_bundle[local_bundle.size() - 1];
      const image_t image_id2 = local_bundle[local_bundle.size() - 2];
      ba_config.SetConstantPose(image_id1);
      if (!options.fix_existing_images ||
          !existing_image_ids_.count(image_id2)) {
        ba_config.SetConstantTvec(image_id2, {0});
      }
    }

    // Make sure, we refine all new and short-track 3D points, no matter if
    // they are fully contained in the local image set or not. Do not include
    // long track 3D points as they are usually already very stable and adding
    // to them to bundle adjustment and track merging/completion would slow
    // down the local bundle adjustment significantly.
    std::unordered_set<point3D_t> variable_point3D_ids;
    for (const point3D_t point3D_id : point3D_ids) {
      const Point3D& point3D = reconstruction_->Point3D(point3D_id);
      const size_t kMaxTrackLength = 15;
      if (!point3D.HasError() || point3D.Track().Length() <= kMaxTrackLength) {
        ba_config.AddVariablePoint(point3D_id);
        variable_point3D_ids.insert(point3D_id);
      }
    }

    // Adjust the local bundle.
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    bundle_adjuster.Solve(reconstruction_);

    report.num_adjusted_observations =
        bundle_adjuster.Summary().num_residuals / 2;

    // Merge refined tracks with other existing points.
    report.num_merged_observations =
        triangulator_->MergeTracks(tri_options, variable_point3D_ids);
    // Complete tracks that may have failed to triangulate before refinement
    // of camera pose and calibration in bundle-adjustment. This may avoid
    // that some points are filtered and it helps for subsequent image
    // registrations.
    report.num_completed_observations =
        triangulator_->CompleteTracks(tri_options, variable_point3D_ids);
    report.num_completed_observations +=
        triangulator_->CompleteImage(tri_options, image_id);
  }

  // Filter both the modified images and all changed 3D points to make sure
  // there are no outlier points in the model. This results in duplicate work as
  // many of the provided 3D points may also be contained in the adjusted
  // images, but the filtering is not a bottleneck at this point.
  std::unordered_set<image_t> filter_image_ids;
  filter_image_ids.insert(image_id);
  filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
  report.num_filtered_observations = reconstruction_->FilterPoints3DInImages(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      filter_image_ids);
  report.num_filtered_observations += reconstruction_->FilterPoints3D(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      point3D_ids);

  return report;
}

void GlobalMapper::AddModifiedPoint3D(point3D_t point3D_id) {
  triangulator_->AddModifiedPoint3D(point3D_id);
}

const std::unordered_set<point3D_t>& GlobalMapper::GetModifiedPoints3D() {
  return triangulator_->GetModifiedPoints3D();
}

void GlobalMapper::ClearModifiedPoints3D() {
  triangulator_->ClearModifiedPoints3D();
}

bool GlobalMapper::AdjustGlobalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options) {
  CHECK_NOTNULL(reconstruction_);

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                       "registered for global "
                                       "bundle-adjustment";

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // Fix the existing images, if option specified.
  if (options.fix_existing_images) {
    for (const image_t image_id : reg_image_ids) {
      if (existing_image_ids_.count(image_id)) {
        ba_config.SetConstantPose(image_id);
      }
    }
  }

  // Fix 7-DOFs of the bundle adjustment problem.
  ba_config.SetConstantPose(reg_image_ids[0]);
  if (!options.fix_existing_images ||
      !existing_image_ids_.count(reg_image_ids[1])) {
    ba_config.SetConstantTvec(reg_image_ids[1], {0});
  }

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  // Normalize scene for numerical stability and
  // to avoid large scale changes in viewer.
  reconstruction_->Normalize();

  return true;
}

size_t GlobalMapper::FilterImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  // Do not filter images in the early stage of the reconstruction, since the
  // calibration is often still refining a lot. Hence, the camera parameters
  // are not stable in the beginning.
  const size_t kMinNumImages = 20;
  if (reconstruction_->NumRegImages() < kMinNumImages) {
    return {};
  }

  const std::vector<image_t> image_ids = reconstruction_->FilterImages(
      options.min_focal_length_ratio, options.max_focal_length_ratio,
      options.max_extra_param);

  for (const image_t image_id : image_ids) {
    DeRegisterImageEvent(image_id);
    filtered_images_.insert(image_id);
  }

  return image_ids.size();
}

size_t GlobalMapper::FilterPoints(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());
  return reconstruction_->FilterAllPoints3D(options.filter_max_reproj_error,
                                            options.filter_min_tri_angle);
}

void GlobalMapper::RegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_[image.CameraId()];
  num_reg_images_for_camera += 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image += 1;
  if (num_regs_for_image == 1) {
    num_total_reg_images_ += 1;
  } else if (num_regs_for_image > 1) {
    num_shared_reg_images_ += 1;
  }
}

void GlobalMapper::DeRegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_.at(image.CameraId());
  CHECK_GT(num_reg_images_for_camera, 0);
  num_reg_images_for_camera -= 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image -= 1;
  if (num_regs_for_image == 0) {
    num_total_reg_images_ -= 1;
  } else if (num_regs_for_image > 0) {
    num_shared_reg_images_ -= 1;
  }
}

}
