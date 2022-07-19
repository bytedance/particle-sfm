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

#include "controllers/global_mapper.h"
#include <colmap/util/timer.h>
#include <colmap/util/misc.h>

namespace colmap {

GlobalMapper::Options GlobalMapperOptions::Mapper() const {
  GlobalMapper::Options options = mapper;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  options.fix_existing_images = fix_existing_images;
  return options;
}

RobustRotationEstimator::Options GlobalMapperOptions::Rotation() const {
  return rotation;
}

PositionEstimatorOptions GlobalMapperOptions::Position() const {
  return position;
}

BundleAdjustmentOptions GlobalMapperOptions::GlobalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = 1e-6;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 1e-8;
  options.solver_options.max_num_iterations = ba_global_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = true;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.min_num_residuals_for_multi_threading =
	  ba_min_num_residuals_for_multi_threading;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::TRIVIAL;

  // TODO: This is from the subclass. Need to determine the true parameters?
  options.refine_rotation = false;
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_extra_params = false;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  return options;
}

IncrementalTriangulator::Options GlobalMapperOptions::Triangulation() const {
  IncrementalTriangulator::Options options = triangulation;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  return options;
}

bool GlobalMapperOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GE(min_track_length, 0);
  CHECK_OPTION_GT(max_track_length, 1);
  CHECK_OPTION_GE(max_track_length, min_track_length);
  CHECK_OPTION(Triangulation().Check());
  CHECK_OPTION(Mapper().Check());
  CHECK_OPTION(Rotation().Check());
  CHECK_OPTION(Position().Check());
  return true;
}

GlobalMapperController::GlobalMapperController(
    const GlobalMapperOptions* options, const std::string& image_path,
    const std::string& database_path,
    ReconstructionManager* reconstruction_manager)
    : options_(options),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_manager_(reconstruction_manager) {
  CHECK(options_->Check());
}

void GlobalMapperController::Run() {
  if (!LoadDatabase()) {
    return;
  }

  Reconstruct(options_->Mapper());
  std::cout << std::endl;
  GetTimer().PrintMinutes();
}

bool GlobalMapperController::LoadDatabase() {
  // Make sure images of the given reconstruction are also included when
  // manually specifying images for the reconstrunstruction procedure.
  std::cout << "Loading database................................" << std::endl;
  std::unordered_set<std::string> image_names;
  if (reconstruction_manager_->Size() == 1) {
    const Reconstruction& reconstruction = reconstruction_manager_->Get(0);
    for (const image_t image_id : reconstruction.RegImageIds()) {
      const auto& image = reconstruction.Image(image_id);
      image_names.insert(image.Name());
    }
  }

  return LoadDatabaseToCache(*options_, database_path_, image_names, true, &database_cache_);
}

void GlobalMapperController::Reconstruct(const GlobalMapper::Options& init_mapper_options) {
    const bool kDiscardReconstruction = true;
    GlobalMapper mapper(&database_cache_);

    size_t reconstruction_idx = reconstruction_manager_->Add();
    Reconstruction& reconstruction =
          reconstruction_manager_->Get(reconstruction_idx);
    mapper.BeginReconstruction(&reconstruction);

    // estimate global rotation
    PrintHeading1(StringPrintf("Estimate Global Rotations"));
    auto mapper_options = options_->Mapper();
    if (!mapper.EstimateGlobalRotations(mapper_options, options_->Rotation())) {
        std::cout << "  => Global rotation failed" << std::endl;
        mapper.EndReconstruction(kDiscardReconstruction);
        reconstruction_manager_->Delete(reconstruction_idx);
        return;
    }

    // optimize pairwise translations with known rotations.
    PrintHeading1(StringPrintf("Refine Positions with Known Rotations"));
    mapper.OptimizePairwiseTranslations(mapper_options);

    // estimate global translations
    PrintHeading1(StringPrintf("Estimate Global Positions"));
    if (!mapper.EstimatePositions(mapper_options, options_->Position())) {
        std::cout << "  => Global position failed" << std::endl;
        mapper.EndReconstruction(kDiscardReconstruction);
        reconstruction_manager_->Delete(reconstruction_idx);
        return;
    }

    // register all images
    PrintHeading1(StringPrintf("Register All Images"));
    mapper.RegisterAllImages();

    // triangulate all points
    PrintHeading1(StringPrintf("Triangulate All Points"));
    mapper.TriangulateAllPoints(options_->Triangulation());

    // global bundle adjustment
    IterativeGlobalRefinement(*options_, &mapper, false); // first optimize translation with fixed rotation
    IterativeGlobalRefinement(*options_, &mapper, true); // then joint optimize both
    if (options_->extract_colors) {
      PrintHeading1(StringPrintf("Extract Colors"));
      reconstruction.ExtractColorsForAllImages(image_path_);
    }
    mapper.EndReconstruction(false);
}

size_t FilterPoints(const GlobalMapperOptions& options,
                    GlobalMapper* mapper) {
  const size_t num_filtered_observations =
      mapper->FilterPoints(options.Mapper());
  std::cout << "  => Filtered observations: " << num_filtered_observations
            << std::endl;
  return num_filtered_observations;
}

size_t FilterImages(const GlobalMapperOptions& options,
                    GlobalMapper* mapper) {
  const size_t num_filtered_images = mapper->FilterImages(options.Mapper());
  std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
  return num_filtered_images;
}

size_t CompleteAndMergeTracks(const GlobalMapperOptions& options,
                              GlobalMapper* mapper) {
  const size_t num_completed_observations =
      mapper->CompleteTracks(options.Triangulation());
  std::cout << "  => Completed observations: " << num_completed_observations
            << std::endl;
  const size_t num_merged_observations =
      mapper->MergeTracks(options.Triangulation());
  std::cout << "  => Merged observations: " << num_merged_observations
            << std::endl;
  return num_completed_observations + num_merged_observations;
}

void AdjustGlobalBundle(const GlobalMapperOptions& options,
                        GlobalMapper* mapper,
                        bool force_update_rotation) {
  BundleAdjustmentOptions custom_ba_options = options.GlobalBundleAdjustment();
  if (force_update_rotation) {
    custom_ba_options.refine_rotation = !options.ba_fix_prior_rotation;
    custom_ba_options.refine_focal_length = options.ba_refine_focal_length;
    custom_ba_options.refine_principal_point = options.ba_refine_principal_point;
    custom_ba_options.refine_extra_params = options.ba_refine_extra_params;
  }

  // Use stricter convergence criteria for first registered images.
  const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();
  const size_t kMinNumRegImagesForFastBA = 10;
  if (num_reg_images < kMinNumRegImagesForFastBA) {
    custom_ba_options.solver_options.function_tolerance /= 10;
    custom_ba_options.solver_options.gradient_tolerance /= 10;
    custom_ba_options.solver_options.parameter_tolerance /= 10;
    custom_ba_options.solver_options.max_num_iterations *= 2;
    custom_ba_options.solver_options.max_linear_solver_iterations = 200;
  }

  if (custom_ba_options.refine_rotation) {
    PrintHeading1("Global bundle adjustment");
  } else {
    PrintHeading1("Global bundle adjustment (Known rotation)");
  }
  mapper->AdjustGlobalBundle(options.Mapper(), custom_ba_options);
}

void IterativeGlobalRefinement(const GlobalMapperOptions& options,
                               GlobalMapper* mapper,
                               bool force_update_rotation) {
  PrintHeading1("Retriangulation");
  CompleteAndMergeTracks(options, mapper);
  size_t num_retri = mapper->Retriangulate(options.Triangulation());
  std::cout << "  => Retriangulated observations: " << num_retri << std::endl;

  for (int i = 0; i < options.ba_global_max_refinements; ++i) {
    const size_t num_observations =
        mapper->GetReconstruction().ComputeNumObservations();
    size_t num_changed_observations = 0;
    AdjustGlobalBundle(options, mapper, force_update_rotation);
    num_changed_observations += CompleteAndMergeTracks(options, mapper);
    num_changed_observations += FilterPoints(options, mapper);
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    size_t num_retri = 0;
    if (num_retri == 0 && changed < options.ba_global_max_refinement_change) {
      break;
    }
  }

  FilterImages(options, mapper);
}

size_t TriangulateImage(const GlobalMapperOptions& options,
                        const Image& image, GlobalMapper* mapper) {
  std::cout << "  => Continued observations: " << image.NumPoints3D()
            << std::endl;
  const size_t num_tris =
      mapper->TriangulateImage(options.Triangulation(), image.ImageId());
  std::cout << "  => Added observations: " << num_tris << std::endl;
  return num_tris;
}

void ExtractColors(const std::string& image_path, const image_t image_id,
                   Reconstruction* reconstruction) {
  if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
    std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                              reconstruction->Image(image_id).Name().c_str(),
                              image_path.c_str())
              << std::endl;
  }
}

bool LoadDatabaseToCache(const GlobalMapperOptions& options,
    const std::string& database_path,
    const std::unordered_set<std::string>& image_names,
    bool relative_pose,
    DatabaseCache* database_cache) {
  PrintHeading1("Loading database");

  Database database(database_path);
  Timer timer;
  timer.Start();
  const size_t min_num_matches = static_cast<size_t>(options.min_num_matches);
  database_cache->Load(database, min_num_matches, options.ignore_watermarks,
                       image_names, relative_pose, options.camera_path);
  std::cout << std::endl;
  timer.PrintMinutes();

  std::cout << std::endl;

  if (database_cache->NumImages() == 0) {
    std::cout << "WARNING: No images with matches found in the database."
              << std::endl
              << std::endl;
    return false;
  }

  return true;
}

}
