// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
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
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "util/option_manager.h"

#include <boost/filesystem/operations.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <colmap/base/image_reader.h>
#include <colmap/util/random.h>
#include <colmap/util/misc.h>

#include "controllers/global_mapper.h"

namespace config = boost::program_options;

namespace colmap {

OptionManager::OptionManager() {
  database_path.reset(new std::string());
  image_path.reset(new std::string());
  global_mapper.reset(new GlobalMapperOptions());

  Reset();

  desc_->add_options()("help,h", "");

  AddRandomOptions();
  AddLogOptions();
}

void OptionManager::AddAllOptions() {
  AddLogOptions();
  AddRandomOptions();
  AddDatabaseOptions();
  AddImageOptions();
  AddGlobalMapperOptions();
}

void OptionManager::AddLogOptions() {
  if (added_log_options_) {
    return;
  }
  added_log_options_ = true;

  // AddAndRegisterDefaultOption("log_to_stderr", &FLAGS_logtostderr);
  // AddAndRegisterDefaultOption("log_level", &FLAGS_v);
}

void OptionManager::AddRandomOptions() {
  if (added_random_options_) {
    return;
  }
  added_random_options_ = true;

  AddAndRegisterDefaultOption("random_seed", &kDefaultPRNGSeed);
}

void OptionManager::AddDatabaseOptions() {
  if (added_database_options_) {
    return;
  }
  added_database_options_ = true;

  AddAndRegisterRequiredOption("database_path", database_path.get());
}

void OptionManager::AddImageOptions() {
  if (added_image_options_) {
    return;
  }
  added_image_options_ = true;

  AddAndRegisterRequiredOption("image_path", image_path.get());
}

void OptionManager::AddGlobalMapperOptions() {
  if (added_global_mapper_options_) {
    return;
  }
  added_global_mapper_options_ = true;

  AddAndRegisterDefaultOption("GlobalMapper.min_num_matches",
                              &global_mapper->min_num_matches);
  AddAndRegisterDefaultOption("GlobalMapper.ignore_watermarks",
                              &global_mapper->ignore_watermarks);
  AddAndRegisterDefaultOption("GlobalMapper.extract_colors", &global_mapper->extract_colors);
  AddAndRegisterDefaultOption("GlobalMapper.min_track_length", &global_mapper->min_track_length);
  AddAndRegisterDefaultOption("GlobalMapper.max_track_length", &global_mapper->max_track_length);
  AddAndRegisterDefaultOption("GlobalMapper.num_threads", &global_mapper->num_threads);
  AddAndRegisterDefaultOption("GlobalMapper.ba_refine_focal_length",
                              &global_mapper->ba_refine_focal_length);
  AddAndRegisterDefaultOption("GlobalMapper.ba_refine_principal_point",
                              &global_mapper->ba_refine_principal_point);
  AddAndRegisterDefaultOption("GlobalMapper.ba_refine_extra_params",
                              &global_mapper->ba_refine_extra_params);
  AddAndRegisterDefaultOption(
      "GlobalMapper.ba_min_num_residuals_for_multi_threading",
      &global_mapper->ba_min_num_residuals_for_multi_threading);
  AddAndRegisterDefaultOption("GlobalMapper.ba_global_max_num_iterations",
                              &global_mapper->ba_global_max_num_iterations);
  AddAndRegisterDefaultOption("GlobalMapper.ba_global_max_refinements",
                              &global_mapper->ba_global_max_refinements);
  AddAndRegisterDefaultOption("GlobalMapper.ba_global_max_refinement_change",
                              &global_mapper->ba_global_max_refinement_change);
  AddAndRegisterDefaultOption("GlobalMapper.fix_prior_rotation",
                              &global_mapper->ba_fix_prior_rotation);
  AddAndRegisterDefaultOption("GlobalMapper.camera_path",
                              &global_mapper->camera_path);

  // GlobalMapper.
  AddAndRegisterDefaultOption("GlobalMapper.filter_with_1dsfm",
      &global_mapper->mapper.filter_with_1dsfm);
  AddAndRegisterDefaultOption("GlobalMapper.translation_filter_num_iterations",
      &global_mapper->mapper.translation_filter_num_iterations);
  AddAndRegisterDefaultOption("GlobalMapper.translation_filter_tolerance",
      &global_mapper->mapper.translation_filter_tolerance);
  AddAndRegisterDefaultOption("GlobalMapper.filter_max_reproj_error",
                              &global_mapper->mapper.filter_max_reproj_error);
  AddAndRegisterDefaultOption("GlobalMapper.filter_min_tri_angle",
                              &global_mapper->mapper.filter_min_tri_angle);

  // IncrementalTriangulator.
  AddAndRegisterDefaultOption("GlobalMapper.tri_max_transitivity",
                              &global_mapper->triangulation.max_transitivity);
  AddAndRegisterDefaultOption("GlobalMapper.tri_create_max_angle_error",
                              &global_mapper->triangulation.create_max_angle_error);
  AddAndRegisterDefaultOption("GlobalMapper.tri_continue_max_angle_error",
                              &global_mapper->triangulation.continue_max_angle_error);
  AddAndRegisterDefaultOption("GlobalMapper.tri_merge_max_reproj_error",
                              &global_mapper->triangulation.merge_max_reproj_error);
  AddAndRegisterDefaultOption("GlobalMapper.tri_complete_max_reproj_error",
                              &global_mapper->triangulation.complete_max_reproj_error);
  AddAndRegisterDefaultOption("GlobalMapper.tri_complete_max_transitivity",
                              &global_mapper->triangulation.complete_max_transitivity);
  AddAndRegisterDefaultOption("GlobalMapper.tri_re_max_angle_error",
                              &global_mapper->triangulation.re_max_angle_error);
  AddAndRegisterDefaultOption("GlobalMapper.tri_re_min_ratio",
                              &global_mapper->triangulation.re_min_ratio);
  AddAndRegisterDefaultOption("GlobalMapper.tri_re_max_trials",
                              &global_mapper->triangulation.re_max_trials);
  AddAndRegisterDefaultOption("GlobalMapper.tri_min_angle",
                              &global_mapper->triangulation.min_angle);
  AddAndRegisterDefaultOption("GlobalMapper.tri_ignore_two_view_tracks",
                              &global_mapper->triangulation.ignore_two_view_tracks);

  // Rotation
  AddAndRegisterDefaultOption("GlobalMapper.max_num_l1_iterations",
      &global_mapper->rotation.max_num_l1_iterations);
  AddAndRegisterDefaultOption("GlobalMapper.l1_step_convergence_threshold",
      &global_mapper->rotation.l1_step_convergence_threshold);
  AddAndRegisterDefaultOption("GlobalMapper.max_num_irls_iterations",
      &global_mapper->rotation.max_num_irls_iterations);
  AddAndRegisterDefaultOption("GlobalMapper.irls_step_convergence_threshold",
      &global_mapper->rotation.irls_step_convergence_threshold);
  AddAndRegisterDefaultOption("GlobalMapper.irls_loss_parameter_sigma",
      &global_mapper->rotation.irls_loss_parameter_sigma);
  AddAndRegisterDefaultOption("GlobalMapper.rotation_filter_max_degrees",
      &global_mapper->rotation.rotation_filter_max_degrees);

  // Position
  AddAndRegisterDefaultOption("GlobalMapper.position_method",
      &global_mapper->position.method);
  AddAndRegisterDefaultOption("GlobalMapper.nonlinear_max_num_iterations",
      &global_mapper->position.nonlinear_options.max_num_iterations);
  AddAndRegisterDefaultOption("GlobalMapper.nonlinear_robust_loss_width",
      &global_mapper->position.nonlinear_options.robust_loss_width);
  AddAndRegisterDefaultOption("GlobalMapper.nonlinear_min_num_points_per_view",
      &global_mapper->position.nonlinear_options.min_num_points_per_view);
  AddAndRegisterDefaultOption("GlobalMapper.nonlinear_point_to_camera_weight",
      &global_mapper->position.nonlinear_options.point_to_camera_weight);
  AddAndRegisterDefaultOption("GlobalMapper.linear_max_power_iterations",
      &global_mapper->position.linear_options.max_power_iterations);
  AddAndRegisterDefaultOption("GlobalMapper.linear_eigensolver_threshold",
      &global_mapper->position.linear_options.eigensolver_threshold);
  AddAndRegisterDefaultOption("GlobalMapper.linear_min_tri_angle",
      &global_mapper->position.linear_options.min_tri_angle);
  AddAndRegisterDefaultOption("GlobalMapper.lud_max_num_iterations",
      &global_mapper->position.lud_options.max_num_iterations);
  AddAndRegisterDefaultOption("GlobalMapper.lud_max_num_reweighted_iterations",
      &global_mapper->position.lud_options.max_num_reweighted_iterations);
  AddAndRegisterDefaultOption("GlobalMapper.lud_convergence_criterion",
      &global_mapper->position.lud_options.convergence_criterion);
  AddAndRegisterDefaultOption("GlobalMapper.lud_min_tri_angle",
      &global_mapper->position.lud_options.min_tri_angle);
  AddAndRegisterDefaultOption("GlobalMapper.lud_use_scale_constraints",
      &global_mapper->position.lud_options.use_scale_constraints);
  AddAndRegisterDefaultOption("GlobalMapper.lud_max_num_points",
      &global_mapper->position.lud_options.max_num_points);
}

void OptionManager::Reset() {
  // FLAGS_logtostderr = false;
  // FLAGS_v = 2;

  const bool kResetPaths = true;
  ResetOptions(kResetPaths);

  desc_.reset(new boost::program_options::options_description());

  options_bool_.clear();
  options_int_.clear();
  options_double_.clear();
  options_string_.clear();

  added_log_options_ = false;
  added_random_options_ = false;
  added_database_options_ = false;
  added_image_options_ = false;
  added_global_mapper_options_ = false;
}

void OptionManager::ResetOptions(const bool reset_paths) {
  if (reset_paths) {
    *database_path = "";
    *image_path = "";
  }
  *global_mapper = GlobalMapperOptions();
}

bool OptionManager::Check() {
  bool success = true;

  if (added_database_options_) {
    const auto database_parent_path = GetParentDir(*database_path);
    success = success && CHECK_OPTION_IMPL(!ExistsDir(*database_path)) &&
              CHECK_OPTION_IMPL(database_parent_path == "" ||
                                ExistsDir(database_parent_path));
  }

  if (added_image_options_)
    success = success && CHECK_OPTION_IMPL(ExistsDir(*image_path));

  if (global_mapper) success = success && global_mapper->Check();

  return success;
}

void OptionManager::Parse(const int argc, char** argv) {
  config::variables_map vmap;

  try {
    config::store(config::parse_command_line(argc, argv, *desc_), vmap);

    if (vmap.count("help")) {
      std::cout << StringPrintf("GCOLMAP")
                << std::endl
                << std::endl;
      std::cout << *desc_ << std::endl;
      exit(EXIT_SUCCESS);
    }

    vmap.notify();
  } catch (std::exception& exc) {
    std::cerr << "ERROR: Failed to parse options - " << exc.what() << "."
              << std::endl;
    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "ERROR: Failed to parse options for unknown reason."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!Check()) {
    std::cerr << "ERROR: Invalid options provided." << std::endl;
    exit(EXIT_FAILURE);
  }
}

bool OptionManager::Read(const std::string& path) {
  config::variables_map vmap;

  if (!ExistsFile(path)) {
    std::cout << "ERROR: Configuration file does not exist." << std::endl;
    return false;
  }

  try {
    std::ifstream file(path);
    CHECK(file.is_open()) << path;
    config::store(config::parse_config_file(file, *desc_), vmap);
    vmap.notify();
  } catch (std::exception& e) {
    std::cout << "ERROR: Failed to parse options " << e.what() << "."
              << std::endl;
    return false;
  } catch (...) {
    std::cout << "ERROR: Failed to parse options for unknown reason."
              << std::endl;
    return false;
  }

  return Check();
}

bool OptionManager::ReRead(const std::string& path) {
  Reset();
  AddAllOptions();
  return Read(path);
}

void OptionManager::Write(const std::string& path) const {
  boost::property_tree::ptree pt;

  // First, put all options without a section and then those with a section.
  // This is necessary as otherwise older Boost versions will write the
  // options without a section in between other sections and therefore
  // the errors will be assigned to the wrong section if read later.

  for (const auto& option : options_bool_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_int_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_double_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_string_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_bool_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_int_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_double_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_string_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  boost::property_tree::write_ini(path, pt);
}

}  // namespace colmap
