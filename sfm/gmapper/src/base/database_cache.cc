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

#include "base/database_cache.h"

#include <unordered_set>
#include <math.h>
#include <colmap/feature/utils.h>
#include <colmap/util/threading.h>
#include <colmap/util/timer.h>
#include <colmap/util/string.h>

#include "base/reconstruction.h"

namespace colmap {

namespace {
void ComputeTwoViewGeometry(struct TwoViewGeometry* two_view_geometry,
                            TwoViewGeometry::Options options,
                            const Camera* camera1, const Image* image1,
                            const Camera* camera2, const Image* image2) {
  std::vector<Eigen::Vector2d> points1;
  points1.reserve(image1->NumPoints2D());
  for (const auto& point : image1->Points2D()) {
    points1.push_back(point.XY());
  }

  std::vector<Eigen::Vector2d> points2;
  points2.reserve(image2->NumPoints2D());
  for (const auto& point : image2->Points2D()) {
    points2.push_back(point.XY());
  }

  two_view_geometry->EstimateRelativePose(*camera1, points1,
        *camera2, points2, options);
}
}

DatabaseCache::DatabaseCache() {}

void DatabaseCache::Load(const Database& database, const size_t min_num_matches,
                         const bool ignore_watermarks,
                         const std::unordered_set<std::string>& image_names,
                         const bool relative_pose,
                         const std::string& camera_path) {
  //////////////////////////////////////////////////////////////////////////////
  // Load cameras
  //////////////////////////////////////////////////////////////////////////////

  Timer timer;

  timer.Start();
  std::cout << "Loading cameras..." << std::flush;

  if (camera_path.empty()) {
    const std::vector<class Camera> cameras = database.ReadAllCameras();
    cameras_.reserve(cameras.size());
    for (const auto& camera : cameras) {
      cameras_.emplace(camera.CameraId(), camera);
    }
  } else {
    Reconstruction temp_reconstruction;
    bool result = temp_reconstruction.Read(camera_path);
    CHECK(result);
    if (result) {
      cameras_ = temp_reconstruction.Cameras();
    }
  }

  std::cout << StringPrintf(" %d in %.3fs", cameras_.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Load matches
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading matches..." << std::flush;

  std::vector<image_pair_t> image_pair_ids;
  std::vector<struct TwoViewGeometry> two_view_geometries;
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

  std::cout << StringPrintf(" %d in %.3fs", image_pair_ids.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  auto UseInlierMatchesCheck = [min_num_matches, ignore_watermarks](
                                   const struct TwoViewGeometry& two_view_geometry) {
    return static_cast<size_t>(two_view_geometry.inlier_matches.size()) >=
               min_num_matches &&
           (!ignore_watermarks ||
            two_view_geometry.config != TwoViewGeometry::WATERMARK);
  };
  size_t num_ignored_image_pairs = 0;

  for (size_t i = 0; i < image_pair_ids.size(); ++i) {
    if (!UseInlierMatchesCheck(two_view_geometries[i])) {
      image_pair_ids[i] = kInvalidImagePairId;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Load images
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading images..." << std::endl;

  std::unordered_set<image_t> image_ids;

  {
    const std::vector<class Image> images = database.ReadAllImages();

    // Determines for which images data should be loaded.
    if (image_names.empty()) {
      for (const auto& image : images) {
        image_ids.insert(image.ImageId());
      }
    } else {
      for (const auto& image : images) {
        if (image_names.count(image.Name()) > 0) {
          image_ids.insert(image.ImageId());
        }
      }
    }

    // Collect all images that are connected in the correspondence graph.
    std::unordered_set<image_t> connected_image_ids;
    connected_image_ids.reserve(image_ids.size());
    for (size_t i = 0; i < image_pair_ids.size(); ++i) {
      if (image_pair_ids[i] == kInvalidImagePairId) {
        continue;
      }
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
      if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
        connected_image_ids.insert(image_id1);
        connected_image_ids.insert(image_id2);
      }
    }

    // Load images with correspondences and discard images without
    // correspondences, as those images are useless for SfM.
    images_.reserve(connected_image_ids.size());
    for (const auto& image : images) {
      if (image_ids.count(image.ImageId()) > 0 &&
          connected_image_ids.count(image.ImageId()) > 0) {
        images_.emplace(image.ImageId(), image);
        const FeatureKeypoints keypoints =
            database.ReadKeypoints(image.ImageId());
        const std::vector<Eigen::Vector2d> points =
            FeatureKeypointsToPointsVector(keypoints);
        images_[image.ImageId()].SetPoints2D(points);
      }
    }

    std::cout << StringPrintf("Loaded %d in %.3fs (connected %d)", images.size(),
                              timer.ElapsedSeconds(),
                              connected_image_ids.size())
              << std::endl;

  }

  two_view_geometries_.reserve(two_view_geometries.size());
  for (size_t i = 0; i < image_pair_ids.size(); ++i) {
    if (image_pair_ids[i] != kInvalidImagePairId && two_view_geometries[i].config != TwoViewGeometry::UNDEFINED) {
      two_view_geometries_.emplace(image_pair_ids[i], two_view_geometries[i]);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Optimize the poses
  //////////////////////////////////////////////////////////////////////////////

  if (relative_pose) {
    timer.Restart();
    std::cout << "Optimizing the poses..." << std::endl;
    std::unique_ptr<ThreadPool> pool(new ThreadPool());

    for (auto& two_view_geometry : two_view_geometries_) {
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(two_view_geometry.first, &image_id1, &image_id2);

      const auto& image1 = Image(image_id1);
      const auto& camera1 = Camera(image1.CameraId());
      const auto& image2 = Image(image_id2);
      const auto& camera2 = Camera(image2.CameraId());
      TwoViewGeometry::Options options;
      pool->AddTask(ComputeTwoViewGeometry,
                    &two_view_geometry.second, options,
                    &camera1, &image1,
                    &camera2, &image2);
    }
    pool->Wait();

    std::cout << StringPrintf("Optimized in %.3fs (ignored %d)", timer.ElapsedSeconds(),
                              num_ignored_image_pairs)
              << std::endl;
  }

  std::vector<image_pair_t> removed_pairs;
  for (const auto& two_view_geometry : two_view_geometries_) {
    if (two_view_geometry.second.config == TwoViewGeometry::UNDEFINED) {
      removed_pairs.push_back(two_view_geometry.first);
      // LOG(INFO) << "Remove: " << two_view_geometry.first;
    }
  }
  for (auto key : removed_pairs) {
    two_view_geometries_.erase(key);
  }
}

}  // namespace colmap
