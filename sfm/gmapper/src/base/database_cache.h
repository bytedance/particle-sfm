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

#ifndef COLMAP_SRC_BASE_DATABASE_CACHE_H_
#define COLMAP_SRC_BASE_DATABASE_CACHE_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include <colmap/base/camera.h>
#include <colmap/base/camera_models.h>
#include <colmap/base/database.h>
#include <colmap/base/image.h>
#include <colmap/util/alignment.h>
#include <colmap/util/types.h>

#include "estimators/two_view_geometry.h"

namespace colmap {

// A class that caches the contents of the database in memory, used to quickly
// create new reconstruction instances when multiple models are reconstructed.
class DatabaseCache {
 public:
  DatabaseCache();

  // Get number of objects.
  inline size_t NumCameras() const;
  inline size_t NumImages() const;
  inline size_t NumTwoViewGeometries() const;

  // Get specific objects.
  inline class Camera& Camera(const camera_t camera_id);
  inline const class Camera& Camera(const camera_t camera_id) const;
  inline class Image& Image(const image_t image_id);
  inline const class Image& Image(const image_t image_id) const;
  inline struct TwoViewGeometry& TwoViewGeometry(const image_pair_t image_pair_id);
  inline const struct TwoViewGeometry& TwoViewGeometry(const image_pair_t image_pair_id) const;

  // Get all objects.
  inline const EIGEN_STL_UMAP(camera_t, class Camera) & Cameras() const;
  inline const EIGEN_STL_UMAP(image_t, class Image) & Images() const;
  inline const EIGEN_STL_UMAP(image_pair_t, struct TwoViewGeometry) & TwoViewGeometries() const;

  // Check whether specific object exists.
  inline bool ExistsCamera(const camera_t camera_id) const;
  inline bool ExistsImage(const image_t image_id) const;
  inline bool ExistsTwoViewGeometry(const image_pair_t image_pair_id) const;

  // Load cameras, images, features, and matches from database.
  //
  // @param database              Source database from which to load data.
  // @param min_num_matches       Only load image pairs with a minimum number
  //                              of matches.
  // @param ignore_watermarks     Whether to ignore watermark image pairs.
  // @param image_names           Whether to use only load the data for a subset
  //                              of the images. All images are used if empty.
  void Load(const Database& database, const size_t min_num_matches,
            const bool ignore_watermarks,
            const std::unordered_set<std::string>& image_names,
            const bool relative_pose = false,
            const std::string& camera_path = "");

 private:
  EIGEN_STL_UMAP(camera_t, class Camera) cameras_;
  EIGEN_STL_UMAP(image_t, class Image) images_;
  EIGEN_STL_UMAP(image_pair_t, struct TwoViewGeometry) two_view_geometries_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t DatabaseCache::NumCameras() const { return cameras_.size(); }
size_t DatabaseCache::NumImages() const { return images_.size(); }
size_t DatabaseCache::NumTwoViewGeometries() const { return two_view_geometries_.size(); }

class Camera& DatabaseCache::Camera(const camera_t camera_id) {
  return cameras_.at(camera_id);
}

const class Camera& DatabaseCache::Camera(const camera_t camera_id) const {
  return cameras_.at(camera_id);
}

class Image& DatabaseCache::Image(const image_t image_id) {
  return images_.at(image_id);
}

const class Image& DatabaseCache::Image(const image_t image_id) const {
  return images_.at(image_id);
}

struct TwoViewGeometry& DatabaseCache::TwoViewGeometry(const image_pair_t image_pair_id) {
  return two_view_geometries_.at(image_pair_id);
}

const struct TwoViewGeometry& DatabaseCache::TwoViewGeometry(const image_pair_t image_pair_id) const {
  return two_view_geometries_.at(image_pair_id);
}

const EIGEN_STL_UMAP(camera_t, class Camera) & DatabaseCache::Cameras() const {
  return cameras_;
}

const EIGEN_STL_UMAP(image_t, class Image) & DatabaseCache::Images() const {
  return images_;
}

const EIGEN_STL_UMAP(image_pair_t, struct TwoViewGeometry) & DatabaseCache::TwoViewGeometries() const {
  return two_view_geometries_;
}

bool DatabaseCache::ExistsCamera(const camera_t camera_id) const {
  return cameras_.find(camera_id) != cameras_.end();
}

bool DatabaseCache::ExistsImage(const image_t image_id) const {
  return images_.find(image_id) != images_.end();
}

bool DatabaseCache::ExistsTwoViewGeometry(const image_pair_t image_pair_id) const {
  return two_view_geometries_.find(image_pair_id) != two_view_geometries_.end();
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_DATABASE_CACHE_H_
