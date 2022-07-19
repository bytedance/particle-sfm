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

#include "global/triplet_util.h"

#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <set>
#include <vector>

#include <colmap/base/pose.h>
#include <colmap/base/projection.h>
#include <colmap/base/triangulation.h>
#include <colmap/util/misc.h>
#include <theia/math/graph/triplet_extractor.h>

#include "base/reconstruction.h"

namespace colmap {
namespace {

using Eigen::Vector2d;
using Eigen::Vector3d;

// Triangulate the point and return the depth of the point relative to each
// view. Returns true if the depth was recovered successfully and false if the
// point could not be triangulated.
bool GetTriangulatedPointDepths(const ImagePair& info,
                                double min_tri_angle,
                                const Vector2d& feature1,
                                const Vector2d& feature2,
                                double* depth1,
                                double* depth2) {
  const double kMinTriangulationAngle = DegToRad(min_tri_angle);
  Eigen::Matrix3x4d proj_matrix1 = ComposeProjectionMatrix(ComposeIdentityQuaternion(),
      Eigen::Vector3d::Zero());
  Eigen::Vector3d proj_center1(0, 0, 0);
  Eigen::Matrix3x4d proj_matrix2 = ComposeProjectionMatrix(info.qvec, info.tvec);
  Eigen::Vector3d proj_center2 = ProjectionCenterFromPose(info.qvec, info.tvec);

  const Eigen::Vector3d point3d = TriangulatePoint(proj_matrix1, proj_matrix2,
      feature1, feature2);
  if (!HasPointPositiveDepth(proj_matrix1, point3d)
      || !HasPointPositiveDepth(proj_matrix2, point3d)
      || CalculateTriangulationAngle(proj_center1, proj_center2,
        point3d) < kMinTriangulationAngle) {
    return false;
  }

  // Compute depths.
  *depth1 = point3d.norm();
  *depth2 = (QuaternionRotatePoint(info.qvec, point3d) + info.tvec).norm();
  return true;
}

}  // namespace

int ComputeTripletBaselineRatios(const ImageTriplet& triplet,
                                  double min_tri_angle,
                                  const std::vector<Eigen::Vector2d>& feature1,
                                  const std::vector<Eigen::Vector2d>& feature2,
                                  const std::vector<Eigen::Vector2d>& feature3,
                                  Eigen::Vector3d* baseline) {
  CHECK_NOTNULL(baseline)->setZero();
  CHECK_EQ(feature1.size(), feature2.size())
      << "The feature containers must be the same size when computing the "
         "triplet baseline ratios.";
  CHECK_EQ(feature1.size(), feature3.size())
      << "The feature containers must be the same size when computing the "
         "triplet baseline ratios.";

  Eigen::Vector4d point12, point13, point23;
  double depth1_12, depth2_12, depth1_13, depth3_13, depth2_23, depth3_23;

  std::vector<double> baseline2, baseline3;
  baseline2.reserve(feature2.size());
  baseline3.reserve(feature3.size());
  for (int i = 0; i < feature1.size(); i++) {
    if (!GetTriangulatedPointDepths(triplet.info_one_two,
                                    min_tri_angle,
                                    feature1[i],
                                    feature2[i],
                                    &depth1_12, &depth2_12)) {
      continue;
    }

    // Compute triangulation from views 1, 3.
    if (!GetTriangulatedPointDepths(triplet.info_one_three,
                                    min_tri_angle,
                                    feature1[i],
                                    feature3[i],
                                    &depth1_13, &depth3_13)) {
      continue;
    }

    // Compute triangulation from views 2, 3.
    if (!GetTriangulatedPointDepths(triplet.info_two_three,
                                    min_tri_angle,
                                    feature2[i],
                                    feature3[i],
                                    &depth2_23, &depth3_23)) {
      continue;
    }

    baseline2.emplace_back(depth1_12 / depth1_13);
    baseline3.emplace_back(depth2_12 / depth2_23);
  }

  if (baseline2.size() == 0) {
    /*
    VLOG(2) << "Could not compute the triplet baseline ratios. An inusfficient "
               "number of well-constrained 3D points were observed.";
    */
    return 0;
  }

  // Take the median as the baseline ratios.
  const int mid_index = baseline2.size() / 2;
  std::nth_element(baseline2.begin(),
                   baseline2.begin() + mid_index,
                   baseline2.end());
  std::nth_element(baseline3.begin(),
                   baseline3.begin() + mid_index,
                   baseline3.end());
  *baseline = Vector3d(1.0, baseline2[mid_index], baseline3[mid_index]);
  return baseline2.size();
}

std::vector<ImageIdTriplet> GetLargetConnectedTripletGraph(
    const std::vector<ImagePair>& view_pairs) {
  static const int kLargestCCIndex = 0;

  // Get a list of all edges in the view graph.
  std::unordered_set<std::pair<image_t, image_t>> view_id_pairs;
  view_id_pairs.reserve(view_pairs.size());
  for (const auto& view_pair : view_pairs) {
    view_id_pairs.emplace(view_pair.image_id1, view_pair.image_id2);
  }

  // Extract connected triplets.
  theia::TripletExtractor<image_t> extractor;
  std::vector<std::vector<ImageIdTriplet> > triplets;
  CHECK(extractor.ExtractTriplets(view_id_pairs, &triplets));
  CHECK_GT(triplets.size(), 0);
  return triplets[kLargestCCIndex];
}

std::vector<point3D_t> FindCommonTracksInViews(
    const Reconstruction& reconstruction, const std::vector<image_t>& view_ids) {
  std::set<point3D_t> point_set;
  if (!view_ids.empty()) {
    const auto points2D = reconstruction.Image(view_ids[0]).Points2D();
    for (const auto& pt: points2D) {
        if (pt.HasPoint3D())
            point_set.insert(pt.Point3DId());
    } 
    for (size_t i = 1; i < view_ids.size(); ++i) {
      std::set<point3D_t> single_set;
      const auto& points2D_i = reconstruction.Image(view_ids[i]).Points2D();
      for (const auto& pt: points2D_i) {
          if (pt.HasPoint3D())
              single_set.insert(pt.Point3DId());
      }
      std::set<point3D_t> new_point_set;
      std::set_intersection(point_set.begin(), point_set.end(),
          single_set.begin(), single_set.end(), std::inserter(new_point_set, new_point_set.begin()));
      std::swap(point_set, new_point_set);
    }
  }
  return std::vector<point3D_t>(point_set.begin(), point_set.end());
}

}
