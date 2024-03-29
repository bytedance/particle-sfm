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

#ifndef COLMAP_SRC_GLOBAL_POSITION_ESTIMATOR_H_
#define COLMAP_SRC_GLOBAL_POSITION_ESTIMATOR_H_

// Modified from Theia SfM
// [LINK] https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/global_pose_estimation/position_estimator.h

#include <unordered_map>
#include <vector>
#include "base/correspondence_graph.h"

namespace colmap {

// A generic class defining the interface for global position estimation
// methods. These methods take in as input the (global/absolute) orientation
// estimated for each camera and pairwise translation directions between pairs
// of cameras. Additional information such as track/correspondences can also be
// passed in as needed, but those will be specific to the subclass
// implementation.
class PositionEstimator {
 public:
  PositionEstimator() {}
  virtual ~PositionEstimator() {}

  // Input the view pairs containing relative poses between matched
  // geometrically verified views, as well as the global (absolute) orientations
  // of the camera that were previously estimated.
  //
  // Returns true if the position estimation was a success, false if there was a
  // failure. If false is returned, the contents of positions are undefined.
  virtual bool EstimatePositions(
      const std::vector<ImagePair>& view_pairs,
      const std::unordered_map<image_t, Eigen::Vector4d>& orientation,
      std::unordered_map<image_t, Eigen::Vector3d>* positions) = 0;
};

}

#endif
