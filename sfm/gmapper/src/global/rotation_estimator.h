// Copyright (C) 2016 The Regents of the University of California (Regents).
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

#ifndef COLMAP_SRC_GLOBAL_ROTATION_ESTIMATOR_H_
#define COLMAP_SRC_GLOBAL_ROTATION_ESTIMATOR_H_

// Modified from Theia SfM
// [LINK] https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/global_pose_estimation/rotation_estimator.h

#include <unordered_map>
#include <vector>
#include "base/correspondence_graph.h"

namespace colmap {

// A generic class defining the interface for global rotation estimation
// methods. These methods take in as input the relative pairwise orientations
// and output estimates for the global orientation of each view.
class RotationEstimator {
 public:
  RotationEstimator() {}
  virtual ~RotationEstimator() {}
  // Input the view pairs containing relative rotations between matched
  // geometrically verified views and outputs a rotation estimate for each view.
  //
  // Returns true if the rotation estimation was a success, false if there was a
  // failure. If false is returned, the contents of rotations are undefined.
  virtual bool EstimateRotations(
      const std::vector<ImagePair>& view_pairs,
      std::unordered_map<image_t, Eigen::Vector4d>* rotations) = 0;
};

}

#endif
