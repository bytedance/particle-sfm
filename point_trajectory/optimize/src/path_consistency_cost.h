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

#ifndef OPTIMIZE_PATH_CONSISTENCY_COST_H_
#define OPTIMIZE_PATH_CONSISTENCY_COST_H_

#include "linear_interpolation.h"
#include "ceres/cubic_interpolation.h"

namespace particlesfm {

class PathConsistencyError {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Grid = ceres::Grid2D<double, 2>;
  using Interpolator = ceres::BiLinearInterpolator<Grid>;
 
  PathConsistencyError(const Interpolator& flow12_map,
                     const Eigen::Matrix<double, 2, 1> uv_ref1,
                     const Eigen::Matrix<double, 2, 1> uv_ref2,
                     const Eigen::Matrix<double, 1, 1> ref2_scale)
    : flow12_map_(flow12_map),
      uv_ref1_(uv_ref1),
      uv_ref2_(uv_ref2),
      ref2_scale_(ref2_scale){
  }

  template <typename T>
  inline bool operator()(const T* const uv12_ptr, 
                            T* residuals_ptr) const {
    
    Eigen::Matrix<T, 4, 1> uv12(uv12_ptr);
    T flow_target[2];
    flow12_map_.Evaluate(uv12[1], uv12[0], flow_target);
    // flow01 loss
    residuals_ptr[0] = (uv12[0] - uv_ref1_[0]);
    residuals_ptr[1] = (uv12[1] - uv_ref1_[1]);
    // flow02 loss
    residuals_ptr[2] = (uv12[2] - uv_ref2_[0]) * ref2_scale_[0];
    residuals_ptr[3] = (uv12[3] - uv_ref2_[1]) * ref2_scale_[0];
    // flow12 loss
    residuals_ptr[4] = (uv12[2] - uv12[0]) - flow_target[0];
    residuals_ptr[5] = (uv12[3] - uv12[1]) - flow_target[1];
    return true;
  }

 private:
  const Interpolator& flow12_map_;
  const Eigen::Matrix<double, 2, 1> uv_ref1_;
  const Eigen::Matrix<double, 2, 1> uv_ref2_;
  const Eigen::Matrix<double, 1, 1> ref2_scale_;
};

}  // namespace particlesfm 

#endif

