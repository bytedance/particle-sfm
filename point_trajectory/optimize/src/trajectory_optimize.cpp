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

#include "trajectory_optimize.h"

namespace particlesfm {

double ComputeCost(ceres::Problem& problem) {
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.apply_loss_function = false;
  double cost = 0.0;
  problem.Evaluate(eval_options, &cost, nullptr, nullptr, nullptr);
  return std::sqrt(cost / problem.NumResiduals());
}


Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> optimize_location(
                      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> uv12, 
                      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> uv_ref1,
                      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> uv_ref2,
                      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ref2_scale,
                      py::array_t<double> flow12_map_arr,
                      const int total_num,
                      const int width, const int height) {
  // const parameters
  const double bound = 3;
  //py::buffer_info buf1 = uv_arr.request(), buf2 = ref_arr.request();
  py::buffer_info buf3 = flow12_map_arr.request();
  // copy data
  std::vector< Eigen::Matrix<double, 4, 1> > uv12_list;
  for (int i = 0; i < total_num; i++){
    Eigen::Matrix<double, 4, 1> uv12_row(uv12.row(i));
    uv12_list.push_back(uv12_row);
  }
  
  
  // Define ceres problem
  ceres::Problem problem;
  ceres::LossFunction* robust_loss = new ceres::TrivialLoss();
  ceres::Grid2D<double, 2> grid((double*)buf3.ptr, 0, height, 0, width);
  ceres::BiLinearInterpolator<ceres::Grid2D<double, 2>> interpolator(grid);

  for (int i = 0; i < total_num; i++) {
      Eigen::Matrix<double, 2, 1> uv_ref1_row(uv_ref1.row(i));
      Eigen::Matrix<double, 2, 1> uv_ref2_row(uv_ref2.row(i));
      Eigen::Matrix<double, 1, 1> ref2_scale_row(ref2_scale.row(i));
      PathConsistencyError* cost_function = new PathConsistencyError(
          interpolator, uv_ref1_row, uv_ref2_row, ref2_scale_row);
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<PathConsistencyError, 6, 4>(cost_function),
          robust_loss,
          uv12_list[i].data());
      //problem.SetParameterUpperBound(uv12_list[i].data(), 0, uv_list[i][0] + bound);
      //problem.SetParameterLowerBound(uv12_list[i].data(), 0, uv_list[i][0] - bound);
      //problem.SetParameterUpperBound(uv12_list[i].data(), 1, uv_list[i][1] + bound);
      //problem.SetParameterLowerBound(uv12_list[i].data(), 1, uv_list[i][1] - bound);
  }

  double cost_before = ComputeCost(problem);

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.max_num_iterations = 200;
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.trust_region_strategy_type = ceres::DOGLEG;
  solver_options.num_threads = 8;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  // logging
  // printf("Residual number: %d\n", problem.NumResiduals());
  // printf("Summary:\n%s\n", summary.FullReport().c_str());
  // double cost_after = ComputeCost(problem);
  // printf("Flow loop cost: %lf -> %lf\n", cost_before, cost_after);
  
  // return value
  Eigen::MatrixXd res(total_num, 4);
  for (int i = 0; i < total_num; i++){
    res.row(i) = uv12_list[i];
  }
  return res;
}

} // namespace particlesfm

