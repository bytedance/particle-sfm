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

#ifndef OPTIMIZE_LINEAR_INTERPOLATION_H_
#define OPTIMIZE_LINEAR_INTERPOLATION_H_

#include "Eigen/Core"
#include "ceres/internal/port.h"
#include "glog/logging.h"

namespace ceres {

// p0 = f(0)
// p1 = f(1)
template <int kDataDimension>
void LinearInterpolate(const Eigen::Matrix<double, kDataDimension, 1>& p0,
                       const Eigen::Matrix<double, kDataDimension, 1>& p1,
                       const double x,
                       double* f,
                       double* dfdx) {
  typedef Eigen::Matrix<double, kDataDimension, 1> VType;

  if (f != NULL) {
    Eigen::Map<VType>(f, kDataDimension) = (1 - x) * p0 + x * p1;
    ;
  }

  if (dfdx != NULL) {
    Eigen::Map<VType>(dfdx, kDataDimension) = p1 - p0;
  }
}

template <typename Grid>
class LinearInterpolator {
 public:
  explicit LinearInterpolator(const Grid& grid) : grid_(grid) {
    // The + casts the enum into an int before doing the
    // comparison. It is needed to prevent
    // "-Wunnamed-type-template-args" related errors.
    CHECK_GE(+Grid::DATA_DIMENSION, 1);
  }

  void Evaluate(double x, double* f, double* dfdx) const {
    const int n = std::floor(x);
    Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> p0, p1;
    grid_.GetValue(n, p0.data());
    grid_.GetValue(n + 1, p1.data());
    LinearInterpolate<Grid::DATA_DIMENSION>(p0, p1, x - n, f, dfdx);
  }

  // The following two Evaluate overloads are needed for interfacing
  // with automatic differentiation. The first is for when a scalar
  // evaluation is done, and the second one is for when Jets are used.
  void Evaluate(const double& x, double* f) const {
    Evaluate(x, f, NULL);
  }

  template <typename JetT>
  void Evaluate(const JetT& x, JetT* f) const {
    double fx[Grid::DATA_DIMENSION], dfdx[Grid::DATA_DIMENSION];
    Evaluate(x.a, fx, dfdx);
    for (int i = 0; i < Grid::DATA_DIMENSION; ++i) {
      f[i].a = fx[i];
      f[i].v = dfdx[i] * x.v;
    }
  }

 private:
  const Grid& grid_;
};

template <typename Grid>
class BiLinearInterpolator {
 public:
  explicit BiLinearInterpolator(const Grid& grid) : grid_(grid) {
    // The + casts the enum into an int before doing the
    // comparison. It is needed to prevent
    // "-Wunnamed-type-template-args" related errors.
    CHECK_GE(+Grid::DATA_DIMENSION, 1);
  }

  // Evaluate the interpolated function value and/or its
  // derivative. Returns false if r or c is out of bounds.
  void Evaluate(double r, double c, double* f, double* dfdr, double* dfdc) const {
    const int row = std::floor(r);
    const int col = std::floor(c);

    Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> p0, p1;

    // Interpolate along each of the four rows, evaluating the function
    // value and the horizontal derivative in each row.
    Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> f0, f1;
    Eigen::Matrix<double, Grid::DATA_DIMENSION, 1> df0dc, df1dc;

    grid_.GetValue(row, col, p0.data());
    grid_.GetValue(row, col + 1, p1.data());
    LinearInterpolate<Grid::DATA_DIMENSION>(p0, p1, c - col, f0.data(), df0dc.data());

    grid_.GetValue(row + 1, col, p0.data());
    grid_.GetValue(row + 1, col + 1, p1.data());
    LinearInterpolate<Grid::DATA_DIMENSION>(p0, p1, c - col, f1.data(), df1dc.data());

    // Interpolate vertically the interpolated value from each row and
    // compute the derivative along the columns.
    LinearInterpolate<Grid::DATA_DIMENSION>(f0, f1, r - row, f, dfdr);
    if (dfdc != NULL) {
      // Interpolate vertically the derivative along the columns.
      LinearInterpolate<Grid::DATA_DIMENSION>(df0dc, df1dc, r - row, dfdc, NULL);
    }
  }

  // The following two Evaluate overloads are needed for interfacing
  // with automatic differentiation. The first is for when a scalar
  // evaluation is done, and the second one is for when Jets are used.
  void Evaluate(const double& r, const double& c, double* f) const {
    Evaluate(r, c, f, NULL, NULL);
  }

  template <typename JetT>
  void Evaluate(const JetT& r, const JetT& c, JetT* f) const {
    double frc[Grid::DATA_DIMENSION];
    double dfdr[Grid::DATA_DIMENSION];
    double dfdc[Grid::DATA_DIMENSION];
    Evaluate(r.a, c.a, frc, dfdr, dfdc);
    for (int i = 0; i < Grid::DATA_DIMENSION; ++i) {
      f[i].a = frc[i];
      f[i].v = dfdr[i] * r.v + dfdc[i] * c.v;
    }
  }

 private:
  const Grid& grid_;
};

}  // namespace ceres

#endif

