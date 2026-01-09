#pragma once

#include "types.hpp"
#include <functional>

namespace airio::ekf {

// Simple interface for Jacobian computation. Implementations may use FD or analytic.
struct JacobianCalculator {
  virtual ~JacobianCalculator() = default;

  // Compute A = df/dx (15x15)
  virtual Mat15 computeA(const std::function<Vec15(const Vec15&)>& f, const Vec15& x) = 0;

  // Compute B = df/du (15x12)
  virtual Mat15x12 computeB(const std::function<Vec15(const Vec12&)>& f_u, const Vec12& u) = 0;
  
  // Compute C = dg/dx (3x15)
  virtual Eigen::Matrix<double, 3, 15> computeC(const std::function<Vec3(const Vec15&)>& g, const Vec15& x) = 0;
};

// A simple finite difference Jacobian implementation
struct FiniteDiffJacobian : public JacobianCalculator {
  double eps_ = 1e-6;

  Mat15 computeA(const std::function<Vec15(const Vec15&)>& f, const Vec15& x) override {
    Mat15 A = Mat15::Zero();
    for (int i = 0; i < 15; ++i) {
      Vec15 xp = x; xp(i) += eps_;
      Vec15 xm = x; xm(i) -= eps_;
      Vec15 diff = (f(xp) - f(xm)) / (2.0 * eps_);
      A.col(i) = diff;
    }
    return A;
  }

  Mat15x12 computeB(const std::function<Vec15(const Vec12&)>& f_u, const Vec12& u) override {
    Mat15x12 B = Mat15x12::Zero();
    for (int i = 0; i < 12; ++i) {
      Vec12 up = u; up(i) += eps_;
      Vec12 um = u; um(i) -= eps_;
      Vec15 diff = (f_u(up) - f_u(um)) / (2.0 * eps_);
      B.col(i) = diff;
    }
    return B;
  }

  Eigen::Matrix<double, 3, 15> computeC(const std::function<Vec3(const Vec15&)>& g, const Vec15& x) override {
    Eigen::Matrix<double, 3, 15> C = Eigen::Matrix<double, 3, 15>::Zero();
    for (int i = 0; i < 15; ++i) {
      Vec15 xp = x; xp(i) += eps_;
      Vec15 xm = x; xm(i) -= eps_;
      Vec3 diff = (g(xp) - g(xm)) / (2.0 * eps_);
      C.col(i) = diff;
    }
    return C;
  }
};

} // namespace airio::ekf
