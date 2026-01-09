#pragma once

#include "types.hpp"
#include "so3_ops.hpp"
#include "jacobian.hpp"

#include <optional>
#include <memory>

namespace airio::ekf {

class VelocityEKF {
public:
  VelocityEKF();

  void Reset(const Vec15& x0, const Mat15& P0);

  Vec12 BuildInput(const ImuInput& imu, const Vec15& x) const;

  Vec15 StateTransition(const Vec15& x, const Vec12& u, double dt) const;

  Vec3 Observation(const Vec15& x, const Vec12& u, double dt) const;

  void Predict(const Vec12& u, double dt, const Mat12& Q);

  void UpdateVelocity(const Vec3& z, const Mat3& R);

  // dt: time delta (seconds) to be used for prediction
  void Step(const ImuInput& imu, const std::optional<Vec3>& z, const Mat12& Q, const Mat3& R, double dt);

  // Accessors
  const Vec15& state() const { return x_; }
  const Mat15& cov() const { return P_; }

  // Allow injecting a Jacobian calculator (FD by default)
  std::unique_ptr<JacobianCalculator> jacobian_;

private:
  Vec15 x_ = Vec15::Zero();
  Mat15 P_ = Mat15::Identity();

  static inline const Vec3 GRAVITY = Vec3(0.0, 0.0, 9.8107);
};

} // namespace airio::ekf
