#pragma once

#include "types.hpp"
#include <Eigen/Dense>

namespace airio::ekf {

// Exponential map: rotation vector (so3) -> rotation matrix (SO3)
inline Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& phi) {
  double theta = phi.norm();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  if (theta < 1e-12) return R;
  Eigen::Vector3d k = phi / theta;
  Eigen::Matrix3d K;
  K <<     0, -k.z(),  k.y(),
        k.z(),     0, -k.x(),
       -k.y(),  k.x(),     0;
  R = Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * (K * K);
  return R;
}

// Log map: rotation matrix -> rotation vector
inline Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
  double cos_theta = (R.trace() - 1.0) / 2.0;
  cos_theta = std::min(1.0, std::max(-1.0, cos_theta));
  double theta = std::acos(cos_theta);
  if (theta < 1e-12) return Eigen::Vector3d::Zero();
  Eigen::Vector3d v;
  v << R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1);
  v *= 1.0 / (2.0 * std::sin(theta));
  return v * theta;
}

inline Eigen::Matrix3d InvSO3(const Eigen::Matrix3d& R) { return R.transpose(); }

inline Eigen::Vector3d Rotate(const Eigen::Matrix3d& R, const Eigen::Vector3d& v) { return R * v; }

} // namespace airio::ekf
