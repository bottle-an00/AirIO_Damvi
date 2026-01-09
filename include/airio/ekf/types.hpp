#pragma once

#include <Eigen/Dense>

namespace airio::ekf {

using Vec3 = Eigen::Vector3d;
using Vec15 = Eigen::Matrix<double, 15, 1>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat15 = Eigen::Matrix<double, 15, 15>;
using Mat12 = Eigen::Matrix<double, 12, 12>;
using Mat15x12 = Eigen::Matrix<double, 15, 12>;

struct ImuInput {
  Vec3 gyro;
  Vec3 acc;
  Vec3 d_bg; // placeholder, usually ignored per design notes
  Vec3 d_ba;
};

} // namespace airio::ekf
