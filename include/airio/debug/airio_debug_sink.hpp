#pragma once

#include <Eigen/Dense>
#include <optional>
#include <cstdint>

namespace airio::debug {

// AirIMU 결과(row r 기준)
struct AirimuDebug
{
  Eigen::Vector3d acc_corr = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_corr = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_cov  = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_cov = Eigen::Vector3d::Zero();
};

// AirIO 결과
struct AirioDebug
{
  bool has_net_vel = false;
  Eigen::Vector3d net_vel = Eigen::Vector3d::Zero();

  bool has_cov = false;
  Eigen::Vector3d cov_diag = Eigen::Vector3d::Zero(); // cov[0], cov[1], cov[2]
};

// EKF step 관련
struct EkfDebug
{
  double dt = 0.0;

  // pre/post 핵심 상태만
  Eigen::Vector3d so3_pre = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel_pre = Eigen::Vector3d::Zero();
  Eigen::Vector3d pos_pre = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg_pre  = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba_pre  = Eigen::Vector3d::Zero();

  Eigen::Vector3d so3_post = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel_post = Eigen::Vector3d::Zero();
  Eigen::Vector3d pos_post = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg_post  = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba_post  = Eigen::Vector3d::Zero();

  // measurement
  bool has_z = false;
  Eigen::Vector3d z_vel = Eigen::Vector3d::Zero();   // AirIO가 준 velocity (지금 코드 기준 Step에 들어간 값)
  Eigen::Vector3d R_diag = Eigen::Vector3d::Zero();  // R_meas diag

  // Q diag만
  Eigen::Matrix<double,12,1> Q_diag = Eigen::Matrix<double,12,1>::Zero();
};

// Debug sink interface
class AirioDebugSink
{
public:
  virtual ~AirioDebugSink() = default;

  // step boundary
  virtual void onPreEkf(const EkfDebug& ekf_pre) = 0;

  // module outputs
  virtual void onAirimu(const AirimuDebug& airimu) = 0;
  virtual void onAirio(const AirioDebug& airio) = 0;

  // EKF inputs
  virtual void onEkfMeas(const std::optional<Eigen::Vector3d>& z_vel,
                         const Eigen::Vector3d& R_diag) = 0;
  virtual void onEkfQ(const Eigen::Matrix<double,12,1>& Q_diag) = 0;

  // step end
  virtual void onPostEkf(const EkfDebug& ekf_post) = 0;
};

} // namespace airio::debug
