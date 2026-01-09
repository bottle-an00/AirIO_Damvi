#pragma once

#include "airio/debug/airio_debug_sink.hpp"
#include <rclcpp/rclcpp.hpp>
#include <string>

namespace airio::debug {

struct RosDebugParams
{
  bool enable_airimu = true;
  bool enable_airio  = true;
  bool enable_ekf    = true;

  // 로그 폭주 방지: ms 단위 throttle (기본 1000ms)
  int throttle_ms_airimu = 1000;
  int throttle_ms_airio  = 1000;
  int throttle_ms_ekf    = 1000;
  int throttle_ms_qr     = 1000;

  // EKF는 요약만 INFO로 찍고, Q/R은 DEBUG로 찍는 형태가 기본
  bool ekf_info_summary = true;
  bool ekf_debug_qr     = true;

  // child logger suffix
  std::string child_airimu = "airimu";
  std::string child_airio  = "airio";
  std::string child_ekf    = "ekf";
};

class AirioRosDebugLogger final : public AirioDebugSink
{
public:
  AirioRosDebugLogger(rclcpp::Logger base_logger,
                      rclcpp::Clock::SharedPtr clock,
                      RosDebugParams params);

  void onPreEkf(const EkfDebug& ekf_pre) override;
  void onAirimu(const AirimuDebug& airimu) override;
  void onAirio(const AirioDebug& airio) override;
  void onEkfMeas(const std::optional<Eigen::Vector3d>& z_vel,
                 const Eigen::Vector3d& R_diag) override;
  void onEkfQ(const Eigen::Matrix<double,12,1>& Q_diag) override;
  void onPostEkf(const EkfDebug& ekf_post) override;

private:
  rclcpp::Logger l_airimu_;
  rclcpp::Logger l_airio_;
  rclcpp::Logger l_ekf_;
  rclcpp::Clock::SharedPtr clock_;
  RosDebugParams p_;

  // pre/post를 매칭해서 한 번에 요약 출력하기 위한 캐시
  EkfDebug ekf_pre_cache_;
  bool has_pre_ = false;

  // meas/Q 캐시(원하면 post 때 같이 출력도 가능)
  std::optional<Eigen::Vector3d> z_cache_;
  Eigen::Vector3d R_cache_ = Eigen::Vector3d::Zero();
  Eigen::Matrix<double,12,1> Q_cache_ = Eigen::Matrix<double,12,1>::Zero();
  bool has_meas_ = false;
  bool has_q_ = false;
};

} // namespace airio::debug
