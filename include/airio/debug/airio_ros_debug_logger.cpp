#include "airio/debug/airio_ros_debug_logger.hpp"

namespace airio::debug {

AirioRosDebugLogger::AirioRosDebugLogger(rclcpp::Logger base_logger,
                                         rclcpp::Clock::SharedPtr clock,
                                         RosDebugParams params)
  : l_airimu_(base_logger.get_child(params.child_airimu)),
    l_airio_(base_logger.get_child(params.child_airio)),
    l_ekf_(base_logger.get_child(params.child_ekf)),
    clock_(std::move(clock)),
    p_(std::move(params))
{}

void AirioRosDebugLogger::onPreEkf(const EkfDebug& ekf_pre)
{
  if (!p_.enable_ekf) return;
  ekf_pre_cache_ = ekf_pre;
  has_pre_ = true;

  // step 시작마다 캐시 리셋
  z_cache_.reset();
  R_cache_.setZero();
  Q_cache_.setZero();
  has_meas_ = false;
  has_q_ = false;
}

void AirioRosDebugLogger::onAirimu(const AirimuDebug& d)
{
  if (!p_.enable_airimu) return;

  RCLCPP_INFO_THROTTLE(
    l_airimu_, *clock_, p_.throttle_ms_airimu,
    "[AirIMU] acc_corr=%.4f %.4f %.4f | gyro_corr=%.4f %.4f %.4f | "
    "acc_cov=%.6f %.6f %.6f | gyro_cov=%.6f %.6f %.6f",
    d.acc_corr.x(), d.acc_corr.y(), d.acc_corr.z(),
    d.gyro_corr.x(), d.gyro_corr.y(), d.gyro_corr.z(),
    d.acc_cov.x(), d.acc_cov.y(), d.acc_cov.z(),
    d.gyro_cov.x(), d.gyro_cov.y(), d.gyro_cov.z()
  );
}

void AirioRosDebugLogger::onAirio(const AirioDebug& d)
{
  if (!p_.enable_airio) return;

  if (d.has_net_vel && d.has_cov) {
    RCLCPP_INFO_THROTTLE(
      l_airio_, *clock_, p_.throttle_ms_airio,
      "[AirIO ] net_vel=%.4f %.4f %.4f | cov_diag=%.6f %.6f %.6f",
      d.net_vel.x(), d.net_vel.y(), d.net_vel.z(),
      d.cov_diag.x(), d.cov_diag.y(), d.cov_diag.z()
    );
  } else {
    RCLCPP_DEBUG_THROTTLE(
      l_airio_, *clock_, p_.throttle_ms_airio,
      "[AirIO ] N/A (output not ready)"
    );
  }
}

void AirioRosDebugLogger::onEkfMeas(const std::optional<Eigen::Vector3d>& z_vel,
                                   const Eigen::Vector3d& R_diag)
{
  if (!p_.enable_ekf) return;

  z_cache_ = z_vel;
  R_cache_ = R_diag;
  has_meas_ = true;

  if (p_.ekf_debug_qr) {
    if (z_cache_) {
      RCLCPP_DEBUG_THROTTLE(
        l_ekf_, *clock_, p_.throttle_ms_qr,
        "[EKF-z] z_vel=%.4f %.4f %.4f | R_diag=%.6f %.6f %.6f",
        z_cache_->x(), z_cache_->y(), z_cache_->z(),
        R_cache_.x(), R_cache_.y(), R_cache_.z()
      );
    } else {
      RCLCPP_DEBUG_THROTTLE(
        l_ekf_, *clock_, p_.throttle_ms_qr,
        "[EKF-z] N/A | R_diag=%.6f %.6f %.6f",
        R_cache_.x(), R_cache_.y(), R_cache_.z()
      );
    }
  }
}

void AirioRosDebugLogger::onEkfQ(const Eigen::Matrix<double,12,1>& Q_diag)
{
  if (!p_.enable_ekf) return;

  Q_cache_ = Q_diag;
  has_q_ = true;

  if (p_.ekf_debug_qr) {
    RCLCPP_DEBUG_THROTTLE(
      l_ekf_, *clock_, p_.throttle_ms_qr,
      "[EKF-Q] diag=[%.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e]",
      Q_cache_(0), Q_cache_(1), Q_cache_(2),
      Q_cache_(3), Q_cache_(4), Q_cache_(5),
      Q_cache_(6), Q_cache_(7), Q_cache_(8),
      Q_cache_(9), Q_cache_(10), Q_cache_(11)
    );
  }
}

void AirioRosDebugLogger::onPostEkf(const EkfDebug& ekf_post)
{
  if (!p_.enable_ekf) return;

  if (!has_pre_) {
    // pre가 없으면 그냥 post만 요약
    if (p_.ekf_info_summary) {
      RCLCPP_INFO_THROTTLE(
        l_ekf_, *clock_, p_.throttle_ms_ekf,
        "[EKF  ] dt=%.4f | vel_post=%.3f %.3f %.3f | pos_post=%.3f %.3f %.3f | "
        "bg=%.5f %.5f %.5f | ba=%.5f %.5f %.5f",
        ekf_post.dt,
        ekf_post.vel_post.x(), ekf_post.vel_post.y(), ekf_post.vel_post.z(),
        ekf_post.pos_post.x(), ekf_post.pos_post.y(), ekf_post.pos_post.z(),
        ekf_post.bg_post.x(),  ekf_post.bg_post.y(),  ekf_post.bg_post.z(),
        ekf_post.ba_post.x(),  ekf_post.ba_post.y(),  ekf_post.ba_post.z()
      );
    }
    return;
  }

  if (p_.ekf_info_summary) {
    RCLCPP_INFO_THROTTLE(
      l_ekf_, *clock_, p_.throttle_ms_ekf,
      "[EKF  ] dt=%.4f | vel: %.3f %.3f %.3f -> %.3f %.3f %.3f | "
      "pos_post=%.3f %.3f %.3f | bg=%.5f %.5f %.5f | ba=%.5f %.5f %.5f",
      ekf_post.dt,
      ekf_pre_cache_.vel_pre.x(), ekf_pre_cache_.vel_pre.y(), ekf_pre_cache_.vel_pre.z(),
      ekf_post.vel_post.x(),      ekf_post.vel_post.y(),      ekf_post.vel_post.z(),
      ekf_post.pos_post.x(),      ekf_post.pos_post.y(),      ekf_post.pos_post.z(),
      ekf_post.bg_post.x(),       ekf_post.bg_post.y(),       ekf_post.bg_post.z(),
      ekf_post.ba_post.x(),       ekf_post.ba_post.y(),       ekf_post.ba_post.z()
    );
  }

  // state machine
  has_pre_ = false;
}

} // namespace airio::debug
