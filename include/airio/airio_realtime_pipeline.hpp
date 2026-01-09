#pragma once

#include "../core/imu_buffer.hpp"
#include "../onnx/airimu_onnx_runner.hpp"
#include "../onnx/airio_onnx_runner.hpp"
#include "debug/airio_debug_sink.hpp"
#include "ekf/velocity_ekf.hpp"
#include <rclcpp/rclcpp.hpp>

#include <memory>

namespace airio {

struct ImuSampleSimple {
  double dt;
  airio::ekf::Vec3 gyro;
  airio::ekf::Vec3 acc;
};

struct AirioEkfState {
  airio::ekf::Vec3 so3_log;
  airio::ekf::Vec3 velocity_world;
  airio::ekf::Vec3 position_world;
  airio::ekf::Vec3 gyro_bias;
  airio::ekf::Vec3 acc_bias;
  double timestamp_sec;
};

struct AirioRealtimeParams {
  size_t buffer_len = 50;
  size_t window_len = 50;

  // AirIMU
  int airimu_row = 40;     // newest (50 - interval=9 - 1)
  double q_scale = 1.0;

  // Bias random walk
  double gyro_bias_rw = 1e-8;
  double acc_bias_rw  = 1e-6;

  // Warm-up behavior
  bool publish_during_warmup = false;

  // Default IMU noise during warm-up (raw IMU)
  double warmup_gyro_noise = 1e-4;
  double warmup_acc_noise  = 1e-2;
};

class AirioRealtimePipeline {
public:
  AirioRealtimePipeline(ImuBuffer* imu_buffer,
                        airimu_onnx::Runner* airimu_runner,
                        airio_onnx::Runner* airio_runner,
                        const AirioRealtimeParams& params = AirioRealtimeParams());

  // push IMU sample; returns true when EKF state was updated
  bool pushImu(const ImuSample& sample);

  // retrieve latest EKF state
  bool getLatestState(AirioEkfState& out) const;

  void setDebugSink(airio::debug::AirioDebugSink* sink) { dbg_ = sink; }

private:
  ImuBuffer* imu_buffer_;
  airimu_onnx::Runner* airimu_runner_;
  airio_onnx::Runner* airio_runner_;

  airio::ekf::VelocityEKF ekf_;
  AirioEkfState latest_state_;
  AirioRealtimeParams params_;

  // rot ring buffer (EKF so3_log sequence)
  size_t rot_cap_;
  std::vector<Eigen::Vector3d> rot_data_;
  size_t rot_size_;
  size_t rot_head_;

  // rot helpers
  void rot_push_(const Eigen::Vector3d& so3);
  bool rot_full_() const;
  bool fill_rot_flat_(std::vector<float>& out) const;  // [T*3], oldest->newest
  void rot_overwrite_latest_(const Eigen::Vector3d& so3);

  airio::ekf::Mat12 makeWarmupQ_() const;

  airio::debug::AirioDebugSink* dbg_ = nullptr;
};

} // namespace airio
