#pragma once

#include "../core/imu_buffer.hpp"
#include "../onnx/airimu_onnx_runner.hpp"
#include "../onnx/airio_onnx_runner.hpp"
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

private:
  ImuBuffer* imu_buffer_;
  airimu_onnx::Runner* airimu_runner_;
  airio_onnx::Runner* airio_runner_;

  airio::ekf::VelocityEKF ekf_;
  AirioEkfState latest_state_;
  AirioRealtimeParams params_;
};

} // namespace airio
