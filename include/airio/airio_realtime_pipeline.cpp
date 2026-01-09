#include "airio/airio_realtime_pipeline.hpp"

#include <Eigen/Dense>

namespace airio {

AirioRealtimePipeline::AirioRealtimePipeline(ImuBuffer* imu_buffer,
                                           airimu_onnx::Runner* airimu_runner,
                                           airio_onnx::Runner* airio_runner,
                                           const AirioRealtimeParams& params)
  : imu_buffer_(imu_buffer), airimu_runner_(airimu_runner), airio_runner_(airio_runner), params_(params)
{
}

bool AirioRealtimePipeline::pushImu(const ImuSample& sample)
{
  // push sample into provided buffer
  imu_buffer_->push(sample);

  if (!imu_buffer_->full()) return false;

  // --- AirIMU inference ---
  std::vector<float> feat;
  if (!imu_buffer_->fill_feat_flat(feat)) return false;
  auto imu_out = airimu_runner_->run(feat);

  // --- AirIO inference ---
  std::vector<float> acc, gyro;
  if (!imu_buffer_->fill_acc_flat(acc) || !imu_buffer_->fill_gyro_flat(gyro)) return false;
  std::vector<float> rot(acc.size(), 0.0f);
  auto io_out = airio_runner_->run(acc, gyro, rot);

  // Build observation z (body-frame velocity) if available
  std::optional<airio::ekf::Vec3> z = std::nullopt;
  if (io_out.net_vel.size() >= 3) {
    airio::ekf::Vec3 zb;
    zb << static_cast<double>(io_out.net_vel[0]),
          static_cast<double>(io_out.net_vel[1]),
          static_cast<double>(io_out.net_vel[2]);
    z = zb;
  }

  // Build R from AirIO cov (if available)
  airio::ekf::Mat3 R = airio::ekf::Mat3::Zero();
  if (io_out.cov.size() >= 3) {
    R = airio::ekf::Mat3::Zero();
    R(0,0) = static_cast<double>(io_out.cov[0]);
    R(1,1) = static_cast<double>(io_out.cov[1]);
    R(2,2) = static_cast<double>(io_out.cov[2]);
  }

  // Q is left as zero by default; user may later expose parameterization
  airio::ekf::Mat12 Q = airio::ekf::Mat12::Zero();

  // Build ImuInput for EKF using newest sample
  airio::ekf::ImuInput imu_in;
  imu_in.gyro = airio::ekf::Vec3(sample.gx, sample.gy, sample.gz);
  imu_in.acc  = airio::ekf::Vec3(sample.ax, sample.ay, sample.az);
  imu_in.d_bg = airio::ekf::Vec3::Zero();
  imu_in.d_ba = airio::ekf::Vec3::Zero();

  // Step EKF (use sample.dt as dt)
  ekf_.Step(imu_in, z, Q, R);

  // Update cached state
  const auto& x = ekf_.state();
  latest_state_.so3_log = x.segment<3>(0);
  latest_state_.velocity_world = x.segment<3>(3);
  latest_state_.position_world = x.segment<3>(6);
  latest_state_.gyro_bias = x.segment<3>(9);
  latest_state_.acc_bias = x.segment<3>(12);
  latest_state_.timestamp_sec = 0.0;

  return true;
}

bool AirioRealtimePipeline::getLatestState(AirioEkfState& out) const
{
  out = latest_state_;
  return true;
}

} // namespace airio
