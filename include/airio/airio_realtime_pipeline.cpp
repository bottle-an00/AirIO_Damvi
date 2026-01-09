#include "airio/airio_realtime_pipeline.hpp"

#include <Eigen/Dense>

namespace airio {

AirioRealtimePipeline::AirioRealtimePipeline(ImuBuffer* imu_buffer,
                                           airimu_onnx::Runner* airimu_runner,
                                           airio_onnx::Runner* airio_runner,
                                           const AirioRealtimeParams& params)
  : imu_buffer_(imu_buffer), airimu_runner_(airimu_runner), airio_runner_(airio_runner), params_(params),
    rot_cap_(params.window_len),
    rot_data_(params.window_len, Eigen::Vector3d::Zero()),
    rot_size_(0),
    rot_head_(0)
{}

void AirioRealtimePipeline::rot_push_(const Eigen::Vector3d& so3)
{
  // ImuBuffer::push()와 동일한 로직
  rot_data_[rot_head_] = so3;
  rot_head_ = (rot_head_ + 1) % rot_cap_;
  if (rot_size_ < rot_cap_) ++rot_size_;
}

bool AirioRealtimePipeline::rot_full_() const
{
  return rot_size_ == rot_cap_;
}

bool AirioRealtimePipeline::fill_rot_flat_(std::vector<float>& out) const
{
  if (!rot_full_()) return false;

  const size_t T = rot_cap_;
  out.resize(T * 3);

  // ImuBuffer::fill_*_flat()과 동일: oldest -> newest
  // idx = (head - T + i + T) % T  (T == rot_cap_)
  for (size_t i = 0; i < T; ++i) {
    const size_t idx = (rot_head_ - T + i + T) % T;
    const Eigen::Vector3d& r = rot_data_[idx];

    out[i*3 + 0] = static_cast<float>(r.x());
    out[i*3 + 1] = static_cast<float>(r.y());
    out[i*3 + 2] = static_cast<float>(r.z());
  }
  return true;
}

void AirioRealtimePipeline::rot_overwrite_latest_(const Eigen::Vector3d& so3)
{
  // 최신 원소(방금 push된 slot)를 post-step EKF 값으로 덮어쓰기 용도
  if (rot_size_ == 0) return;

  const size_t T = rot_cap_;
  const size_t latest_idx = (rot_head_ - 1 + T) % T;
  rot_data_[latest_idx] = so3;
}

bool AirioRealtimePipeline::pushImu(const ImuSample& sample)
{
  // ------------------------------------------------
  // 1. Push IMU + rot (always)
  // ------------------------------------------------
  imu_buffer_->push(sample);

  Eigen::Vector3d so3_pre = ekf_.state().segment<3>(0);
  rot_push_(so3_pre);

  // ------------------------------------------------
  // 2. WARM-UP: window not full
  // ------------------------------------------------
  if (!imu_buffer_->full() || !rot_full_()) {

    // raw IMU propagation only
    airio::ekf::ImuInput imu_in;
    imu_in.gyro = airio::ekf::Vec3(sample.gx, sample.gy, sample.gz);
    imu_in.acc  = airio::ekf::Vec3(sample.ax, sample.ay, sample.az);
    imu_in.d_bg = airio::ekf::Vec3::Zero();
    imu_in.d_ba = airio::ekf::Vec3::Zero();

    auto Q = makeWarmupQ_();

    ekf_.Step(
      imu_in,
      std::nullopt,              // ❌ no measurement
      Q,
      airio::ekf::Mat3::Zero(),  // unused
      sample.dt
    );

    // overwrite latest rot with post-step orientation
    Eigen::Vector3d so3_post = ekf_.state().segment<3>(0);
    rot_overwrite_latest_(so3_post);

    // cache state
    const auto& x = ekf_.state();
    latest_state_.so3_log = x.segment<3>(0);
    latest_state_.velocity_world = x.segment<3>(3);
    latest_state_.position_world = x.segment<3>(6);
    latest_state_.gyro_bias = x.segment<3>(9);
    latest_state_.acc_bias  = x.segment<3>(12);

    return params_.publish_during_warmup;
  }

  // ------------------------------------------------
  // 3. STEADY STATE (window full)
  // ------------------------------------------------

  // --- AirIMU ---
  std::vector<float> feat;
  imu_buffer_->fill_feat_flat(feat);
  const auto imu_out = airimu_runner_->run(feat);

  const int r = params_.airimu_row;
  const size_t base = r * 6;

  // corr: [acc(3), gyro(3)]
  Eigen::Vector3d acc_corr(
    imu_out.corr[base+0],
    imu_out.corr[base+1],
    imu_out.corr[base+2]);

  Eigen::Vector3d gyro_corr(
    imu_out.corr[base+3],
    imu_out.corr[base+4],
    imu_out.corr[base+5]);

  // cov: [acc(3), gyro(3)]
  Eigen::Vector3d acc_cov(
    imu_out.cov[base+0],
    imu_out.cov[base+1],
    imu_out.cov[base+2]);

  Eigen::Vector3d gyro_cov(
    imu_out.cov[base+3],
    imu_out.cov[base+4],
    imu_out.cov[base+5]);

  // --- AirIO ---
  std::vector<float> acc, gyro, rot;
  imu_buffer_->fill_acc_flat(acc);
  imu_buffer_->fill_gyro_flat(gyro);
  fill_rot_flat_(rot);

  auto io_out = airio_runner_->run(acc, gyro, rot);

  std::optional<airio::ekf::Vec3> z;
  if (io_out.net_vel.size() >= 3) {
    airio::ekf::Vec3 zb;
    zb << io_out.net_vel[0], io_out.net_vel[1], io_out.net_vel[2];
    z = zb;
  }

  airio::ekf::Mat3 R_meas = airio::ekf::Mat3::Zero();
  R_meas(0,0) = io_out.cov[0];
  R_meas(1,1) = io_out.cov[1];
  R_meas(2,2) = io_out.cov[2];

  // --- Q from AirIMU ---
  airio::ekf::Mat12 Q = airio::ekf::Mat12::Zero();
  Q(0,0) = params_.q_scale * gyro_cov.x();
  Q(1,1) = params_.q_scale * gyro_cov.y();
  Q(2,2) = params_.q_scale * gyro_cov.z();
  Q(3,3) = params_.q_scale * acc_cov.x();
  Q(4,4) = params_.q_scale * acc_cov.y();
  Q(5,5) = params_.q_scale * acc_cov.z();
  Q(6,6) = params_.gyro_bias_rw;
  Q(7,7) = params_.gyro_bias_rw;
  Q(8,8) = params_.gyro_bias_rw;
  Q(9,9) = params_.acc_bias_rw;
  Q(10,10)= params_.acc_bias_rw;
  Q(11,11)= params_.acc_bias_rw;

  // --- EKF ---
  airio::ekf::ImuInput imu_in;
  imu_in.acc  = airio::ekf::Vec3(
    sample.ax + acc_corr.x(),
    sample.ay + acc_corr.y(),
    sample.az + acc_corr.z());
  imu_in.gyro = airio::ekf::Vec3(
    sample.gx + gyro_corr.x(),
    sample.gy + gyro_corr.y(),
    sample.gz + gyro_corr.z());
  imu_in.d_bg = airio::ekf::Vec3::Zero();
  imu_in.d_ba = airio::ekf::Vec3::Zero();

  ekf_.Step(imu_in, z, Q, R_meas, sample.dt);

  // rot overwrite
  rot_overwrite_latest_(ekf_.state().segment<3>(0));

  // cache
  const auto& x = ekf_.state();
  latest_state_.so3_log = x.segment<3>(0);
  latest_state_.velocity_world = x.segment<3>(3);
  latest_state_.position_world = x.segment<3>(6);
  latest_state_.gyro_bias = x.segment<3>(9);
  latest_state_.acc_bias  = x.segment<3>(12);

  return true;
}

bool AirioRealtimePipeline::getLatestState(AirioEkfState& out) const
{
  out = latest_state_;
  return true;
}

airio::ekf::Mat12 airio::AirioRealtimePipeline::makeWarmupQ_() const {
  airio::ekf::Mat12 Q = airio::ekf::Mat12::Zero();
  Q(0,0)=params_.warmup_gyro_noise; Q(1,1)=params_.warmup_gyro_noise; Q(2,2)=params_.warmup_gyro_noise;
  Q(3,3)=params_.warmup_acc_noise;  Q(4,4)=params_.warmup_acc_noise;  Q(5,5)=params_.warmup_acc_noise;
  Q(6,6)=params_.gyro_bias_rw;      Q(7,7)=params_.gyro_bias_rw;      Q(8,8)=params_.gyro_bias_rw;
  Q(9,9)=params_.acc_bias_rw;       Q(10,10)=params_.acc_bias_rw;     Q(11,11)=params_.acc_bias_rw;
  return Q;
}

} // namespace airio
