#include "airio/ekf/velocity_ekf.hpp"

#include <Eigen/Dense>
#include <iostream>

namespace airio::ekf {

VelocityEKF::VelocityEKF() {
  jacobian_ = std::make_unique<FiniteDiffJacobian>();
}

void VelocityEKF::Reset(const Vec15& x0, const Mat15& P0) {
  x_ = x0;
  P_ = P0;
}

Vec12 VelocityEKF::BuildInput(const ImuInput& imu, const Vec15& /*x*/) const {
  Vec12 u;
  u.segment<3>(0) = imu.gyro;
  u.segment<3>(3) = imu.acc;
  u.segment<3>(6) = imu.d_bg;
  u.segment<3>(9) = imu.d_ba;
  return u;
}

Vec15 VelocityEKF::StateTransition(const Vec15& x, const Vec12& u, double dt) const {
  Vec3 r = x.segment<3>(0);
  Vec3 v = x.segment<3>(3);
  Vec3 p = x.segment<3>(6);
  Vec3 bg = x.segment<3>(9);
  Vec3 ba = x.segment<3>(12);

  Eigen::Matrix3d R0 = ExpSO3(r);
  Vec3 gyro = u.segment<3>(0);
  Vec3 acc = u.segment<3>(3);

  Vec3 w = gyro - bg;
  Vec3 a = acc - InvSO3(R0) * GRAVITY - ba;

  Eigen::Matrix3d Dr = ExpSO3(w * dt);
  Vec3 Dv = (Dr * a) * dt;
  Vec3 Dp = Dv * dt + (Dr * a) * (0.5 * dt * dt);

  Eigen::Matrix3d Rnext = R0 * Dr;
  Vec3 r_next = LogSO3(Rnext);
  Vec3 v_next = v + R0 * Dv;
  Vec3 p_next = p + v * dt + R0 * Dp;

  Vec15 xn = Vec15::Zero();
  xn.segment<3>(0) = r_next;
  xn.segment<3>(3) = v_next;
  xn.segment<3>(6) = p_next;
  xn.segment<3>(9) = bg;
  xn.segment<3>(12) = ba;
  return xn;
}

Vec3 VelocityEKF::Observation(const Vec15& x, const Vec12& u, double dt) const {
  // observation uses nstate produced by f(x,u,dt)
  Vec15 nstate = StateTransition(x, u, dt);
  Eigen::Matrix3d R = ExpSO3(nstate.segment<3>(0));
  Vec3 v_body = InvSO3(R) * nstate.segment<3>(3);
  return v_body;
}

void VelocityEKF::Predict(const Vec12& u, double dt, const Mat12& Q) {
  // x_p
  Vec15 x_p = StateTransition(x_, u, dt);

  // A and B via finite difference on f
  auto f_x = [&](const Vec15& x_in)->Vec15 { return StateTransition(x_in, u, dt); };
  auto f_u = [&](const Vec12& u_in)->Vec15 { return StateTransition(x_, u_in, dt); };

  Mat15 A = jacobian_->computeA(f_x, x_);
  Mat15x12 B = jacobian_->computeB(f_u, u);

  Mat15 P_p = A * P_ * A.transpose();
  P_p.noalias() += B * Q * B.transpose();

  x_ = x_p;
  P_ = P_p;
}

void VelocityEKF::UpdateVelocity(const Vec3& z, const Mat3& R) {
  // Compute observation and C = dg/dx (finite diff)
  auto g = [&](const Vec15& x_in)->Vec3 { return Observation(x_in, Vec12::Zero(), 0.0); };

  // Compute C = dg/dx using the injected Jacobian calculator
  Eigen::Matrix<double, 3, 15> C = jacobian_->computeC(g, x_);

  Eigen::Matrix3d S = C * P_ * C.transpose() + R;
  // Use LDLT for numerical stability
  Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
  Eigen::Matrix<double, 15, 3> K = P_ * C.transpose() * ldlt.solve(Eigen::Matrix3d::Identity());

  Vec3 g_x = g(x_);
  Vec3 e = z - g_x;

  x_ = x_ + K * e;

  // Joseph form
  Mat15 I = Mat15::Identity();
  Mat15 IKC = I - K * C;
  P_ = IKC * P_ * IKC.transpose() + K * R * K.transpose();
}

void VelocityEKF::Step(const ImuInput& imu, const std::optional<Vec3>& z, const Mat12& Q, const Mat3& R) {
  Vec12 u = BuildInput(imu, x_);
  Predict(u, /*dt*/ 1e-3, Q); // note: user should pass real dt; placeholder here
  if (z.has_value()) {
    UpdateVelocity(z.value(), R);
  }
}

} // namespace airio::ekf
