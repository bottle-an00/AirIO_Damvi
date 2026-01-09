#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <chrono>
#include <cmath>
#include <memory>

#include "core/imu_buffer.hpp"

#include "onnx/airimu_onnx_runner.hpp"
#include "onnx/airio_onnx_runner.hpp"
#include "airio/airio_realtime_pipeline.hpp"
#include "airio/debug/airio_ros_debug_logger.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <Eigen/Dense>

static inline Eigen::Matrix3d so3Exp(const Eigen::Vector3d& w)
{
    const double theta = w.norm();
    Eigen::Matrix3d W;
    W <<     0.0, -w.z(),  w.y(),
          w.z(),    0.0, -w.x(),
         -w.y(),  w.x(),   0.0;

    if (theta < 1e-12) {
        return Eigen::Matrix3d::Identity() + W; // small-angle
    }

    const double a = std::sin(theta) / theta;
    const double b = (1.0 - std::cos(theta)) / (theta * theta);
    return Eigen::Matrix3d::Identity() + a * W + b * (W * W);
}

static inline geometry_msgs::msg::Quaternion toQuatMsg(const Eigen::Vector3d& so3_log)
{
    Eigen::Quaterniond q(so3Exp(so3_log));
    q.normalize();

    geometry_msgs::msg::Quaternion out;
    out.w = q.w();
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    return out;
}

class ImuSubscriberNode : public rclcpp::Node
{
public:
    ImuSubscriberNode()
        : Node("imu_subscriber"),
            message_count_(0),
            last_stamp_(rclcpp::Time(0, 0)),
            imu_buffer_(50),
            airimu_runner_("/root/AirIO_Damvi/model/airimu/airimu_codenet_fp32_T50.onnx"),
            airio_runner_("/root/AirIO_Damvi/model/airio/airio_codewithrot_fp32_T50.onnx")
    {

        // QoS 설정: SensorDataQoS
        rclcpp::SensorDataQoS qos;
        qos.keep_last(1);  // 최신 데이터 유지

        // Subscriber 생성
        subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", qos,
            std::bind(&ImuSubscriberNode::imuCallback, this, std::placeholders::_1));
        // Publisher 생성
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/airio/odometry", 10);

        // create realtime pipeline using existing buffer and runners
        pipeline_ = std::make_unique<airio::AirioRealtimePipeline>(&imu_buffer_, &airimu_runner_, &airio_runner_);

        // create debug logger
        airio::debug::RosDebugParams dp;
        dp.throttle_ms_airimu = 1;
        dp.throttle_ms_airio  = 1;
        dp.throttle_ms_ekf    = 1;
        dp.throttle_ms_qr     = 1;
        dp.enable_airimu = true;
        dp.enable_airio  = true;
        dp.enable_ekf    = true;

        dbg_logger_ = std::make_unique<airio::debug::AirioRosDebugLogger>(
            this->get_logger().get_child("airio"),
            this->get_clock(),
            dp
        );

        pipeline_->setDebugSink(dbg_logger_.get());

        RCLCPP_INFO(this->get_logger(), "IMU Subscriber Node started. Subscribing to /imu/data_raw");
    }

private:

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        message_count_++;

        double dt;
        ImuSample sample = createSampleFromMsg(msg, dt);

        if (dt <= 0.0) return;

        if (!pipeline_) return;

        const bool updated = pipeline_->pushImu(sample);

        if (!updated) {
            return;
        }

        airio::AirioEkfState st;
        if (!pipeline_->getLatestState(st)) return;

        publishOdometry(st, rclcpp::Time(msg->header.stamp));
    }

    ImuSample createSampleFromMsg(const sensor_msgs::msg::Imu::SharedPtr msg, double& dt_out)
    {
        // dt 계산 (inter-message period)
        double dt = 0.0;
        rclcpp::Time current_stamp(msg->header.stamp);
        if (last_stamp_.seconds() != 0.0 || last_stamp_.nanoseconds() != 0) {
            rclcpp::Duration duration = current_stamp - last_stamp_;
            dt = duration.seconds();
        }
        last_stamp_ = current_stamp;
        dt_out = dt;

        // ImuSample 생성
        ImuSample sample;
        sample.ax = msg->linear_acceleration.x;
        sample.ay = msg->linear_acceleration.y;
        sample.az = msg->linear_acceleration.z;
        sample.gx = msg->angular_velocity.x;
        sample.gy = msg->angular_velocity.y;
        sample.gz = msg->angular_velocity.z;
        sample.dt = dt;

        return sample;
    }

    void publishOdometry(const airio::AirioEkfState& st, const rclcpp::Time& stamp)
    {
        if (!odom_pub_) return;

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = frame_id_;
        odom.child_frame_id = child_frame_id_;

        odom.pose.pose.position.x = st.position_world.x();
        odom.pose.pose.position.y = st.position_world.y();
        odom.pose.pose.position.z = st.position_world.z();
        odom.pose.pose.orientation = toQuatMsg(st.so3_log);

        odom.twist.twist.linear.x = st.velocity_world.x();
        odom.twist.twist.linear.y = st.velocity_world.y();
        odom.twist.twist.linear.z = st.velocity_world.z();

        // covariance: 일단 0 (추후 EKF P로 채우면 더 좋을듯)
        for (int i = 0; i < 36; ++i) {
            odom.pose.covariance[i]  = 0.0;
            odom.twist.covariance[i] = 0.0;
        }

        odom_pub_->publish(odom);
    }

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscription_;
    int message_count_;
    rclcpp::Time last_stamp_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::string frame_id_ = "odom";
    std::string child_frame_id_ = "base_link";

    //imu buffer
    ImuBuffer imu_buffer_;

    // onnx runners
    airimu_onnx::Runner airimu_runner_;
    airio_onnx::Runner airio_runner_;
    std::unique_ptr<airio::AirioRealtimePipeline> pipeline_;

    //logger
    std::unique_ptr<airio::debug::AirioRosDebugLogger> dbg_logger_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::executors::SingleThreadedExecutor executor;
    auto node = std::make_shared<ImuSubscriberNode>();
    executor.add_node(node);

    RCLCPP_INFO(node->get_logger(), "Starting IMU Subscriber with SingleThreadedExecutor");
    executor.spin();

    rclcpp::shutdown();
    return 0;
}