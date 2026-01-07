#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <chrono>
#include <cmath>
#include "core/imu_buffer.hpp"

class ImuSubscriberNode : public rclcpp::Node
{
public:
    ImuSubscriberNode()
        : Node("imu_subscriber"),
          last_stamp_(rclcpp::Time(0, 0)),
          imu_buffer_(200)
    {
        // QoS 설정: SensorDataQoS
        rclcpp::SensorDataQoS qos;
        qos.keep_last(1);  // 최신 데이터 유지

        // Subscriber 생성
        subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", qos,
            std::bind(&ImuSubscriberNode::imuCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "IMU Subscriber Node started. Subscribing to /imu/data_raw");
    }

private:

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        message_count_++;

        double dt;
        ImuSample sample = createSampleFromMsg(msg, dt);

        RCLCPP_INFO(this->get_logger(), "Message count: %d, dt: %.6f", message_count_, dt);

        imu_buffer_.push(sample);

        if (imu_buffer_.full()) {
            std::vector<float> features;
            if (imu_buffer_.fill_features(features)) {
                // 모델 입력으로 사용
            }
        }
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

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscription_;
    int message_count_;
    rclcpp::Time last_stamp_;
    ImuBuffer imu_buffer_;
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