#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <chrono>
#include <cmath>

#include "core/imu_buffer.hpp"

#include "onnx/airimu_onnx_runner.hpp"
#include "onnx/airio_onnx_runner.hpp"

class ImuSubscriberNode : public rclcpp::Node
{
public:
    ImuSubscriberNode()
        : Node("imu_subscriber"),
          last_stamp_(rclcpp::Time(0, 0)),
          imu_buffer_(50),
          airimu_runner_("/home/jba/AirIO_Damvi/model/airimu/airimu_codenet_fp32_T50.onnx"),
          airio_runner_("/home/jba/AirIO_Damvi/model/airio/airio_codewithrot_fp32_T50.onnx")
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

        if (!imu_buffer_.full()) return;

        std::vector<float> feat;
        if (imu_buffer_.fill_feat_flat(feat)) {
            auto corr = airimu_runner_.run(feat);
        
            // corr는 [41*6] 가정(airimu runner가 그렇게 반환)
            RCLCPP_INFO(this->get_logger(),
                "[AIRIMU] corr size=%zu, corr[0]=%.6f %.6f %.6f %.6f %.6f %.6f",
                corr.size(),
                corr.size() >= 6 ? corr[0] : 0.0f,
                corr.size() >= 6 ? corr[1] : 0.0f,
                corr.size() >= 6 ? corr[2] : 0.0f,
                corr.size() >= 6 ? corr[3] : 0.0f,
                corr.size() >= 6 ? corr[4] : 0.0f,
                corr.size() >= 6 ? corr[5] : 0.0f
            );
        } else {
            RCLCPP_WARN(this->get_logger(), "[AIRIMU] fill_feat_flat failed");
        }

        std::vector<float> acc, gyro;
        if (imu_buffer_.fill_acc_flat(acc) && imu_buffer_.fill_gyro_flat(gyro)) {

            // rot은 외부 입력 예정이므로, 현재는 "동작 확인용" 더미 0으로 채움
            std::vector<float> rot(acc.size(), 0.0f); // size = T*3

            auto out = airio_runner_.run(acc, gyro, rot);

            RCLCPP_INFO(this->get_logger(),
                "[AIRIO] cov size=%zu, net_vel size=%zu", out.cov.size(), out.net_vel.size());

            if (out.net_vel.size() >= 3) {
                RCLCPP_INFO(this->get_logger(),
                    "[AIRIO] net_vel[0]=%.6f %.6f %.6f",
                    out.net_vel[0], out.net_vel[1], out.net_vel[2]);
            }
            if (out.cov.size() >= 3) {
                RCLCPP_INFO(this->get_logger(),
                    "[AIRIO] cov[0]=%.6f %.6f %.6f",
                    out.cov[0], out.cov[1], out.cov[2]);
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "[AIRIO] fill_acc/gyro failed");
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
    
    //imu buffer
    ImuBuffer imu_buffer_;
    
    // onnx runners
    airimu_onnx::Runner airimu_runner_;
    airio_onnx::Runner airio_runner_;
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