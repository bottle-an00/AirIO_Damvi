#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <chrono>
#include <cmath>

#include "core/imu_buffer.hpp"

#include "tensorrt/airimu_tensorrt_runner.hpp"
#include "tensorrt/airio_tensorrt_runner.hpp"

class ImuSubscriberNode : public rclcpp::Node
{
public:
    ImuSubscriberNode()
        : Node("imu_subscriber"),
          message_count_(0),
          last_stamp_(rclcpp::Time(0, 0)),
          imu_buffer_(50),

          airimu_runner_("/root/AirIO_Damvi/model/airimu/airimu_codenet_fp32_T50.engine"),
          airio_runner_("/root/AirIO_Damvi/model/airio/airio_codewithrot_fp32_T50.engine")
    {
        rclcpp::SensorDataQoS qos;
        qos.keep_last(1);

        subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", qos,
            std::bind(&ImuSubscriberNode::imuCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "IMU Subscriber Node started. Subscribing to /imu/data");
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

        // -------------------------
        // AIRIMU (feat -> corr)
        // -------------------------
        std::vector<float> feat;
        if (imu_buffer_.fill_feat_flat(feat)) {
            auto out = airimu_runner_.run(feat);
            const auto& corr = out.corr;
            const auto& cov  = out.cov;
            
            constexpr int OUT_T = 41;
            constexpr int OUT_C = 6;
            int last = (OUT_T - 1) * OUT_C;

            RCLCPP_INFO(this->get_logger(),
                "[AIRIMU] corr_last=%f %f %f %f %f %f",
                corr[last+0], corr[last+1], corr[last+2],
                corr[last+3], corr[last+4], corr[last+5]);

            RCLCPP_INFO(this->get_logger(),
              "[AIRIMU] cov_last=%f %f %f %f %f %f",
              cov[last+0], cov[last+1], cov[last+2],
              cov[last+3], cov[last+4], cov[last+5]);
        } else {
            RCLCPP_WARN(this->get_logger(), "[AIRIMU-TRT] fill_feat_flat failed");
        }

        // -------------------------
        // AIRIO (acc/gyro/rot -> cov/net_vel)
        // -------------------------
        std::vector<float> acc, gyro;
        if (imu_buffer_.fill_acc_flat(acc) && imu_buffer_.fill_gyro_flat(gyro)) {

            std::vector<float> rot(acc.size(), 0.0f); // size = T*3

            auto out = airio_runner_.run(acc, gyro, rot);

            RCLCPP_INFO(this->get_logger(),
                "[AIRIO-TRT] cov size=%zu, net_vel size=%zu", out.cov.size(), out.net_vel.size());

            if (out.net_vel.size() >= 3) {
                RCLCPP_INFO(this->get_logger(),
                    "[AIRIO-TRT] net_vel[0]=%.6f %.6f %.6f",
                    out.net_vel[0], out.net_vel[1], out.net_vel[2]);
            }
            if (out.cov.size() >= 3) {
                RCLCPP_INFO(this->get_logger(),
                    "[AIRIO-TRT] cov[0]=%.6f %.6f %.6f",
                    out.cov[0], out.cov[1], out.cov[2]);
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "[AIRIO-TRT] fill_acc/gyro failed");
        }
    }

    ImuSample createSampleFromMsg(const sensor_msgs::msg::Imu::SharedPtr msg, double& dt_out)
    {
        double dt = 0.0;
        rclcpp::Time current_stamp(msg->header.stamp);
        if (last_stamp_.seconds() != 0.0 || last_stamp_.nanoseconds() != 0) {
            rclcpp::Duration duration = current_stamp - last_stamp_;
            dt = duration.seconds();
        }
        last_stamp_ = current_stamp;
        dt_out = dt;

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

    airimu_trt::Runner airimu_runner_;
    airio_trt::Runner airio_runner_;
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
