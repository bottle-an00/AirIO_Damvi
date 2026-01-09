#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <chrono>
#include <cmath>
#include <memory>

#include "core/imu_buffer.hpp"

#include "onnx/airimu_onnx_runner.hpp"
#include "onnx/airio_onnx_runner.hpp"
#include "airio/airio_realtime_pipeline.hpp"

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

        // create realtime pipeline using existing buffer and runners
        pipeline_ = std::make_unique<airio::AirioRealtimePipeline>(&imu_buffer_, &airimu_runner_, &airio_runner_);

        RCLCPP_INFO(this->get_logger(), "IMU Subscriber Node started. Subscribing to /imu/data_raw");
    }

private:

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        message_count_++;

        double dt;
        ImuSample sample = createSampleFromMsg(msg, dt);

        RCLCPP_INFO(this->get_logger(), "Message count: %d, dt: %.6f", message_count_, dt);

        // push into unified pipeline; pipeline owns inference + EKF steps
        imu_buffer_.push(sample); // keep buffer state consistent
        if (!pipeline_) return;

        bool updated = pipeline_->pushImu(sample);
        if (updated) {
            airio::AirioEkfState st;
            if (pipeline_->getLatestState(st)) {
                RCLCPP_INFO(this->get_logger(), "EKF updated: vel=%.6f %.6f %.6f",
                    st.velocity_world.x(), st.velocity_world.y(), st.velocity_world.z());
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
    
    //imu buffer
    ImuBuffer imu_buffer_;
    
    // onnx runners
    airimu_onnx::Runner airimu_runner_;
    airio_onnx::Runner airio_runner_;
    std::unique_ptr<airio::AirioRealtimePipeline> pipeline_;
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