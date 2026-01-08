#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace airio_onnx {

/**
 * AirIO(codewithrot) ONNX Runner
 * Input :
 *  - acc  [1, 50, 3] FP32 (name: "acc")
 *  - gyro [1, 50, 3] FP32 (name: "gyro")
 *  - rot  [1, 50, 3] FP32 (name: "rot")
 * Output:
 *  - cov     (name: "cov")     shape는 모델에 따라 다를 수 있어 element count로 반환
 *  - net_vel (name: "net_vel") shape는 모델에 따라 다를 수 있어 element count로 반환
 *
 * - onnx_airio.py의 export 규약을 그대로 따른다.
 */
struct Output {
  std::vector<float> cov;
  std::vector<float> net_vel;
};

class Runner {
public:
  explicit Runner(const std::string& onnx_path)
  : env_(ORT_LOGGING_LEVEL_WARNING, "airio_onnx"),
    opt_(make_options_()),
    session_(env_, onnx_path.c_str(), opt_),
    mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

  // acc/gyro/rot flat: 각각 50*3=150 floats
  Output run(const std::vector<float>& acc_flat,
             const std::vector<float>& gyro_flat,
             const std::vector<float>& rot_flat) {
    constexpr int64_t B = 1, T = 50, C = 3;
    constexpr size_t N = static_cast<size_t>(T * C);

    if (acc_flat.size()  != N) throw std::runtime_error("airio_onnx::Runner::run(): acc must be 50*3 floats");
    if (gyro_flat.size() != N) throw std::runtime_error("airio_onnx::Runner::run(): gyro must be 50*3 floats");
    if (rot_flat.size()  != N) throw std::runtime_error("airio_onnx::Runner::run(): rot must be 50*3 floats");

    std::array<int64_t, 3> shape{B, T, C};

    Ort::Value acc = Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(acc_flat.data()),  acc_flat.size(),  shape.data(), shape.size());
    Ort::Value gyr = Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(gyro_flat.data()), gyro_flat.size(), shape.data(), shape.size());
    Ort::Value rot = Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(rot_flat.data()),  rot_flat.size(),  shape.data(), shape.size());

    const char* input_names[]  = {"acc", "gyro", "rot"};
    const char* output_names[] = {"cov", "net_vel"};

    std::array<Ort::Value, 3> inputs = {std::move(acc), std::move(gyr), std::move(rot)};

    auto outs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), inputs.size(),
        output_names, 2);

    Output out;
    out.cov     = copy_out_(outs[0]);
    out.net_vel = copy_out_(outs[1]);
    return out;
  }

private:
  static Ort::SessionOptions make_options_() {
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(1);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    return opt;
  }

  static std::vector<float> copy_out_(Ort::Value& v) {
    auto ti = v.GetTensorTypeAndShapeInfo();
    size_t n = ti.GetElementCount();
    float* p = v.GetTensorMutableData<float>();
    return std::vector<float>(p, p + n);
  }

  Ort::Env env_;
  Ort::SessionOptions opt_;
  Ort::Session session_{nullptr};
  Ort::MemoryInfo mem_;
};

}  // namespace airio_onnx
