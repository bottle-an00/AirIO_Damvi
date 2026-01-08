#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace airimu_onnx {

/**
 * AirIMU ONNX Runner
 * Input : feat [1, 50, 6] FP32   (name: "feat")
 * Output: corr [1, 41, 6] FP32   (name: "corr")
 *
 * - onnx_airimu.py의 export 규약을 그대로 따른다.
 */
class Runner {
public:
  explicit Runner(const std::string& onnx_path)
  : env_(ORT_LOGGING_LEVEL_WARNING, "airimu_onnx"),
    opt_(make_options_()),
    session_(env_, onnx_path.c_str(), opt_),
    mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

  // feat_flat: 50*6=300 floats
  // return  : corr_flat: 41*6=246 floats
  std::vector<float> run(const std::vector<float>& feat_flat) {
    constexpr int64_t B = 1, T = 50, C = 6;
    constexpr size_t FEAT_N = static_cast<size_t>(T * C);
    constexpr size_t CORR_N = static_cast<size_t>(41 * 6);

    if (feat_flat.size() != FEAT_N) {
      throw std::runtime_error("airimu_onnx::Runner::run(): feat must be 50*6 floats");
    }

    std::array<int64_t, 3> in_shape{B, T, C};

    Ort::Value in = Ort::Value::CreateTensor<float>(
        mem_,
        const_cast<float*>(feat_flat.data()),
        feat_flat.size(),
        in_shape.data(),
        in_shape.size());

    const char* input_names[]  = {"feat"};
    const char* output_names[] = {"corr"};

    auto outs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, &in, 1,
        output_names, 1);

    float* out_ptr = outs[0].GetTensorMutableData<float>();
    return std::vector<float>(out_ptr, out_ptr + CORR_N);
  }

private:
  static Ort::SessionOptions make_options_() {
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(1);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    return opt;
  }

  Ort::Env env_;
  Ort::SessionOptions opt_;
  Ort::Session session_{nullptr};
  Ort::MemoryInfo mem_;
};

}  // namespace airimu_onnx
