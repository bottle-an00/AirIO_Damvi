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
 * Output: corr [1, 41, 6], cov [1, 41, 6] FP32   (name: "corr","cov")
 *
 */
struct AirImuOut {
  std::vector<float> corr;  // 41*6
  std::vector<float> cov;   // 41*6
};

class Runner {
public:
  explicit Runner(const std::string& onnx_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "airimu_onnx"),
      opt_(make_options_()),
      session_(env_, onnx_path.c_str(), opt_),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

  AirImuOut run(const std::vector<float>& feat_flat) {
    constexpr int64_t B = 1, T = 50, C = 6;
    constexpr size_t FEAT_N = static_cast<size_t>(T * C);
    constexpr size_t CORR_N = static_cast<size_t>(41 * 6);
    constexpr size_t COV_N  = static_cast<size_t>(41 * 6);

    if (feat_flat.size() != FEAT_N) {
      throw std::runtime_error("Runner::run(): feat must be 50*6 floats");
    }

    std::array<int64_t, 3> in_shape{B, T, C};

    // ORT는 입력 버퍼를 "읽기"만 하므로 const_cast는 관용적으로 사용됨.
    Ort::Value in = Ort::Value::CreateTensor<float>(
        mem_,
        const_cast<float*>(feat_flat.data()),
        feat_flat.size(),
        in_shape.data(),
        in_shape.size());

    const char* input_names[]  = {"feat"};
    const char* output_names[] = {"corr", "cov"};

    auto outs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, &in, 1,
        output_names, 2);

    if (outs.size() != 2) {
      throw std::runtime_error("Runner::run(): expected 2 outputs (corr, cov)");
    }

    // ---- corr ----
    auto corr_info  = outs[0].GetTensorTypeAndShapeInfo();
    auto corr_shape = corr_info.GetShape();
    auto corr_count = corr_info.GetElementCount();
    if (static_cast<size_t>(corr_count) != CORR_N) {
      throw std::runtime_error("Runner::run(): corr element count mismatch");
    }
    const float* corr_ptr = outs[0].GetTensorData<float>();

    // ---- cov ----
    auto cov_info  = outs[1].GetTensorTypeAndShapeInfo();
    auto cov_shape = cov_info.GetShape();
    auto cov_count = cov_info.GetElementCount();
    if (static_cast<size_t>(cov_count) != COV_N) {
      throw std::runtime_error("Runner::run(): cov element count mismatch");
    }
    const float* cov_ptr = outs[1].GetTensorData<float>();

    AirImuOut out;
    out.corr.assign(corr_ptr, corr_ptr + CORR_N);
    out.cov.assign(cov_ptr,  cov_ptr  + COV_N);
    return out;
  }

private:
  static Ort::SessionOptions make_options_() {
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(1);
    opt.SetInterOpNumThreads(1);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opt;
  }

  Ort::Env env_;
  Ort::SessionOptions opt_;
  Ort::Session session_;
  Ort::MemoryInfo mem_;
};


}  // namespace airimu_onnx
