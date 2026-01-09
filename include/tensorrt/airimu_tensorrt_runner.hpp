#pragma once

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace airimu_trt {

  struct AirImuOut {
  std::vector<float> corr;  // engine output "corr"
  std::vector<float> cov;   // engine output "cov"
};

/**
 * AirIMU TensorRT Runner
 * Input : feat (name: "feat")  FP32, 엔진 shape 기반 element count 사용
 * Output: corr (name: "corr")  FP32, 엔진 shape 기반 element count 사용
 *
 * - .engine(serialize된 TensorRT 엔진)을 로드해서 실행한다.
 * - 엔진이 고정 shape여도, 나중에 고정값이 바뀌면 새 엔진만 만들면 여기 코드는 그대로 동작한다.
 */
class Runner {
public:
  explicit Runner(const std::string& engine_path)
  : logger_(),
    runtime_(nullptr),
    engine_(nullptr),
    context_(nullptr),
    stream_(nullptr),
    d_feat_(nullptr),
    d_corr_(nullptr),
    feat_elems_(0),
    corr_elems_(0),
    cov_elems_(0) {
    init_(engine_path);
  }

  ~Runner() { release_(); }

  AirImuOut run(const std::vector<float>& feat_flat) {
    if (!context_) throw std::runtime_error("airimu_trt::Runner::run(): not initialized");
    if (feat_flat.size() != feat_elems_) {
      throw std::runtime_error("airimu_trt::Runner::run(): feat size mismatch. expected=" +
                               std::to_string(feat_elems_) + " got=" + std::to_string(feat_flat.size()));
    }

    std::vector<float> corr_flat(corr_elems_);

    // H2D
    checkCuda_(cudaMemcpyAsync(d_feat_, feat_flat.data(),
                               feat_elems_ * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));

    // enqueue
    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("airimu_trt::Runner::run(): enqueueV3 failed");
    }


    AirImuOut out;
    out.corr.resize(corr_elems_);
    out.cov.resize(cov_elems_);

    // D2H
    checkCuda_(cudaMemcpyAsync(out.corr.data(), d_corr_,
                               corr_elems_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    checkCuda_(cudaMemcpyAsync(out.cov.data(), d_cov_,
                               cov_elems_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    checkCuda_(cudaStreamSynchronize(stream_));

    return out;
  }

  size_t feat_elems() const { return feat_elems_; }
  size_t corr_elems() const { return corr_elems_; }
  size_t cov_elems()  const { return cov_elems_; }

private:
  class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
      // WARNING 이상만 출력
      if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << "\n";
    }
  };

  struct TRTDestroy {
    template <typename T>
    void operator()(T* p) const noexcept { if (p) delete p; }
  };

  static std::vector<char> loadFile_(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
  }

  static size_t volume_(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
      if (d.d[i] < 0) throw std::runtime_error("airimu_trt: dynamic dims not supported in this runner");
      v *= static_cast<size_t>(d.d[i]);
    }
    return v;
  }

  static void checkCuda_(cudaError_t e) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
    }
  }

  void init_(const std::string& engine_path) {
    auto blob = loadFile_(engine_path);
    if (blob.empty()) throw std::runtime_error("airimu_trt::Runner: failed to load engine: " + engine_path);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("airimu_trt::Runner: createInferRuntime failed");

    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("airimu_trt::Runner: deserializeCudaEngine failed");

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("airimu_trt::Runner: createExecutionContext failed");

    // IO tensor name 찾기 (첫 input/첫 output)
    std::string feat_name;
    std::string corr_name;
    std::string cov_name;

    const int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
      const char* n = engine_->getIOTensorName(i);
      auto mode = engine_->getTensorIOMode(n);

      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        if (feat_name.empty()) feat_name = n;
      } else {
        // outputs: try match by name first
        std::string ns(n);
        if (ns == "corr") corr_name = ns;
        else if (ns == "cov") cov_name = ns;
      }
    }

    // fallback: if names not exactly "corr"/"cov", pick first/second output deterministically
    if (corr_name.empty() || cov_name.empty()) {
      std::vector<std::string> outs;
      outs.reserve(2);
      for (int i = 0; i < nb; ++i) {
        const char* n = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(n) == nvinfer1::TensorIOMode::kOUTPUT) {
          outs.emplace_back(n);
        }
      }
      if (outs.size() < 2) {
        throw std::runtime_error("airimu_trt::Runner: expected 2 outputs (corr,cov) but found " + std::to_string(outs.size()));
      }
      if (corr_name.empty()) corr_name = outs[0];
      if (cov_name.empty())  cov_name  = outs[1];
    }

    if (feat_name.empty() || corr_name.empty() || cov_name.empty()) {
      throw std::runtime_error("airimu_trt::Runner: failed to find required IO tensors (feat,corr,cov)");
    }

    // shape 기반 element count
    auto feat_dims = engine_->getTensorShape(feat_name.c_str());
    auto corr_dims = engine_->getTensorShape(corr_name.c_str());
    auto cov_dims  = engine_->getTensorShape(cov_name.c_str());

    feat_elems_ = volume_(feat_dims);
    corr_elems_ = volume_(corr_dims);
    cov_elems_  = volume_(cov_dims);

    // CUDA resources
    checkCuda_(cudaStreamCreate(&stream_));
    checkCuda_(cudaMalloc(&d_feat_, feat_elems_ * sizeof(float)));
    checkCuda_(cudaMalloc(&d_corr_, corr_elems_ * sizeof(float)));
    checkCuda_(cudaMalloc(&d_cov_,  cov_elems_  * sizeof(float)));

    // tensor address bind
    if (!context_->setTensorAddress(feat_name.c_str(), d_feat_))
      throw std::runtime_error("airimu_trt::Runner: setTensorAddress(feat) failed");
    if (!context_->setTensorAddress(corr_name.c_str(), d_corr_))
      throw std::runtime_error("airimu_trt::Runner: setTensorAddress(corr) failed");
    if (!context_->setTensorAddress(cov_name.c_str(), d_cov_))
      throw std::runtime_error("airimu_trt::Runner: setTensorAddress(cov) failed");
  }

  void release_() {
    if (d_feat_) { cudaFree(d_feat_); d_feat_ = nullptr; }
    if (d_corr_) { cudaFree(d_corr_); d_corr_ = nullptr; }
    if (d_cov_)  { cudaFree(d_cov_ ); d_cov_  = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }

    context_.reset();
    engine_.reset();
    runtime_.reset();
    feat_elems_ = corr_elems_ = cov_elems_ = 0;
  }

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context_;

  cudaStream_t stream_;
  void* d_feat_;
  void* d_corr_;
  void* d_cov_;

  size_t feat_elems_;
  size_t corr_elems_;
  size_t cov_elems_;
};

}  // namespace airimu_trt
