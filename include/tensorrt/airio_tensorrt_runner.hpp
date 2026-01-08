#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace airio_trt {

/**
 * AirIO(codewithrot) TensorRT Runner
 * Input :
 *  - acc  (name: "acc")  FP32
 *  - gyro (name: "gyro") FP32
 *  - rot  (name: "rot")  FP32
 * Output:
 *  - cov     (name: "cov")     FP32 (shape는 엔진 기반)
 *  - net_vel (name: "net_vel") FP32 (shape는 엔진 기반)
 */
struct Output {
  std::vector<float> cov;
  std::vector<float> net_vel;
};

class Runner {
public:
  explicit Runner(const std::string& engine_path)
  : logger_(),
    runtime_(nullptr),
    engine_(nullptr),
    context_(nullptr),
    stream_(nullptr),
    d_acc_(nullptr),
    d_gyro_(nullptr),
    d_rot_(nullptr),
    d_cov_(nullptr),
    d_net_vel_(nullptr),
    acc_elems_(0),
    gyro_elems_(0),
    rot_elems_(0),
    cov_elems_(0),
    net_vel_elems_(0) {
    init_(engine_path);
  }

  ~Runner() { release_(); }

  // acc/gyro/rot flat: element count는 엔진 입력 shape에 의해 결정됨
  Output run(const std::vector<float>& acc_flat,
             const std::vector<float>& gyro_flat,
             const std::vector<float>& rot_flat) {
    if (!context_) throw std::runtime_error("airio_trt::Runner::run(): not initialized");
    if (acc_flat.size()  != acc_elems_)  throw std::runtime_error("airio_trt::Runner::run(): acc size mismatch");
    if (gyro_flat.size() != gyro_elems_) throw std::runtime_error("airio_trt::Runner::run(): gyro size mismatch");
    if (rot_flat.size()  != rot_elems_)  throw std::runtime_error("airio_trt::Runner::run(): rot size mismatch");

    Output out;
    out.cov.resize(cov_elems_);
    out.net_vel.resize(net_vel_elems_);

    checkCuda_(cudaMemcpyAsync(d_acc_,  acc_flat.data(),  acc_elems_  * sizeof(float), cudaMemcpyHostToDevice, stream_));
    checkCuda_(cudaMemcpyAsync(d_gyro_, gyro_flat.data(), gyro_elems_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
    checkCuda_(cudaMemcpyAsync(d_rot_,  rot_flat.data(),  rot_elems_  * sizeof(float), cudaMemcpyHostToDevice, stream_));

    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("airio_trt::Runner::run(): enqueueV3 failed");
    }

    checkCuda_(cudaMemcpyAsync(out.cov.data(),     d_cov_,     cov_elems_     * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    checkCuda_(cudaMemcpyAsync(out.net_vel.data(), d_net_vel_, net_vel_elems_ * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    checkCuda_(cudaStreamSynchronize(stream_));

    return out;
  }

  size_t acc_elems() const { return acc_elems_; }
  size_t gyro_elems() const { return gyro_elems_; }
  size_t rot_elems() const { return rot_elems_; }
  size_t cov_elems() const { return cov_elems_; }
  size_t net_vel_elems() const { return net_vel_elems_; }

private:
  class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
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
      if (d.d[i] < 0) throw std::runtime_error("airio_trt: dynamic dims not supported in this runner");
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
    if (blob.empty()) throw std::runtime_error("airio_trt::Runner: failed to load engine: " + engine_path);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("airio_trt::Runner: createInferRuntime failed");

    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("airio_trt::Runner: deserializeCudaEngine failed");

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("airio_trt::Runner: createExecutionContext failed");

    // 텐서 이름은 export 규약에 맞춰 acc/gyro/rot, cov/net_vel을 우선 사용
    // (만약 엔진에서 이름이 다르면 여기만 바꾸면 됨)
    const std::string acc_name = "acc";
    const std::string gyro_name = "gyro";
    const std::string rot_name = "rot";
    const std::string cov_name = "cov";
    const std::string net_vel_name = "net_vel";

    // shape 기반 element count
    acc_elems_     = volume_(engine_->getTensorShape(acc_name.c_str()));
    gyro_elems_    = volume_(engine_->getTensorShape(gyro_name.c_str()));
    rot_elems_     = volume_(engine_->getTensorShape(rot_name.c_str()));
    cov_elems_     = volume_(engine_->getTensorShape(cov_name.c_str()));
    net_vel_elems_ = volume_(engine_->getTensorShape(net_vel_name.c_str()));

    checkCuda_(cudaStreamCreate(&stream_));

    checkCuda_(cudaMalloc(&d_acc_,  acc_elems_  * sizeof(float)));
    checkCuda_(cudaMalloc(&d_gyro_, gyro_elems_ * sizeof(float)));
    checkCuda_(cudaMalloc(&d_rot_,  rot_elems_  * sizeof(float)));
    checkCuda_(cudaMalloc(&d_cov_,  cov_elems_  * sizeof(float)));
    checkCuda_(cudaMalloc(&d_net_vel_, net_vel_elems_ * sizeof(float)));

    if (!context_->setTensorAddress(acc_name.c_str(), d_acc_))
      throw std::runtime_error("airio_trt::Runner: setTensorAddress(acc) failed");
    if (!context_->setTensorAddress(gyro_name.c_str(), d_gyro_))
      throw std::runtime_error("airio_trt::Runner: setTensorAddress(gyro) failed");
    if (!context_->setTensorAddress(rot_name.c_str(), d_rot_))
      throw std::runtime_error("airio_trt::Runner: setTensorAddress(rot) failed");
    if (!context_->setTensorAddress(cov_name.c_str(), d_cov_))
      throw std::runtime_error("airio_trt::Runner: setTensorAddress(cov) failed");
    if (!context_->setTensorAddress(net_vel_name.c_str(), d_net_vel_))
      throw std::runtime_error("airio_trt::Runner: setTensorAddress(net_vel) failed");
  }

  void release_() {
    if (d_acc_) { cudaFree(d_acc_); d_acc_ = nullptr; }
    if (d_gyro_) { cudaFree(d_gyro_); d_gyro_ = nullptr; }
    if (d_rot_) { cudaFree(d_rot_); d_rot_ = nullptr; }
    if (d_cov_) { cudaFree(d_cov_); d_cov_ = nullptr; }
    if (d_net_vel_) { cudaFree(d_net_vel_); d_net_vel_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    context_.reset();
    engine_.reset();
    runtime_.reset();
    acc_elems_ = gyro_elems_ = rot_elems_ = cov_elems_ = net_vel_elems_ = 0;
  }

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context_;

  cudaStream_t stream_;
  void* d_acc_;
  void* d_gyro_;
  void* d_rot_;
  void* d_cov_;
  void* d_net_vel_;

  size_t acc_elems_;
  size_t gyro_elems_;
  size_t rot_elems_;
  size_t cov_elems_;
  size_t net_vel_elems_;
};

}  // namespace airio_trt
