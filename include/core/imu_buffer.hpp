#pragma once
#include <vector>
#include <cstddef>

struct ImuSample
{
    double ax, ay, az;  // linear acceleration
    double gx, gy, gz;  // angular velocity
    double dt;          // time delta (seconds)
};

class ImuBuffer
{
public:
    explicit ImuBuffer(size_t buffer_len);
    ~ImuBuffer() = default;

    ImuBuffer(const ImuBuffer &) = delete;
    ImuBuffer &operator=(const ImuBuffer &) = delete;

    // Core API
    void push(const ImuSample& sample);
    size_t size() const;
    bool full() const;
    void clear();

    // Sequence 조회
    bool get_sequence(std::vector<ImuSample>& out) const;

    // ONNX 입력용 flat 벡터 추출 (oldest -> newest, 길이 = capacity_)
    bool fill_feat_flat(std::vector<float>& out) const;  // [T*6]  ax..gz
    bool fill_acc_flat(std::vector<float>& out) const;   // [T*3]  ax..az
    bool fill_gyro_flat(std::vector<float>& out) const;  // [T*3]  gx..gz
    bool fill_dt_flat(std::vector<float>& out) const;    // [T]    dt

private:
    size_t capacity_;
    std::vector<ImuSample> data_;
    size_t size_;
    size_t head_;
};

