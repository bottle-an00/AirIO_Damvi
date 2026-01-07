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

    // 모델 입력용 연속 feature 채우기
    bool fill_features(std::vector<float>& out) const;

private:
    size_t capacity_;
    std::vector<ImuSample> data_;
    size_t size_;
    size_t head_;
};

