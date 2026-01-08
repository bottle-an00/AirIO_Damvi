#include "core/imu_buffer.hpp"

ImuBuffer::ImuBuffer(size_t buffer_len)
    : capacity_(buffer_len), data_(buffer_len), size_(0), head_(0)
{
}

void ImuBuffer::push(const ImuSample& sample)
{
    data_[head_] = sample;
    head_ = (head_ + 1) % capacity_;
    if (size_ < capacity_) {
        ++size_;
    }
}

size_t ImuBuffer::size() const
{
    return size_;
}

bool ImuBuffer::full() const
{
    return size_ == capacity_;
}

void ImuBuffer::clear()
{
    size_ = 0;
    head_ = 0;
}

bool ImuBuffer::get_sequence(std::vector<ImuSample>& out) const
{
    if (size_ < capacity_) {
        return false;
    }
    out.resize(capacity_);
    // 오래된 -> 최신 순서로 채움
    for (size_t i = 0; i < capacity_; ++i) {
        size_t idx = (head_ - capacity_ + i + capacity_) % capacity_;
        out[i] = data_[idx];
    }
    return true;
}

bool ImuBuffer::fill_feat_flat(std::vector<float>& out) const
{
    if (size_ < capacity_) return false;

    const size_t T = capacity_;
    out.resize(T * 6);

    // oldest -> newest
    for (size_t i = 0; i < T; ++i) {
        const size_t idx = (head_ - T + i + T) % T;  // capacity_ == T
        const ImuSample& s = data_[idx];

        out[i*6 + 0] = static_cast<float>(s.ax);
        out[i*6 + 1] = static_cast<float>(s.ay);
        out[i*6 + 2] = static_cast<float>(s.az);
        out[i*6 + 3] = static_cast<float>(s.gx);
        out[i*6 + 4] = static_cast<float>(s.gy);
        out[i*6 + 5] = static_cast<float>(s.gz);
    }
    return true;
}

bool ImuBuffer::fill_acc_flat(std::vector<float>& out) const
{
    if (size_ < capacity_) return false;

    const size_t T = capacity_;
    out.resize(T * 3);

    for (size_t i = 0; i < T; ++i) {
        const size_t idx = (head_ - T + i + T) % T;
        const ImuSample& s = data_[idx];

        out[i*3 + 0] = static_cast<float>(s.ax);
        out[i*3 + 1] = static_cast<float>(s.ay);
        out[i*3 + 2] = static_cast<float>(s.az);
    }
    return true;
}

bool ImuBuffer::fill_gyro_flat(std::vector<float>& out) const
{
    if (size_ < capacity_) return false;

    const size_t T = capacity_;
    out.resize(T * 3);

    for (size_t i = 0; i < T; ++i) {
        const size_t idx = (head_ - T + i + T) % T;
        const ImuSample& s = data_[idx];

        out[i*3 + 0] = static_cast<float>(s.gx);
        out[i*3 + 1] = static_cast<float>(s.gy);
        out[i*3 + 2] = static_cast<float>(s.gz);
    }
    return true;
}

bool ImuBuffer::fill_dt_flat(std::vector<float>& out) const
{
    if (size_ < capacity_) return false;

    const size_t T = capacity_;
    out.resize(T);

    for (size_t i = 0; i < T; ++i) {
        const size_t idx = (head_ - T + i + T) % T;
        const ImuSample& s = data_[idx];

        out[i] = static_cast<float>(s.dt);
    }
    return true;
}