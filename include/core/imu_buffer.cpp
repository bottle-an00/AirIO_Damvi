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

bool ImuBuffer::fill_features(std::vector<float>& out) const
{
    if (size_ < capacity_) {
        return false;
    }
    out.resize(capacity_ * 7);
    // 오래된 -> 최신 순서로 채움
    for (size_t i = 0; i < capacity_; ++i) {
        size_t idx = (head_ - capacity_ + i + capacity_) % capacity_;
        const ImuSample& sample = data_[idx];
        out[i * 7 + 0] = static_cast<float>(sample.ax);
        out[i * 7 + 1] = static_cast<float>(sample.ay);
        out[i * 7 + 2] = static_cast<float>(sample.az);
        out[i * 7 + 3] = static_cast<float>(sample.gx);
        out[i * 7 + 4] = static_cast<float>(sample.gy);
        out[i * 7 + 5] = static_cast<float>(sample.gz);
        out[i * 7 + 6] = static_cast<float>(sample.dt);
    }
    return true;
}