#include "LowPassFilter.h"

LowPassFilter::LowPassFilter(float alpha) 
    : alpha_(alpha), prev_output_(0.0f) {}

void LowPassFilter::reset(float value) {
    prev_output_ = value;
}

float LowPassFilter::update(float input) {
    prev_output_ = alpha_ * input + (1.0f - alpha_) * prev_output_;
    return prev_output_;
}

float LowPassFilter::getOutput() const {
    return prev_output_;
}
