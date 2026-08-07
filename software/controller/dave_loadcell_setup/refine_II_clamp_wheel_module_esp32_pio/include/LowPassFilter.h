#ifndef LOWPASS_FILTER_H
#define LOWPASS_FILTER_H

class LowPassFilter {
public:
    LowPassFilter(float alpha = 0.0f);  // Constructor with default alpha
    void reset(float value = 0.0f);     // Optional reset method
    float update(float input);         // Update with new input
    float getOutput() const;           // Access the current filtered output

private:
    float alpha_;                      // Filter coefficient (0.0 - 1.0)
    float prev_output_;                // Previous output
};

#endif // LOWPASS_FILTER_H
