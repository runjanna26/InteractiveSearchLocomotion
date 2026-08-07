#ifndef SWERVE_MODULE_H
#define SWERVE_MODULE_H

#include <cmath>

class SwerveModule
{
public:
    SwerveModule();

    // Call once AFTER steering encoder feedback is valid
    void initialize(float current_steer_position);

    // Main update step (call every control loop)
    void update(
        float wheel_speed_cmd,     // rad/s
        float target_angle_cmd,    // [-pi, pi]
        float current_steer_pos,   // multi-turn encoder
        float &steer_target_out,   // multi-turn position command
        float &drive_cmd_out       // rad/s
    );

private:
    float last_steer_target_;
    const float steering_lock_speed_ = 0.05f;
    float steer_target;

    // ---------- helpers ----------
    static float normalize_angle(float a);
    static float unwrap_angle(float prev, float wrapped);
};

#endif // SWERVE_MODULE_H
