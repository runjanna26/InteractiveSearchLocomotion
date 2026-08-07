#include <math.h>
#include "swerve_module_control.h"

// ================= Constructor =================

SwerveModule::SwerveModule()
: last_steer_target_(0.0f)
{
}

// ================= Initialization =================

void SwerveModule::initialize(float current_steer_position)
{
    last_steer_target_ = current_steer_position;
}

// ================= Main Update =================

void SwerveModule::update(
    float wheel_speed_cmd,
    float target_angle_cmd,
    float current_steer_pos,
    float &steer_target_out,
    float &drive_cmd_out
)
{
    // ----------------------------------------------------------------------------------------
    // -------------------------------          OLD         -----------------------------------
    // ----------------------------------------------------------------------------------------

    // // Normalize logical target
    // target_angle_cmd = normalize_angle(target_angle_cmd);

    // // Current steering angle in logical space
    // float current_wrapped = normalize_angle(current_steer_pos);   // work
    // // float current_wrapped = current_steer_pos;       // work

    // // Shortest angular error
    // float error = normalize_angle(target_angle_cmd - current_wrapped);

    // // -------- 180° flip optimization --------
    // // if (fabs(error) > M_PI_2)
    // // {
    // //     target_angle_cmd = normalize_angle(target_angle_cmd + M_PI);
    // //     wheel_speed_cmd  = -wheel_speed_cmd;
    // // }

    // // -------- Low-speed steering lock --------
    // // if (fabs(wheel_speed_cmd) < steering_lock_speed_)
    // // {
    // //     target_angle_cmd = last_steer_target_;
    // // }

    // // -------- Wrapped → continuous --------
    // // steer_target = unwrap_angle(last_steer_target_, target_angle_cmd);  // buggy
    // steer_target = unwrap_angle(current_steer_pos, target_angle_cmd);

    

    // // Outputs
    // steer_target_out = steer_target;
    // drive_cmd_out    = wheel_speed_cmd;

    // // State update
    // last_steer_target_ = steer_target;


    // ----------------------------------------------------------------------------------------
    // -------------------------------          NEW         -----------------------------------
    // ----------------------------------------------------------------------------------------

    // 1. Normalize logical target to [-PI, PI]
    target_angle_cmd = normalize_angle(target_angle_cmd);

    // 2. Wrap the current steering position to [-PI, PI]
    float current_wrapped = normalize_angle(current_steer_pos);

    float error = normalize_angle(target_angle_cmd - current_wrapped);

    if (fabs(error) > 2.96f) 
    {
        target_angle_cmd = normalize_angle(target_angle_cmd + M_PI);
        wheel_speed_cmd  = -wheel_speed_cmd;
    }

    float steer_target = unwrap_angle(current_steer_pos, target_angle_cmd);

    // 4. Set Outputs
    steer_target_out = steer_target;
    drive_cmd_out    = wheel_speed_cmd;

    // 5. State update
    last_steer_target_ = steer_target;

}

// ================= Helpers =================

float SwerveModule::normalize_angle(float a)
{
    while (a > M_PI)  a -= 2.0f * M_PI;
    while (a < -M_PI) a += 2.0f * M_PI;
    return a;
}

float SwerveModule::unwrap_angle(float prev, float wrapped)
{
    float delta = normalize_angle(wrapped - prev);
    return prev + delta;
}
