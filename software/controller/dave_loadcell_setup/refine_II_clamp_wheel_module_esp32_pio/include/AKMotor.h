// my_header.h

#ifndef AKMOTOR_H
#define AKMOTOR_H

#include <stdint.h>
#include <math.h>
#include <ESP32CAN.h>
#include <Arduino.h>

// #define AKMOTOR_DEBUG     /* uncomment to use serial print debug */

#ifdef AKMOTOR_DEBUG
#define DebugPrintf(f,x) 	    Serial.printf(f,x)
#define DebugPrintln(x)	        Serial.println(x)
#else
#define DebugPrintf(f,x) 	   
#define DebugPrintln(x)	  
#endif

typedef enum servo_mode {
    DutyCycleMode           = 0,
    CurrentMode             = 1,
    CurrentBrakeMode        = 2,
    VelocityMode            = 3,
    PositionMode            = 4,
    SetOriginMode           = 5,
    PositionSpeedMode       = 6,
    MITControlMode          = 8
}servo_mode_t;


struct motor_feedback_t {
    int id;
    float position;
    float velocity;
    float current;
    float torque;
    int temperature;
    int error;
    uint8_t connection_status;
};

typedef struct {
    float P_MIN;
    float P_MAX;
    float V_MIN;
    float V_MAX;
    float T_MIN;
    float T_MAX;
    float Kp_MIN;
    float Kp_MAX;
    float Kd_MIN;
    float Kd_MAX;
    float Kt;
} motor_config_t;

extern motor_config_t AK60_6_V3_0;
extern motor_config_t AK10_9_V3_0;
extern motor_config_t AK70_9_V3_0;



class AKMotor {
public:
    AKMotor(uint16_t id, motor_config_t motor_config);
    ~AKMotor();

    // Servo Mode
    void send_motor_dutycycle(float duty_cycle);
    void send_motor_current(float current);
    void send_motor_velocity(float velocity_rpm);
    void send_motor_position( float position);
    void send_motor_position_velocity(float position, int16_t velocity_erpm, int16_t acceleration_erpmps2);

    // // MIT Mode
    void send_mit_force_command(motor_config_t motor_config, float p_des, float v_des, float kp, float kd, float t_ff);

    // // Feedback
    bool update_motor();
    void unpack_reply(twai_message_t rx_message);

    motor_feedback_t motor_feedback;


private:

    uint16_t motor_id;
    motor_config_t motor_model;


    uint16_t float_to_uint(float x, float x_min, float x_max, unsigned int bits);
    float uint_to_float(int x_int, float x_min, float x_max, int bits);
    void buffer_append_int32(uint8_t* buffer, int32_t number, int32_t *index);
    void buffer_append_int16(uint8_t* buffer, int16_t number, int16_t *index);
};

#endif // AKMOTOR_H