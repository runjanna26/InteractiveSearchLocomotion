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

#define _PI 3.14159265359


typedef enum servo_mode {
    DutyCycleMode           = 0,
    CurrentMode             = 1,
    CurrentBrakeMode        = 2,
    VelocityMode            = 3,
    PositionMode            = 4,
    SetOriginMode           = 5,
    PositionSpeedMode       = 6,
    MITControlMode          = 0x400
}servo_mode_t;

struct motor_feedback_t {
    int id;
    float position;
    float velocity;
    float current;
    float voltage;
    float torque;
    int temperature;
    uint16_t error;

    float kp;
    float kd;
    float tff;

    uint8_t connection_status;
    uint8_t connection_error;

    int timeout_counter;

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

extern motor_config_t RMD_X4_10;
extern motor_config_t RMD_X4_36;
extern motor_config_t RMD_X4_36_driven;

#define SINGLE  0x140
#define MULTI  0x240
#define RECEIVE_ID 0x240
#define READ_MULTI_TURN_OUTPUT_SHAFT_ANGLE_ID   0x60
#define READ_SINGLE_TURN_OUTPUT_SHAFT_ANGLE_ID  0x94
#define READ_MOTOR_STATUS_1_ID                  0x9A
#define READ_MOTOR_STATUS_2_ID                  0x9C
#define READ_MOTOR_STATUS_3_ID                  0x9D
#define READ_MOTOR_MODEL_ID                     0xB5
#define READ_ACCELATION_ID                      0x42

class AKMotor {
public:
    AKMotor(uint16_t id, motor_config_t motor_config);
    ~AKMotor();

    // Servo Mode
    void send_motor_velocity(float vel_des);
    void send_motor_position(float pos_des, float vel_max);

    // MIT Mode
    void send_mit_force_command(float p_des, float v_des, float kp, float kd, float t_ff);

    // Feedback
    void motor_update();
    void unpack_reply(twai_message_t &rx_message);


    void motor_reboot();
    void motor_stop();

    motor_feedback_t motor_feedback;

    void tick_timeout();
private:

    uint32_t motor_id;
    motor_config_t motor_config;

    void request_motor_struct(uint32_t motor_id, uint16_t request_id);


    uint16_t float_to_uint(float x, float x_min, float x_max, unsigned int bits);
    float uint_to_float(int x_int, float x_min, float x_max, int bits);
    void buffer_append_int32(uint8_t* buffer, int32_t number, int32_t *index);
    void buffer_append_int16(uint8_t* buffer, int16_t number, int16_t *index);
};

#endif // AKMOTOR_H