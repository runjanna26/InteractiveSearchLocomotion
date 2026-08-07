#include "AKMotor.h"
#include <Arduino.h>

// Motor configuration
motor_config_t AK60_6_V3_0 = {
    .P_MIN     = -12.56f,
    .P_MAX     =  12.56f,
    .V_MIN     = -60.0f,
    .V_MAX     =  60.0f,
    .T_MIN     = -12.0f,
    .T_MAX     =  12.0f,
    .Kp_MIN    =  0,
    .Kp_MAX    =  500.0f,
    .Kd_MIN    =  0,
    .Kd_MAX    =  5.0f,
    .Kt        =  0.135,
};

motor_config_t AK10_9_V3_0 = {
    .P_MIN     = -12.56f,
    .P_MAX     =  12.56f,
    .V_MIN     = -28.0f,
    .V_MAX     =  28.0f,
    .T_MIN     = -54.0f,
    .T_MAX     =  54.0f,
    .Kp_MIN    =  0,
    .Kp_MAX    =  500.0f,
    .Kd_MIN    =  0,
    .Kd_MAX    =  5.0f,
    .Kt        =  0.16,
};

motor_config_t AK70_9_V3_0 = {
    .P_MIN     = -12.56f,
    .P_MAX     =  12.56f,
    .V_MIN     = -30.0f,
    .V_MAX     =  30.0f,
    .T_MIN     = -32.0f,
    .T_MAX     =  32.0f,
    .Kp_MIN    =  0,
    .Kp_MAX    =  500.0f,
    .Kd_MIN    =  0,
    .Kd_MAX    =  5.0f,
    .Kt        =  0.16,
};

AKMotor::AKMotor(uint16_t id, motor_config_t motor_config)
{
    motor_model = motor_config;
    motor_id = id;
}

AKMotor::~AKMotor()
{

}


/**
 * @brief Send the motor command with duty cycle in Servo mode.
 * 
 * @param duty_cycle : Desired duty cycle in the range of [0,1]
 */
void AKMotor::send_motor_dutycycle(float duty_cycle) 
{
    twai_message_t message;
    
    message.identifier = (DutyCycleMode << 8) | motor_id;
    message.extd = 1; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 4; // 4 bytes for speed data

    // Convert velocity to byte array
    int32_t send_index = 0;
    uint8_t buffer[4];
    buffer_append_int32(buffer, (int32_t)(duty_cycle * 100000.0), &send_index);
    // can transmitt
    message.data[0] = buffer[0];
    message.data[1] = buffer[1];
    message.data[2] = buffer[2];
    message.data[3] = buffer[3];

    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
}


/**
 * @brief Send the motor command with current in Servo mode.
 * 
 * @param current : Desired current range of -60 to 60 Amperes
 */
void AKMotor::send_motor_current(float current) 
{
    twai_message_t message;
    
    message.identifier = (CurrentMode << 8) | motor_id;
    message.extd = 1; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 4; // 4 bytes for speed data

    // Convert velocity to byte array
    int32_t send_index = 0;
    uint8_t buffer[4];
    buffer_append_int32(buffer, (int32_t)(current * 1000.0), &send_index);
    // can transmitt
    message.data[0] = buffer[0];
    message.data[1] = buffer[1];
    message.data[2] = buffer[2];
    message.data[3] = buffer[3];

    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
}



/**
 * @brief Send the motor command with velocity in Servo mode.
 * 
 * @param velocity_rpm : Desired velocity range from -100000 to 100000 electrical RPM.
 */
void AKMotor::send_motor_velocity(float velocity_rpm) 
{
    twai_message_t message;

    int32_t send_index = 0;
    uint8_t buffer[4];
    buffer_append_int32(buffer, (int32_t)velocity_rpm, &send_index);

    message.extd = 1;             
    message.data_length_code = 4;  
    message.identifier = (VelocityMode << 8) | motor_id;    

    message.data[0] = buffer[0];
    message.data[1] = buffer[1];
    message.data[2] = buffer[2];
    message.data[3] = buffer[3];

    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
}



/**
 * @brief Send the motor command with position in Servo mode.
 * 
 * @param position : Desired position range from -36000° to 36000°.
 */
void AKMotor::send_motor_position(float position) 
{
    twai_message_t message;
    
    message.identifier = (PositionMode << 8) | motor_id;
    message.extd = 1; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 4; // 4 bytes for position data

    // Convert position to byte array
    int32_t send_index = 0;
    uint8_t buffer[4];
    buffer_append_int32(buffer, (int32_t)(position * 10000.0), &send_index);
    // can transmitt
    message.data[0] = buffer[0];
    message.data[1] = buffer[1];
    message.data[2] = buffer[2];
    message.data[3] = buffer[3];
    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
}


/**
 * @brief Send the motor command with position, velocity and acceleration (motion control) in Servo mode.
 * 
 * @param position : Desired position range from -36000° to 36000°.
 * @param velocity_erpm : Desired velocity range from -327680 to 327680 electrical RPM.
 * @param acceleration_erpmps2 : Desired acceleration range from 0 to 327670, with 1 unit equal to 10 electrical RPM/s²
 */
void AKMotor::send_motor_position_velocity(float position, int16_t velocity_erpm, int16_t acceleration_erpmps2) 
{
    twai_message_t message;
    
    message.identifier = (PositionSpeedMode << 8) | motor_id;
    message.extd = 1; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 8; // 4 bytes for position data

    // Convert position to byte array
    int32_t send_index = 0;
    int16_t send_index1 = 4;
    uint8_t buffer[8];
    buffer_append_int32(buffer, (int32_t)(position * 10000.0), &send_index);
    buffer_append_int16(buffer, velocity_erpm/10.0, & send_index1);
    buffer_append_int16(buffer, acceleration_erpmps2/10.0, & send_index1);
    // can transmitt
    message.data[0] = buffer[0];
    message.data[1] = buffer[1];
    message.data[2] = buffer[2];
    message.data[3] = buffer[3];
    message.data[4] = buffer[4];
    message.data[5] = buffer[5];
    message.data[6] = buffer[6];
    message.data[7] = buffer[7];
    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
}




void AKMotor::send_mit_force_command(motor_config_t motor_config, float p_des, float v_des, float kp, float kd, float t_ff)
{
    twai_message_t message;

    message.identifier = (MITControlMode << 8) | motor_id;
    message.extd = 1; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 8; // 4 bytes for speed data

    p_des   = fminf(fmaxf(motor_config.P_MIN,   p_des),     motor_config.P_MAX);
    v_des   = fminf(fmaxf(motor_config.V_MIN,   v_des),     motor_config.V_MAX);
    kp      = fminf(fmaxf(motor_config.Kp_MIN,  kp),        motor_config.Kp_MAX);
    kd      = fminf(fmaxf(motor_config.Kd_MIN,  kd),        motor_config.Kd_MAX);
    t_ff    = fminf(fmaxf(motor_config.T_MIN,   t_ff),      motor_config.T_MAX);
    /// convert floats to unsigned ints ///
    uint16_t p_int   = float_to_uint(p_des, motor_config.P_MIN,     motor_config.P_MAX,     16);
    uint16_t v_int   = float_to_uint(v_des, motor_config.V_MIN,     motor_config.V_MAX,     12);
    uint16_t kp_int  = float_to_uint(kp,    motor_config.Kp_MIN,    motor_config.Kp_MAX,    12);
    uint16_t kd_int  = float_to_uint(kd,    motor_config.Kd_MIN,    motor_config.Kd_MAX,    12);
    uint16_t t_int   = float_to_uint(t_ff,  motor_config.T_MIN,     motor_config.T_MAX,     12);

    /// pack ints into the can buffer ///
    message.data[0] = kp_int >> 4;  
    message.data[1] = ((kp_int & 0xF) << 4)|( kd_int >> 8);
    message.data[2] = kd_int & 0xFF;
    message.data[3] = p_int >> 8;
    message.data[4] = p_int & 0xFF; 
    message.data[5] = v_int >> 4; 
    message.data[6] = ((v_int & 0xF) << 4)|(t_int >> 8);
    message.data[7] = t_int & 0xFF;

    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
}


bool AKMotor::update_motor()
{
    twai_message_t rx_message;
    uint32_t alerts;
    if (ESP32CAN_OK == ESP32Can.CANReadFrame(&rx_message)) 
    {  
        unpack_reply(rx_message);
    }
    else
    {
           DebugPrintf("[ID %0d] cannot read data\n", motor_id);
           motor_feedback.connection_status = 0;
    }
    return true;
}


/*
0 indicating no fault
1 indicating motor over-temperaturefault
2 indicating over-current fault
3 indicating over-voltage fault
4 indicating under-voltagefault
5 indicating encoder fault
6 indicating MOSFET over-temperature fault
7 indicatingmotor lock-up.
*/
void AKMotor::unpack_reply(twai_message_t rx_message)
{
    /// unpack ints from can buffer ///
    int id = rx_message.identifier & 0xFF; // Driver ID
    if (id == motor_id)
    {
        int16_t pos_int = ((int16_t)rx_message.data[0] << 8) | (rx_message.data[1]);
        int16_t spd_int = ((int16_t)rx_message.data[2] << 8) | (rx_message.data[3]);
        int16_t cur_int = ((int16_t)rx_message.data[4] << 8) | (rx_message.data[5]);

        /// convert ints to floats ///
        float p = (float)(pos_int * 0.1f) * 0.0174532925f;  // rad
        float v = (float)(spd_int * 10.0f) * 0.1047f;       // rad/s
        float i = (float)(cur_int * 0.01f);                 // Ampere
        int temp = rx_message.data[6];
        int error = rx_message.data[7];

        motor_feedback.id            = id;
        motor_feedback.position      = p;
        motor_feedback.velocity      = v;
        motor_feedback.current       = i;
        motor_feedback.torque        = i * motor_model.Kt;
        motor_feedback.temperature   = temp;
        motor_feedback.error         = error;
        if (motor_feedback.temperature > 0)
            motor_feedback.connection_status = 1;
        else
            motor_feedback.connection_status = 0;
    }
}







// ==================================== Static function ====================================
uint16_t AKMotor::float_to_uint(float x, float x_min, float x_max, unsigned int bits)
{
    /// Converts a float to an unsigned int, given range and number of bits ///
    float span = x_max - x_min;
    if(x < x_min) x = x_min;
    else if(x > x_max) x = x_max;
    return (uint16_t) ((x- x_min)*((float)((1<<bits)/span)));
}
float AKMotor::uint_to_float(int x_int, float x_min, float x_max, int bits){
    /// converts unsigned int to float, given range and number of bits ///
    float span = x_max - x_min;
    float offset = x_min;
    return ((float)x_int)*span/((float)((1<<bits)-1)) + offset;
}
void AKMotor::buffer_append_int32(uint8_t* buffer, int32_t number, int32_t *index) 
{
    buffer[(*index)++] = number >> 24;
    buffer[(*index)++] = number >> 16;
    buffer[(*index)++] = number >> 8;
    buffer[(*index)++] = number;
}
void AKMotor::buffer_append_int16(uint8_t* buffer, int16_t number, int16_t *index) 
{
    buffer[(*index)++] = number >> 8;
    buffer[(*index)++] = number;
}