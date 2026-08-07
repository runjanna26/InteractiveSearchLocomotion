#include "AKMotor.h"
#include <Arduino.h>
// #include "config.h"

// Motor configuration
motor_config_t RMD_X4_10 = {
    .P_MIN     = -12.5f,
    .P_MAX     =  12.5f,
    .V_MIN     = -45.0f,
    .V_MAX     =  45.0f,
    .T_MIN     = -24.0f,
    .T_MAX     =  24.0f,
    .Kp_MIN    =  0,
    .Kp_MAX    =  500.0f,
    .Kd_MIN    =  0,
    .Kd_MAX    =  5.0f,
    .Kt        =  0.85,
};

motor_config_t RMD_X4_36 = {
    .P_MIN     = -12.5f,
    .P_MAX     =  12.5f,
    .V_MIN     = -2.5f,
    .V_MAX     =  2.5f,
    .T_MIN     = -34.0f,
    .T_MAX     =  34.0f,
    .Kp_MIN    =  0,
    .Kp_MAX    =  500.0f,
    .Kd_MIN    =  0,
    .Kd_MAX    =  5.0f,
    .Kt        =  1.9,
};


motor_config_t RMD_X4_36_driven = {
    .P_MIN     = -12.5f,
    .P_MAX     =  12.5f,
    .V_MIN     = -45.0f,
    .V_MAX     =  45.0f,
    .T_MIN     = -34.0f,
    .T_MAX     =  34.0f,
    .Kp_MIN    =  0,
    .Kp_MAX    =  500.0f,
    .Kd_MIN    =  0,
    .Kd_MAX    =  5.0f,
    .Kt        =  1.9,
};


AKMotor::AKMotor(uint16_t id, motor_config_t _motor_config)
{
    motor_config = _motor_config;
    motor_id = id;
}

AKMotor::~AKMotor()
{

}



/**
 * @brief Send the motor command with velocity in Servo mode.
 * 
 * @param motor_config : Motor configuration struct.
 * @param vel_des : Desired velocity (rad/s).
 */
void AKMotor::send_motor_velocity(float vel_des)
{
    twai_message_t message;
    
    message.identifier = SINGLE + motor_id;
    message.extd = 0; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 8; // 8 bytes for speed data


    int32_t v_int = (int32_t)(vel_des* 57.29578f * 100);
    // int32_t v_int = (int32_t)(50000);

    // can transmitt
    message.data[0] = 0xA2;
    message.data[1] = 0xFF;
    message.data[2] = 0x00;
    message.data[3] = 0x00;
    message.data[4] = (v_int) & 0xFF;
    message.data[5] = (v_int >> 8) & 0xFF;
    message.data[6] = (v_int >> 16) & 0xFF;
    message.data[7] = (v_int >> 24) & 0xFF;

    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
}



/**
 * @brief Send the motor command with position in Servo mode.
 * 
 * @param motor_id : Motor driver identification
 * @param position : Desired position range from -36000° to 36000°.
 */
void AKMotor::send_motor_position(float pos_des, float vel_max)
{
    twai_message_t message;
    
    message.identifier = SINGLE + motor_id;
    message.extd = 0; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 8; // 8 bytes for position data

    int32_t p_int = (int32_t)(pos_des * 57.2957795f * 100);
    uint16_t v_max_int = (uint16_t)(vel_max * 57.2957795f); // convert rad/s to deg/s


    // can transmitt
    message.data[0] = 0xA4;
    message.data[1] = 0x00;
    message.data[2] = (v_max_int) & 0xFF;
    message.data[3] = (v_max_int >> 8) & 0xFF;
    message.data[4] = (p_int) & 0xFF;
    message.data[5] = (p_int >> 8) & 0xFF;
    message.data[6] = (p_int >> 16) & 0xFF;
    message.data[7] = (p_int >> 24) & 0xFF;

    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);

}






void AKMotor::send_mit_force_command(float p_des, float v_des, float kp, float kd, float t_ff)
{
    twai_message_t message;

    message.identifier = MITControlMode + motor_id;
    message.extd = 0; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 8; // 4 bytes for speed data

    p_des   = fminf(fmaxf(motor_config.P_MIN,   p_des),     motor_config.P_MAX);
    v_des   = fminf(fmaxf(motor_config.V_MIN,   v_des),     motor_config.V_MAX);
    kp      = fminf(fmaxf(motor_config.Kp_MIN,  kp),        motor_config.Kp_MAX);
    kd      = fminf(fmaxf(motor_config.Kd_MIN,  kd),        motor_config.Kd_MAX);
    t_ff    = fminf(fmaxf(motor_config.T_MIN,   t_ff),      motor_config.T_MAX);

    motor_feedback.kp = kp;
    motor_feedback.kd = kd;
    motor_feedback.tff = t_ff;


    /// convert floats to unsigned ints ///
    uint16_t p_int   = float_to_uint(p_des, motor_config.P_MIN, motor_config.P_MAX, 16);
    uint16_t v_int   = float_to_uint(v_des, motor_config.V_MIN, motor_config.V_MAX, 12);
    uint16_t kp_int  = float_to_uint(kp, motor_config.Kp_MIN, motor_config.Kp_MAX, 12);
    uint16_t kd_int  = float_to_uint(kd, motor_config.Kd_MIN, motor_config.Kd_MAX, 12);
    uint16_t t_int   = float_to_uint(t_ff, motor_config.T_MIN, motor_config.T_MAX, 12);

    /// pack ints into the can buffer ///
    message.data[0] = (p_int >> 8) & 0xFF;
    message.data[1] = (p_int & 0xFF);
    message.data[2] = (v_int >> 4) & 0xFF;
    message.data[3] = ((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF);
    message.data[4] = (kp_int & 0xFF);
    message.data[5] = ((kd_int >> 4) & 0xFF);
    message.data[6] = ((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF);
    message.data[7] = (t_int & 0xFF);


    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
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
void AKMotor::unpack_reply(twai_message_t &rx_message)
{
    if (rx_message.identifier == RECEIVE_ID + motor_id)
    {
        motor_feedback.connection_status = 1;
        motor_feedback.timeout_counter = 100;
        if (rx_message.data[0] == READ_MULTI_TURN_OUTPUT_SHAFT_ANGLE_ID)
        {
            int32_t raw = (rx_message.data[7] << 24 | rx_message.data[6] << 16 | rx_message.data[5] << 8 | rx_message.data[4]);
            if (raw & 0x80000000)
                raw -= 0x100000000;
            motor_feedback.position = (float)(raw * (2.0 * _PI) / (262144.0));
        }
        if (rx_message.data[0] == READ_MOTOR_STATUS_2_ID)
        {
            uint8_t temperature             = rx_message.data[1];
            int32_t torque_current_motor    = (rx_message.data[2] | (rx_message.data[3] << 8));
            int32_t velocity                = (rx_message.data[4] | (rx_message.data[5] << 8));
            int32_t position                = (rx_message.data[6] | (rx_message.data[7] << 8));
            if (torque_current_motor & 0x8000)
                torque_current_motor -= 0x10000;
            if (velocity & 0x8000)
                velocity -= 0x10000;
            if (position & 0x8000)
                position -= 0x10000;

            motor_feedback.temperature   = (float)(temperature);                             // Celsius
            motor_feedback.current       = torque_current_motor * 0.01;                      // Amperes
            motor_feedback.torque        = torque_current_motor * 0.01 * motor_config.Kt;    // Nm
            motor_feedback.velocity      = velocity * _PI / 180.0;                           // rad/s
        }
        if (rx_message.data[0] == READ_MOTOR_STATUS_1_ID)
        {
            uint16_t voltage        = rx_message.data[4] | (rx_message.data[5] << 8);
            uint16_t error          = rx_message.data[6] | (rx_message.data[7] << 8);

            motor_feedback.voltage       = voltage * 0.1;    // Volts
            motor_feedback.error         = error;
            
            
        }

    }
}

void AKMotor::tick_timeout()
{
    // If the counter is above 0, tick it down by 1
    if (motor_feedback.timeout_counter > 0) 
    {
        motor_feedback.timeout_counter--;
        
        // If it just hit 0, the timeout has triggered!
        if (motor_feedback.timeout_counter == 0) 
        {
            motor_feedback.connection_status = 0; // Mark as disconnected
        }
    }
}

void AKMotor::motor_update()
{
    request_motor_struct(motor_id, READ_MULTI_TURN_OUTPUT_SHAFT_ANGLE_ID);
    request_motor_struct(motor_id, READ_MOTOR_STATUS_1_ID);
    request_motor_struct(motor_id, READ_MOTOR_STATUS_2_ID);
}

void AKMotor::request_motor_struct(uint32_t motor_id, uint16_t request_id)
{
    twai_message_t message;
    
    message.identifier = SINGLE + motor_id; 
    message.extd = 0; // Extended Frame
    message.rtr = 0;  // Remote Frame
    message.data_length_code = 8; 

    // can transmitt
    message.data[0] = request_id;
    message.data[1] = 0x00;
    message.data[2] = 0x00;
    message.data[3] = 0x00;
    message.data[4] = 0x00;
    message.data[5] = 0x00;
    message.data[6] = 0x00;
    message.data[7] = 0x00;

    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);

}


void AKMotor::motor_reboot() 
{
    twai_message_t message;
    
    message.identifier = SINGLE + motor_id;
    message.extd = 0; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 8; 

    // can transmitt
    message.data[0] = 0x76;
    message.data[1] = 0x00;
    message.data[2] = 0x00;
    message.data[3] = 0x00;
    message.data[4] = 0x00;
    message.data[5] = 0x00;
    message.data[6] = 0x00;
    message.data[7] = 0x00;

    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
    else 
    {
        DebugPrintf("[ID %0d] finish reboot\n", motor_id);
    }
}

void AKMotor::motor_stop() 
{
    twai_message_t message;
    
    message.identifier = SINGLE + motor_id;
    message.extd = 0; // Extended Frame
    message.rtr = 0;  // Data Frame
    message.data_length_code = 8; 

    // can transmitt
    message.data[0] = 0x81;
    message.data[1] = 0x00;
    message.data[2] = 0x00;
    message.data[3] = 0x00;
    message.data[4] = 0x00;
    message.data[5] = 0x00;
    message.data[6] = 0x00;
    message.data[7] = 0x00;

    // Send the TWAI (CAN) message
    if (ESP32CAN_OK != ESP32Can.CANWriteFrame(&message)) 
        DebugPrintf("[ID %0d] cannot send command\n", motor_id);
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