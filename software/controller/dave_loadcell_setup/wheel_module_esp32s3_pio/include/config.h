#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>
#include <stdio.h>



// #include <string>
// #include <stdlib.h>
// #include <cstring>

// #include <esp_timer.h>
// #include <esp_log.h>

#define DEBUG_SERIAL Serial
#define DEBUG_SERIAL_BAUD 115200
#define MEASURE_TIMER_FREQ(label) do { static int64_t last_##label = 0; int64_t now_##label = esp_timer_get_time(); if (last_##label > 0) { int64_t delta_##label = now_##label - last_##label; DEBUG_SERIAL.printf("[%s] Timer delta: %lld us (%.2f Hz)\n", #label, delta_##label, 1e6 / (float)delta_##label); } last_##label = now_##label; } while (0)
#define EXECUTE_EVERY_N_MS(MS, X) do { static volatile int64_t init = -1; if (init == -1) { init = uxr_millis(); } if (uxr_millis() - init > MS) { X; init = uxr_millis(); } } while (0)


#define SSID_WIFI "REFINE2_robot_router-2.4G"
#define PASS_WIFI "11223344"
#define MICRO_ROS_AGENT_IP "10.10.0.2"
#define MICRO_ROS_AGENT_PORT 8888

// #define SSID_WIFI "VISTEC-GAIT_2.4G"
// #define PASS_WIFI "exvis123"
// #define MICRO_ROS_AGENT_IP "192.168.0.4"
// #define MICRO_ROS_AGENT_PORT 8888


// #define SSID_WIFI "HERO2.1_router_2.4GHz"
// #define PASS_WIFI "heroplusplus"
// #define MICRO_ROS_AGENT_IP "10.0.0.21"
// #define MICRO_ROS_AGENT_PORT 8888


#define EXECUTOR_HANDLE_NUMBER 16


// Pinout
// https://www.xecor.com/files/uploads/editor/b/02dc596f70c843dc9ef6fcb3b4526a40.webp
// GND on the left cannot use

// VSPI
// #define SPI_SS_PIN 5


// Old CAN pins
// #define TX_CAN_PIN GPIO_NUM_16
// #define RX_CAN_PIN GPIO_NUM_17

#define TX_CAN_PIN GPIO_NUM_17
#define RX_CAN_PIN GPIO_NUM_18


// New IMU pins
#define SDA_PIN 9
#define SCL_PIN 8

// Loadcell pins
#define HX711_1_SCK_PIN GPIO_NUM_15
#define HX711_1_DOUT_PIN GPIO_NUM_16 
#define HX711_1_OFFSET -12250.00
#define HX711_1_SCALE 1

#define HX711_2_SCK_PIN GPIO_NUM_9
#define HX711_2_DOUT_PIN GPIO_NUM_10
#define HX711_2_OFFSET 195437.00 
#define HX711_2_SCALE 1


// #define SPI_MISO_PIN 11
// #define SPI_MOSI_PIN 12
// #define SPI_SCK_PIN 13 
// #define SPI_CK_PIN 14


/**
 * ====================================
 *          Control Variables
 * ====================================
 */
// Swerve commands
float swerve_direct_command[4] = {0.0f, 0.0f, 0.0f, 0.0f};  
float swerve_driven_command[4] = {0.0f, 0.0f, 0.0f, 0.0f};  



/**
 * ====================================
 *          AK motor control
 * ====================================
 * if the servo mode cannot control, please see these videos:
 * https://www.youtube.com/watch?v=Na3yCWokKOg
 * https://www.youtube.com/watch?v=DyFzhcsc-SY
 * to update latest firmware and re-calibrate the driver.
 */
#include <ESP32CAN.h>
twai_message_t rx_message;
// Motor IDs
#define DIRECT_MODULE_1 12
#define DRIVEN_MODULE_1 11

#define DIRECT_MODULE_2 22
#define DRIVEN_MODULE_2 21

#define DIRECT_MODULE_3 31
#define DRIVEN_MODULE_3 32

#define DIRECT_MODULE_4 41
#define DRIVEN_MODULE_4 42

#include "AKMotor.h"
AKMotor direct_motor_1  (DIRECT_MODULE_1, RMD_X4_36);
AKMotor driven_motor_1  (DRIVEN_MODULE_1, RMD_X4_36_driven);
AKMotor direct_motor_2  (DIRECT_MODULE_2, RMD_X4_36);
AKMotor driven_motor_2  (DRIVEN_MODULE_2, RMD_X4_36_driven);
AKMotor direct_motor_3  (DIRECT_MODULE_3, RMD_X4_36);
AKMotor driven_motor_3  (DRIVEN_MODULE_3, RMD_X4_36_driven);
AKMotor direct_motor_4  (DIRECT_MODULE_4, RMD_X4_36);
AKMotor driven_motor_4  (DRIVEN_MODULE_4, RMD_X4_36_driven);

#include "swerve_module_control.h"
SwerveModule swerve_module_1, swerve_module_2, swerve_module_3, swerve_module_4;

float steer_target[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float drive_cmd[4]   = {0.0f, 0.0f, 0.0f, 0.0f};

float command_mode = 0.0;

bool start_signal = false;
bool check_start = false;

float tune_kp = 0.0f;
float tune_kd = 0.0f;
float tune_tff = 0.0f;

uint16_t encoder_value = 0;
RTC_DATA_ATTR uint16_t wathcdog_trigger_value = 0;

float swereve_speeed_max = 1.0;


void update_all_motor()
{
  direct_motor_1.motor_update();
  driven_motor_1.motor_update();
  direct_motor_2.motor_update();
  driven_motor_2.motor_update();
  direct_motor_3.motor_update();
  driven_motor_3.motor_update();
  direct_motor_4.motor_update();
  driven_motor_4.motor_update();
}

void read_all_motors_feedback()
{
   while (ESP32CAN_OK == ESP32Can.CANReadFrame(&rx_message)) 
    {   
        direct_motor_1.unpack_reply(rx_message);
        driven_motor_1.unpack_reply(rx_message);
        direct_motor_2.unpack_reply(rx_message);
        driven_motor_2.unpack_reply(rx_message);
        direct_motor_3.unpack_reply(rx_message);
        driven_motor_3.unpack_reply(rx_message);
        direct_motor_4.unpack_reply(rx_message);
        driven_motor_4.unpack_reply(rx_message);
    }
}

void tick_all_motors()
{
    direct_motor_1.tick_timeout();
    driven_motor_1.tick_timeout();
    direct_motor_2.tick_timeout();
    driven_motor_2.tick_timeout();
    direct_motor_3.tick_timeout();
    driven_motor_3.tick_timeout();
    direct_motor_4.tick_timeout();
    driven_motor_4.tick_timeout();
}

void run_motor()
{
    // direct_motor_1.send_motor_position(0.0f, 15.0);		// rad
    // direct_motor_2.send_motor_position(0.0f, 15.0);		// rad
    // direct_motor_3.send_motor_position(0.0f, 15.0);		// rad
    // direct_motor_4.send_motor_position(0.0f, 15.0);		// rad

    driven_motor_1.send_motor_velocity(11.0f);		      // rad/s
    driven_motor_2.send_motor_velocity(11.0f);		      // rad/s
    driven_motor_3.send_motor_velocity(11.0f);		      // rad/s
    driven_motor_4.send_motor_velocity(11.0f);		      // rad/s
}

/**
 * ====================================
 *          Force sensors
 * ====================================
 */
#include "HX711.h"
#define CONV_GRAM2NEWTON 9.80665/1000.0
HX711 force_sensor_1;
HX711 force_sensor_2;

float loadcell_force_1;
float loadcell_force_2;

float loadcell_force_x;
float loadcell_force_y;

const float C11 = -2.69774822e-05; 
const float C12 = 3.50210363e-07;
const float C21 = -8.55005606e-07;
const float C22 = -2.68001817e-05;
const float Bx = -0.29828594 - -0.3;
const float By =  0.50674671 - 1.01;

void get_forces(int32_t L1, int32_t L2, float* Fx, float* Fy) 
{
    *Fx = (C11 * L1) + (C12 * L2) + Bx;
    *Fy = (C21 * L1) + (C22 * L2) + By;
}
 

void force_sensor_1_read()
{
  if (force_sensor_1.wait_ready_timeout(5)) 
  {
    // portDISABLE_INTERRUPTS();
    loadcell_force_1 = force_sensor_1.get_units(5);  
    // DEBUG_SERIAL.printf("Loadcell 1 Force: %.2f N\n", loadcell_force_1);
    // portENABLE_INTERRUPTS();
  } 
  else 
  {
    // DEBUG_SERIAL.println("HX711 Timeout: Sensor is not ready.");
  }
}
void force_sensor_2_read()
{
  if (force_sensor_2.wait_ready_timeout(5)) 
  {
    loadcell_force_2 = force_sensor_2.get_units(5);
    // DEBUG_SERIAL.printf("Loadcell 2 Force: %.2f N\n", loadcell_force_2);
  } 
}



/**
 * ====================================
 *              Timers
 * ====================================
 */
// esp_timer_handle_t sensor_timer;
// esp_timer_handle_t motor_control_timer;
 

/**
 * ====================================
 *           Watchdog Timer
 * ====================================
 */
class WatchdogTimer
{
  public:
    explicit WatchdogTimer(uint32_t timeout_ms)
    : timeout_ms(timeout_ms), last_kick_ms(millis()) {}

    bool checktimeout()
    {
      return (uint32_t)(millis() - last_kick_ms) >= timeout_ms;
    }

    void restart_counter()
    {
      last_kick_ms = millis();
    }

  private:
    uint32_t timeout_ms;
    uint32_t last_kick_ms;
};
WatchdogTimer watchdog_cmd(750);        // stop robot
WatchdogTimer watchdog_uros(1000);      // reconnect micro-ROS
WatchdogTimer watchdog_wifi(3000);      // reset Wi-Fi
WatchdogTimer watchdog_reset(5000);     // reboot ESP

bool reconnect_ok = false;
bool wifi_reset_ok = false;
/**
 * ====================================
 *              Operations
 * ====================================
 */



#endif // CONFIG_H
