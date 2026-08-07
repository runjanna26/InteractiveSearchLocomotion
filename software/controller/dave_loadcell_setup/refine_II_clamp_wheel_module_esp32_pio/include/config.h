



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

// #define SSID_WIFI "runj-wifi"
// #define PASS_WIFI "11223344"
// #define AGENT_IP "10.42.0.1"

#define SSID_WIFI "REFINE2_robot_router-2.4G"
#define PASS_WIFI "11223344"
#define AGENT_IP "10.10.0.2"
#define AGENT_PORT 8888

#define EXECUTOR_HANDLE_NUMBER 10


class WatchdogTimer
{
public:
  WatchdogTimer(int32_t interval) : watchdog_time(0), timestep(interval) {}

  bool checktimeout()
  {
    watchdog_time++;
    if (watchdog_time > timestep)
    {
      timeout = true;
    }
    return timeout;
  }
  void restart_counter()
  {
    watchdog_time = 0;
    timeout = false;
  }


  int32_t watchdog_time;
  int32_t timestep;
  bool timeout = false;
};

int32_t watchdog_time = 0;
int32_t last_time = 0;
WatchdogTimer watchdogtime(200); 
WatchdogTimer watchdogtime_restartESP(1000);
bool using_connectioncheck = true;
// Pinout
// https://www.xecor.com/files/uploads/editor/b/02dc596f70c843dc9ef6fcb3b4526a40.webp
// GND on the left cannot use

// VSPI
#define SPI_SS_PIN 5
#define SPI_MISO_PIN 19
#define SPI_MOSI_PIN 23
#define SPI_SCK_PIN 18


#define TX_CAN_PIN GPIO_NUM_17
#define RX_CAN_PIN GPIO_NUM_16

#define HX711_1_SCK_PIN 14
#define HX711_1_DOUT_PIN 27 
#define HX711_1_OFFSET 0
#define HX711_1_SCALE 121.34

#define HX711_2_SCK_PIN 26
#define HX711_2_DOUT_PIN 25
#define HX711_2_OFFSET 0  // 40673.0
#define HX711_2_SCALE 1

#define HX711_3_SCK_PIN 33
#define HX711_3_DOUT_PIN 32
#define HX711_3_OFFSET 1434593 //1680600, 1773914, 1513165.0, 1434593.0
#define HX711_3_SCALE 700098.0/1500.0

/**
 * ====================================
 *          Control Variables
 * ====================================
 */
#define WHEEL_VELOCITY_MAX 40000
float wheel_vel = 0;
float wheel_vel_cmd;


#define CLAMP_VELOCITY_MAX 5000
#define CLAMP_ACCELERATION_MAX 50000
int clamp_dir;         // 1, 0,-1
float clamp_vel_cmd;       // 

#define CLEAN_VELOCITY_MAX 255
float clean_vol;
int clean_cmd;


int led_state = 0;
bool led_cmd_1 = false;
bool led_cmd_2 = false;
bool led_cmd_3 = false;
bool led_cmd_4 = false;

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
// Motor IDs
#define CLAMP_MOTOR 1
#define LEFT_WHEEL_MOTOR 2
#define RIGHT_WHEEL_MOTOR 3

#include "AKMotor.h"
AKMotor clamp_motor       (CLAMP_MOTOR,       AK60_6_V3_0);
AKMotor left_wheel_motor  (LEFT_WHEEL_MOTOR,  AK60_6_V3_0);
AKMotor right_wheel_motor (RIGHT_WHEEL_MOTOR, AK60_6_V3_0);
 
/**
 * ====================================
 *          Force sensors
 * ====================================
 */
#include "HX711.h"
// float grav_acc = 9.80665;
#define CONV_GRAM2NEWTON 9.80665/1000.0
HX711 force_sensor_1;
HX711 force_sensor_2;
HX711 force_sensor_3;

float clamp_force_1;
float clamp_force_2;
float clamp_force_3;
 
/**
 * ====================================
 *          Current sensors
 * ====================================
 */
#include <Adafruit_INA219.h>
Adafruit_INA219 INA219_DC_1(0x40);
Adafruit_INA219 INA219_DC_2(0x41);

float dc_current_2;

void dc_motor_current_read()
{
  dc_current_2 = INA219_DC_1.getCurrent_mA();

    // shuntvoltage = INA219_DC_1.getShuntVoltage_mV();
    // busvoltage = INA219_DC_1.getBusVoltage_V();
    // current_mA = INA219_DC_1.getCurrent_mA();
    // power_mW = INA219_DC_1.getPower_mW();

    // DEBUG_SERIAL.print("getShuntVoltage_mV:   "); 
    // DEBUG_SERIAL.print(INA219_DC_1.getShuntVoltage_mV());
    // DEBUG_SERIAL.println();

    // DEBUG_SERIAL.print(" mV, getBusVoltage_V:   ");
    // DEBUG_SERIAL.print(INA219_DC_1.getBusVoltage_V());
    // DEBUG_SERIAL.println();

    // DEBUG_SERIAL.print("getCurrent_mA:   ");
    // DEBUG_SERIAL.print(INA219_DC_1.getCurrent_mA());
    // DEBUG_SERIAL.println();

    // DEBUG_SERIAL.print(" mA, getPower_mW:   ");     
    // DEBUG_SERIAL.print(INA219_DC_1.getPower_mW());
    // DEBUG_SERIAL.println();
    
}

/**
 * ====================================
 *              Encoder
 * ====================================
 */

#include <AS5X47.h>
AS5X47 encoder(SPI_SS_PIN);
float clamp_angle_offset = 2.4325101375579834; // radian

float clamp_angle;
float clamp_angle_init;


/**
 * ====================================
 *                IMU
 * ====================================
 */
// #define ICM_20948_USE_DMP 

#include "ICM_20948.h"
#define WIRE_PORT Wire
#define AD0_VAL 1

typedef struct
{
    float accel_x;
    float accel_y;
    float accel_z;

    float gyro_x;
    float gyro_y;
    float gyro_z;

    float mag_x;
    float mag_y;
    float mag_z;

    float roll;
    float pitch;
    float yaw;

    float temperature;

    uint32_t timestamp_ms;  // Optional: time in milliseconds when data captured
} imu_data_t;

ICM_20948_I2C myICM;
icm_20948_DMP_data_t imu_data;
imu_data_t imu_recv_data;

int imu_ini_count = 0;

void initializeIMU()
{
  bool initialized = false;
  while (!initialized)
  {
    // Initialize the ICM-20948
    // If the DMP is enabled, .begin performs a minimal startup. We need to configure the sample mode etc. manually.
    DEBUG_SERIAL.print(F("Initialization of the sensor returned: "));
    DEBUG_SERIAL.println(myICM.statusString());
    myICM.begin(WIRE_PORT, AD0_VAL);
    if (myICM.status != ICM_20948_Stat_Ok)
    {
      delay(500);
      imu_ini_count = imu_ini_count + 1;
      if (imu_ini_count > 4)
      {
        imu_ini_count = 0;
        break;
      }
    }
    else
    {
      initialized = true;
    }
  }
  bool success = true;
  success &= (myICM.initializeDMP() == ICM_20948_Stat_Ok);
  DEBUG_SERIAL.printf("initializeDMP() %d\n",success);

  // Enable the DMP orientation sensor
  success &= (myICM.enableDMPSensor(INV_ICM20948_SENSOR_ORIENTATION) == ICM_20948_Stat_Ok);
  DEBUG_SERIAL.println(success);
  
  // Enable any additional sensors / features
  success &= (myICM.enableDMPSensor(INV_ICM20948_SENSOR_RAW_GYROSCOPE) == ICM_20948_Stat_Ok);
  success &= (myICM.enableDMPSensor(INV_ICM20948_SENSOR_RAW_ACCELEROMETER) == ICM_20948_Stat_Ok);
  success &= (myICM.enableDMPSensor(INV_ICM20948_SENSOR_MAGNETIC_FIELD_UNCALIBRATED) == ICM_20948_Stat_Ok);
  DEBUG_SERIAL.println(success);

  success &= (myICM.setDMPODRrate(DMP_ODR_Reg_Quat9, 0) == ICM_20948_Stat_Ok);        // Set to the maximum
  success &= (myICM.setDMPODRrate(DMP_ODR_Reg_Accel, 0) == ICM_20948_Stat_Ok);        // Set to the maximum
  success &= (myICM.setDMPODRrate(DMP_ODR_Reg_Gyro, 0) == ICM_20948_Stat_Ok);         // Set to the maximum
  success &= (myICM.setDMPODRrate(DMP_ODR_Reg_Gyro_Calibr, 0) == ICM_20948_Stat_Ok);  // Set to the maximum
  success &= (myICM.setDMPODRrate(DMP_ODR_Reg_Cpass, 0) == ICM_20948_Stat_Ok);        // Set to the maximum
  success &= (myICM.setDMPODRrate(DMP_ODR_Reg_Cpass_Calibr, 0) == ICM_20948_Stat_Ok); // Set to the maximum
  DEBUG_SERIAL.println(success);
  // Enable the FIFO
  success &= (myICM.enableFIFO() == ICM_20948_Stat_Ok);
  DEBUG_SERIAL.println(success);

  // Enable the DMP
  success &= (myICM.enableDMP() == ICM_20948_Stat_Ok);
  DEBUG_SERIAL.println(success);

  // Reset DMP
  success &= (myICM.resetDMP() == ICM_20948_Stat_Ok);
  DEBUG_SERIAL.println(success);

  // Reset FIFO
  success &= (myICM.resetFIFO() == ICM_20948_Stat_Ok);

  
  DEBUG_SERIAL.println(success);
  if (success)
  {
    DEBUG_SERIAL.println(F("DMP enabled!"));
  }
  else
  {
    DEBUG_SERIAL.println(F("Enable DMP failed!"));
    DEBUG_SERIAL.println(F("Please check that you have uncommented line 29 (#define ICM_20948_USE_DMP) in util/ICM_20948_C.h..."));
    while (1)
      ; // Do nothing more
  }

}

void imu_read()
{
  // MEASURE_TIMER_FREQ(imu_read);
  myICM.readDMPdataFromFIFO(&imu_data);

  if ((myICM.status == ICM_20948_Stat_Ok) || (myICM.status == ICM_20948_Stat_FIFOMoreDataAvail)) // imu_data available
  {
    if ((imu_data.header & DMP_header_bitmap_Quat9) > 0) // We have asked for orientation imu_data so we should receive Quat9
    {

      double q1 = ((double)imu_data.Quat9.Data.Q1) / 1073741824.0; // Convert to double. Divide by 2^30
      double q2 = ((double)imu_data.Quat9.Data.Q2) / 1073741824.0; // Convert to double. Divide by 2^30
      double q3 = ((double)imu_data.Quat9.Data.Q3) / 1073741824.0; // Convert to double. Divide by 2^30
      double q0 = sqrt(1.0 - ((q1 * q1) + (q2 * q2) + (q3 * q3)));

      double q2sqr = q2 * q2;

      // roll (x-axis rotation)
      double t0 = +2.0 * (q0 * q1 + q2 * q3);
      double t1 = +1.0 - 2.0 * (q1 * q1 + q2sqr);
      double roll = atan2(t0, t1) * 180.0 / PI;

      // pitch (y-axis rotation)
      double t2 = +2.0 * (q0 * q2 - q3 * q1);
      t2 = t2 > 1.0 ? 1.0 : t2;
      t2 = t2 < -1.0 ? -1.0 : t2;
      double pitch = asin(t2) * 180.0 / PI;

      // yaw (z-axis rotation)
      double t3 = +2.0 * (q0 * q3 + q1 * q2);
      double t4 = +1.0 - 2.0 * (q2sqr + q3 * q3);
      double yaw = atan2(t3, t4) * 180.0 / PI;

      if (!isnan(q0) && !isnan(q1) && !isnan(q2) && !isnan(q3))
      {
        // DEBUG_SERIAL.printf("Quat9: %f %f %f %f\n", q0, q1, q2, q3);
        // DEBUG_SERIAL.printf("Roll: %f Pitch: %f Yaw: %f\n", roll, pitch, yaw);
        imu_recv_data.roll = roll;
        imu_recv_data.pitch = pitch;
        imu_recv_data.yaw = yaw;
      }
    }

    // Read Accelerometer, Gyroscope and Compass data
    if ((imu_data.header & DMP_header_bitmap_Accel) > 0) 
    {
      // Extract the raw accelerometer imu_data
      float acc_x = (float)imu_data.Raw_Accel.Data.X; 
      float acc_y = (float)imu_data.Raw_Accel.Data.Y;
      float acc_z = (float)imu_data.Raw_Accel.Data.Z;
      // DEBUG_SERIAL.printf("Accel: %f %f %f\n", acc_x, acc_y, acc_z);
      imu_recv_data.accel_x = acc_x;
      imu_recv_data.accel_y = acc_y;
      imu_recv_data.accel_z = acc_z;
    }

    if ((imu_data.header & DMP_header_bitmap_Gyro) > 0) 
    {
      // Extract the raw gyro imu_data
      float gy_x = (float)imu_data.Raw_Gyro.Data.X; 
      float gy_y = (float)imu_data.Raw_Gyro.Data.Y;
      float gy_z = (float)imu_data.Raw_Gyro.Data.Z;
      // DEBUG_SERIAL.printf("Gyro: %f %f %f\n", gy_x, gy_y, gy_z);
      imu_recv_data.gyro_x = gy_x;
      imu_recv_data.gyro_y = gy_y;
      imu_recv_data.gyro_z = gy_z;
    }

    if ((imu_data.header & DMP_header_bitmap_Compass) > 0) 
    {
      float mag_x = (float)imu_data.Compass.Data.X; 
      float mag_y = (float)imu_data.Compass.Data.Y;
      float mag_z = (float)imu_data.Compass.Data.Z;
      // DEBUG_SERIAL.printf("Compass: %f %f %f\n", mag_x, mag_y, mag_z);
      imu_recv_data.mag_x = mag_x;
      imu_recv_data.mag_y = mag_y;
      imu_recv_data.mag_z = mag_z;
    }
  }
}


/**
 * ====================================
 *          PID Controller
 * ====================================
 */
#include <QuickPID.h>


#include "LowPassFilter.h"
LowPassFilter force_lpf_1(0.8);
LowPassFilter force_lpf_2(0.5);
LowPassFilter force_lpf_3(0.5);
 

/**
 * ====================================
 *              Timers
 * ====================================
 */
esp_timer_handle_t sensor_timer;
esp_timer_handle_t motor_control_timer;
 
/**
 * ====================================
 *              GPIO Expander
 * ====================================
 */

#include "TCA9555.h"
TCA9555 TCA(0x25);
#define LED_SUCCEED     0     //
#define CPIOP_12        3     //
#define CPIOP_13        4     //
#define LED_STATUS_1    5     //
#define LED_STATUS_2    6     //
#define LED_NAV_FRONT   7     //
#define LED_NAV_MIDDLE  8     // 
#define LED_NAV_REAR    9     //
#define LED_SURFACE_C1  10    //
#define LED_SURFACE_C2  11    //

#define LCELL_4_DOUT    12
#define LCELL_4_SCK     13

#define DIR_M1          15    //
#define DIR_M2          14    //


/**
 * ====================================
 *              PWM DC Control
 * ====================================
 */
#define DC_M1_PWM_PIN     4     
#define DC_M2_PWM_PIN     13     

const int pwm_frequency   = 250;    //Hz
const int pwm_channel     = 0;      //
const int pwm_resolution  = 8;      //bit

void init_dc_motor()
{
  pinMode(DC_M1_PWM_PIN, OUTPUT);
  
  ledcSetup(pwm_channel, pwm_frequency, pwm_resolution);
  ledcAttachPin(DC_M1_PWM_PIN, pwm_channel);
}

void cleaning_dc_motor_pwm(int duty_cycle)
{
  ledcWrite(pwm_channel, duty_cycle);
}


/**
 * ====================================
 *              Operations
 * ====================================
 */

void force_sensor_1_read()
{
  if (force_sensor_1.wait_ready_timeout(5)) 
  {
    clamp_force_1 = force_sensor_1.get_units(1);  
  } 
  else 
  {
    // DEBUG_SERIAL.println("force_sensor_1 not found.");
  }
}

void force_sensor_2_read()
{
  if (force_sensor_2.wait_ready_timeout(5)) 
  {
    clamp_force_2 = force_sensor_2.get_units(1);
  } 
  else 
  {
    // DEBUG_SERIAL.println("force_sensor_2 not found.");
  }
}

void force_sensor_3_read()
{
  if (force_sensor_3.wait_ready_timeout(5)) 
  {
    clamp_force_3 = force_sensor_3.get_units(1);
  } 
  else 
  {
    // DEBUG_SERIAL.println("force_sensor_3 not found.");
  }
}

void force_sensor_read()
{
  // clamp_force_1 = force_sensor_1.get_units(1);// - -18000.0 ) * (750.0/46820.0) * CONV_GRAM2NEWTON;  
  // clamp_force_2 = force_sensor_2.get_units(1);// - 415740.0 ) * (535.0/17850.0) * CONV_GRAM2NEWTON;
  // clamp_force_3 = (force_sensor_3.get_units(1)- -11666.0 )* (1500.0/205000.0) * CONV_GRAM2NEWTON; 

  clamp_force_1 = (force_sensor_1.get_units(1) - 422500.0) * (1500.0/160500.0) * CONV_GRAM2NEWTON;    // LC-1
  clamp_force_2 = (force_sensor_2.get_units(1) - 1246000.0) * (1500.0/212360.0) * CONV_GRAM2NEWTON;   // LC-2
  clamp_force_3 = (force_sensor_3.get_units(1) - -14300.0) * (1500.0/176000.0) * CONV_GRAM2NEWTON;  // LC-3

  // Low pass filtering
  clamp_force_1 = force_lpf_1.update(clamp_force_1);
  clamp_force_2 = force_lpf_2.update(clamp_force_2);
  clamp_force_3 = force_lpf_3.update(clamp_force_3) ;

}

void encoder_read()
{
    // MEASURE_TIMER_FREQ(sensor_feedback);
    /* ======================= Encoder reading ======================= */
    clamp_angle = encoder.readAngle() * 0.0174532925 - clamp_angle_offset ;    // radian
    // DEBUG_SERIAL.println(clamp_angle);
}

void motor_read_1() 
{
    // MEASURE_TIMER_FREQ(motor_read);
    /* ======================= Motor reading ======================= */
    clamp_motor.update_motor();

    // DEBUG_SERIAL.println(clamp_motor.motor_feedback.id);
    // DEBUG_SERIAL.println(clamp_motor.motor_feedback.connection_status);
    // DEBUG_SERIAL.println(clamp_motor.motor_feedback.position);
    // DEBUG_SERIAL.println(clamp_motor.motor_feedback.velocity);
    // DEBUG_SERIAL.println(clamp_motor.motor_feedback.torque);
    // DEBUG_SERIAL.println(clamp_motor.motor_feedback.current);
    // DEBUG_SERIAL.println(clamp_motor.motor_feedback.temperature);
}

void motor_read_2() 
{
    // MEASURE_TIMER_FREQ(motor_read);
    /* ======================= Motor reading ======================= */
    left_wheel_motor.update_motor();

    // DEBUG_SERIAL.println(left_wheel_motor.motor_feedback.id);
    // DEBUG_SERIAL.println(left_wheel_motor.motor_feedback.connection_status);
    // DEBUG_SERIAL.println(left_wheel_motor.motor_feedback.position);
    // DEBUG_SERIAL.println(left_wheel_motor.motor_feedback.velocity);
    // DEBUG_SERIAL.println(left_wheel_motor.motor_feedback.torque);
    // DEBUG_SERIAL.println(left_wheel_motor.motor_feedback.current);
    // DEBUG_SERIAL.println(left_wheel_motor.motor_feedback.temperature);
}

void motor_read_3() 
{
    // MEASURE_TIMER_FREQ(motor_read);
    /* ======================= Motor reading ======================= */
    right_wheel_motor.update_motor();

    // DEBUG_SERIAL.println(right_wheel_motor.motor_feedback.id);
    // DEBUG_SERIAL.println(right_wheel_motor.motor_feedback.connection_status);
    // DEBUG_SERIAL.println(right_wheel_motor.motor_feedback.position);
    // DEBUG_SERIAL.println(right_wheel_motor.motor_feedback.velocity);
    // DEBUG_SERIAL.println(right_wheel_motor.motor_feedback.torque);
    // DEBUG_SERIAL.println(right_wheel_motor.motor_feedback.current);
    // DEBUG_SERIAL.println(right_wheel_motor.motor_feedback.temperature);
}
#endif // CONFIG_H
