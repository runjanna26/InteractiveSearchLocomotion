/**
 * =========================================[Tasks]=========================================
 * [x] IMU 
 * [x]reading and publishing Motor command and feedback publishing 8 motors
 * [x] Connection check publishing
 * [x] Test the re-connection 
 * [ ] Set steering speed > driving speed by GUI
 * [ ] Motor lost connection made the publish rate drop?
 */

#define USED_UROS 
// #define USED_CONNECTION_CHECK 

// #define USED_ENCODER 
// #define USED_RMD_MOTOR
#define USED_FORCE_SENSOR


#include "config.h"
#include "micro_ros_setup.h"


// ****************************************************************************************
// ****************************************************************************************
// ------------------                                                     -----------------
// ------------------                     Void Setup                      -----------------
// ------------------                                                     -----------------
// ****************************************************************************************
// ****************************************************************************************

void setup()
{
    DEBUG_SERIAL.begin(DEBUG_SERIAL_BAUD);
    DEBUG_SERIAL.println("===========================================[Start Program]============================================");

    DEBUG_SERIAL.println("Start setup module...");

    #ifdef USED_RMD_MOTOR
      ESP32Can.CANInit(TX_CAN_PIN, RX_CAN_PIN, ESP32CAN_SPEED_1MBPS);
      
      driven_motor_1.motor_reboot();
      direct_motor_1.motor_reboot();
      driven_motor_2.motor_reboot();
      direct_motor_2.motor_reboot();
      driven_motor_3.motor_reboot();
      direct_motor_3.motor_reboot();
      driven_motor_4.motor_reboot();
      direct_motor_4.motor_reboot();

      DEBUG_SERIAL.println("Setup Motor");

  #endif
  /* ======== Initialize force sensor ======== */
  #ifdef USED_FORCE_SENSOR
    // LowPassFilter(0.8);
    force_sensor_1.begin(HX711_1_DOUT_PIN, HX711_1_SCK_PIN);
    // force_sensor_1.power_up();
    force_sensor_1.set_offset(HX711_1_OFFSET);  // offset raw data
    // force_sensor_1.set_scale(HX711_1_SCALE);    // convert raw to gram (any unit)

    force_sensor_2.begin(HX711_2_DOUT_PIN, HX711_2_SCK_PIN);
    force_sensor_2.set_offset(HX711_2_OFFSET);  // offset raw data
    // force_sensor_2.set_scale(HX711_2_SCALE);    // convert raw to gram (any unit)
  #endif

  // /* ======== Initialize IMU ======== */  
  #ifdef USED_IMU
    WIRE_PORT.begin();
    WIRE_PORT.setClock(400000);
    initializeIMU();
    DEBUG_SERIAL.println("Setup IMU");
  #endif

    #ifdef USED_NEW_IMU
  //     WIRE_PORT.begin(SDA_PIN, SCL_PIN);
  // WIRE_PORT.setClock(100000);
      // WIRE_PORT.begin();
      // WIRE_PORT.setClock(400000);
      initializeNew_IMU();
      DEBUG_SERIAL.println("Setup New IMU");
    #endif


    #ifdef USE_LED
    init_led();
    DEBUG_SERIAL.println("Setup LED");
    #endif

    /* ======== Micro-ROS setup ======== */
    #ifdef USED_UROS

        set_microros_wifi_transports((char*)SSID_WIFI, 
                                     (char*)PASS_WIFI, 
                                     (char*)MICRO_ROS_AGENT_IP, 
                                     MICRO_ROS_AGENT_PORT);
        
        DEBUG_SERIAL.println("micro-ROS connected");

        if (!setup_multiarray_publisher_msg()) DEBUG_SERIAL.println("setup_multiarray_publisher_msg() succeded");
        else DEBUG_SERIAL.println("setup_multiarray_publisher_msg() failed");

        if (!create_entities()) DEBUG_SERIAL.println("create_entities() succeded");
        else DEBUG_SERIAL.println("create_entities() failed");  // Should restart the esp to reconnect to uros agent

    #endif
}
//****************************************************************************************
//****************************************************************************************
//------------------                                                     -----------------
//------------------                     Void Loop                     -----------------
//------------------                                                     -----------------
//****************************************************************************************
//****************************************************************************************

void loop()
{

    #ifdef USED_FORCE_SENSOR
      EXECUTE_EVERY_N_MS(10, force_sensor_1_read());
      EXECUTE_EVERY_N_MS(10, force_sensor_2_read());
      get_forces(int(loadcell_force_1), int(loadcell_force_2), &loadcell_force_x, &loadcell_force_y);

      // DEBUG_SERIAL.printf("Loadcell Force X: %.2f N, Loadcell Force Y: %.2f N\n", loadcell_force_x, loadcell_force_y);
    #endif
    #ifdef USED_RMD_MOTOR
      EXECUTE_EVERY_N_MS(10, run_motor());
    #endif
    #ifdef USED_NEW_IMU
      EXECUTE_EVERY_N_MS(10, New_imu_read());
    #endif
    
    #ifdef USED_IMU
      EXECUTE_EVERY_N_MS(1, imu_read());
    #endif

    #if defined(USED_ENCODER_1) || defined(USED_ENCODER_2)
      EXECUTE_EVERY_N_MS(10, read_encoder()); 
    #endif

    #ifdef USED_UROS
      rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10)); 
    #endif

    #ifdef USED_CONNECTION_CHECK
      EXECUTE_EVERY_N_MS(10, connectionCheck()); 
      board_connection_msg.data = true;
    #endif
}


// ****************************************************************************************
// ****************************************************************************************
// ---------------------                                         --------------------------
// ---------------------                     END                 --------------------------
// ---------------------                                         --------------------------
// ****************************************************************************************
// ****************************************************************************************