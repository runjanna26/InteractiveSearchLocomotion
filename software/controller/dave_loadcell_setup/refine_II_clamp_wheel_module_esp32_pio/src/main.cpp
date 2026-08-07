/**
 * Task list:
 * [ ] Read IMU and publish
 * [ ]
  * [/] Connection check
 * [/] Optimal Publisher
 * [/] Local CAN bus Library 
 * [/] Cannot read position of AK motor
 * [/] Clamp position control with current limitation

 */
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
    ESP32Can.CANInit(TX_CAN_PIN, RX_CAN_PIN, ESP32CAN_SPEED_1MBPS );

    DEBUG_SERIAL.begin(DEBUG_SERIAL_BAUD);
    DEBUG_SERIAL.println("Start setup module...");

    /* ======== Initialize force sensor ======== */
    // LowPassFilter(0.8);
    force_sensor_1.begin(HX711_1_DOUT_PIN, HX711_1_SCK_PIN);
    // force_sensor_1.set_offset(HX711_1_OFFSET);  // offset raw data
    // force_sensor_1.set_scale(HX711_1_SCALE);    // convert raw to gram (any unit)

    force_sensor_2.begin(HX711_2_DOUT_PIN, HX711_2_SCK_PIN);
    // // force_sensor_2.set_offset(HX711_2_OFFSET);  // offset raw data
    // // force_sensor_2.set_scale(HX711_2_SCALE);    // convert raw to gram (any unit)

    force_sensor_3.begin(HX711_3_DOUT_PIN, HX711_3_SCK_PIN);
    // force_sensor_3.set_offset(HX711_3_OFFSET);  // offset raw data
    // force_sensor_3.set_scale(HX711_3_SCALE);    // convert raw to gram (any unit)
    DEBUG_SERIAL.println("Setup force sensors");

    /* ======== Initialize PWM for DC motors======== */  
    init_dc_motor();
    cleaning_dc_motor_pwm(0);  

    /* ======== Initialize IMU ======== */  
    WIRE_PORT.begin();
    WIRE_PORT.setClock(400000);
    initializeIMU();
    DEBUG_SERIAL.println("Setup IMU");


    /* ======== Micro-ROS setup ======== */
    set_microros_wifi_transports((char*)SSID_WIFI, (char*)PASS_WIFI, (char*)AGENT_IP, AGENT_PORT);
    DEBUG_SERIAL.println("micro-ROS connected");
    
    if (create_entities()) DEBUG_SERIAL.println("create_entities() succeded");
    else DEBUG_SERIAL.println("create_entities() failed");  // Should restart the esp to reconnect to uros agent

    if (setup_multiarray_publisher_msg()) DEBUG_SERIAL.println("setup_multiarray_publisher_msg() succeded");
    else DEBUG_SERIAL.println("setup_multiarray_publisher_msg() failed");
    
    /* ======== Initialize GPIO Expander ======== */ 
    TCA.begin();

    // for (int pin = 0; pin < 18; pin++)
    // {
    //     TCA.pinMode1(pin, OUTPUT);  
    //     TCA.write1(pin, HIGH);  
    // }

    TCA.pinMode1(LED_SUCCEED, OUTPUT);
    TCA.pinMode1(CPIOP_12, OUTPUT);
    TCA.pinMode1(CPIOP_13, OUTPUT);      
    TCA.pinMode1(LED_STATUS_1, OUTPUT);  
    TCA.pinMode1(LED_STATUS_2, OUTPUT);  

    // LED Lighting
    TCA.pinMode1(LED_NAV_FRONT, OUTPUT); 
    TCA.pinMode1(LED_NAV_MIDDLE, OUTPUT);   // Cut it off
    TCA.pinMode1(LED_NAV_REAR, OUTPUT);  
    TCA.pinMode1(LED_SURFACE_C1, OUTPUT);   // Cut it off
    TCA.pinMode1(LED_SURFACE_C2, OUTPUT);   // Cut it off

    TCA.pinMode1(DIR_M1, OUTPUT);       // cannot set direction of dc motor
    TCA.pinMode1(DIR_M2, OUTPUT);       // cannot set direction of dc motor

    TCA.pinMode1(LCELL_4_DOUT, OUTPUT); 
    TCA.pinMode1(LCELL_4_SCK, OUTPUT); 
    DEBUG_SERIAL.println("Setup GPIO Expander");
    
    TCA.write1(LED_SUCCEED, LOW);




    /* ======== Initialize Current Sensors======== */  
    if (!INA219_DC_1.begin()) 
    {
        Serial.println("Failed to find INA219 chip DC 1");
    }
    if (!INA219_DC_2.begin()) 
    {
        Serial.println("Failed to find INA219 chip DC 2");
    }

    INA219_DC_1.setCalibration_32V_2A();
    INA219_DC_2.setCalibration_32V_2A();


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

    // EXECUTE_EVERY_N_MS(10, force_sensor_1_read());
    // EXECUTE_EVERY_N_MS(10, force_sensor_2_read());
    // EXECUTE_EVERY_N_MS(10, force_sensor_3_read());

    EXECUTE_EVERY_N_MS(10, force_sensor_read());



    EXECUTE_EVERY_N_MS(10, encoder_read());

    EXECUTE_EVERY_N_MS(10, motor_read_1());
    EXECUTE_EVERY_N_MS(10, motor_read_2());
    EXECUTE_EVERY_N_MS(10, motor_read_3());

    EXECUTE_EVERY_N_MS(10, dc_motor_current_read());
    

    EXECUTE_EVERY_N_MS(1, imu_read());

    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10)); 
    if (using_connectioncheck){EXECUTE_EVERY_N_MS(1, connectionCheck()); connection_publisher_msg.data = true;}
}


// ****************************************************************************************
// ****************************************************************************************
// ---------------------                                         --------------------------
// ---------------------                     END                 --------------------------
// ---------------------                                         --------------------------
// ****************************************************************************************
// ****************************************************************************************