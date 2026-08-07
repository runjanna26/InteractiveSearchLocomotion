#include "config.h"
#include "math.h"
#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>
#include <rmw/qos_profiles.h>

#include <std_msgs/msg/int32.h>
#include <std_msgs/msg/multi_array_dimension.h>
#include <std_msgs/msg/int16_multi_array.h>
#include <std_msgs/msg/u_int8_multi_array.h>
#include <std_msgs/msg/int8.h>
#include <std_msgs/msg/u_int8.h>
#include <std_msgs/msg/bool.h>
#include <std_msgs/msg/float32.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <std_msgs/msg/int8_multi_array.h>
#include <sensor_msgs/msg/imu.h>
#include <geometry_msgs/msg/twist.h>
#include <sensor_msgs/msg/joy.h>
#define RCCHECK(fn)              {rcl_ret_t temp_rc = fn;if ((temp_rc != RCL_RET_OK)){return false;}}
#define RCSOFTCHECK(fn)          {rcl_ret_t temp_rc = fn;if ((temp_rc != RCL_RET_OK)){}}

#include <EEPROM.h>

#include <string>
#include <stdlib.h>
#include <cstring>

#define PROJECT_NAME "REFINE2"
#define MODULE_NAME "front"
#define MODULE_TYPE "clamp_wheel_module" 
#define NODE_NAME String(MODULE_TYPE) + String("_")


rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;
rcl_timer_t slow_timer;
rclc_executor_t executor;

/** 
 * Publisher Variables
 */
// Optimal version
rcl_publisher_t board_connection_publisher;
rcl_publisher_t clamp_wheel_module_publisher;
rcl_publisher_t imu_publisher;
rcl_publisher_t motor_temperature_publisher;
rcl_publisher_t motor_connection_publisher;

std_msgs__msg__Float32MultiArray clamp_wheel_module_state_msg;
std_msgs__msg__Float32MultiArray imu_msg;
std_msgs__msg__UInt8MultiArray motor_temperature_msg;
std_msgs__msg__UInt8MultiArray motor_connection_msg;

std_msgs__msg__Bool connection_publisher_msg;

// Old version
rcl_publisher_t clamp_motor_publisher;
rcl_publisher_t force_sensors_publisher;
rcl_publisher_t motor_current_publisher;
rcl_publisher_t clamp_angle_publisher;
rcl_publisher_t left_wheel_module_publisher;
rcl_publisher_t right_wheel_module_publisher;

std_msgs__msg__Float32 clamp_angle_msg;
std_msgs__msg__Float32MultiArray clamp_motor_state_msg;
std_msgs__msg__Float32MultiArray force_sensors_msg;
std_msgs__msg__Float32MultiArray motor_current_msg;
std_msgs__msg__Float32MultiArray left_wheel_motor_msg;
std_msgs__msg__Float32MultiArray right_wheel_motor_msg;

#define MOTOR_STATE_MSG_LEN 8   // number of elements in the message
#define FORCE_SENSORS_MSG_LEN 3 // number of elements in the message
#define COMMANDS_MSG_LEN 2      // number of elements in the message

/** 
 * Subscriber Variables
 */
rcl_subscription_t command_subscriber;
rcl_subscription_t led_command_subscriber;

std_msgs__msg__Float32MultiArray command_recv_msg;
std_msgs__msg__UInt8 led_command_recv_msg;




char *create_name(const std::string &str1, const std::string &str2)
{
  // Combine the two strings into a single string
  std::string combined = str1 + str2;

  // Allocate memory for the combined string
  char *name = new char[combined.length() + 1];

  // Copy the combined string into the allocated memory
  std::strcpy(name, combined.c_str());

  return name;
}

//----------------------------------------------------------------------------------------
//------------------                                                     -----------------
//------------------   Callback Function for publisher and subscriber    -----------------
//------------------                                                     -----------------
//----------------------------------------------------------------------------------------
void command_subscription_callback(const void *msgin)
{
    const std_msgs__msg__Float32MultiArray *msg = (const std_msgs__msg__Float32MultiArray *)msgin;
    if (using_connectioncheck)
    {
      watchdogtime.restart_counter();
      watchdogtime_restartESP.restart_counter();
    }

    // extract command message
    wheel_vel = (float)msg->data.data[0];
    clamp_dir = (int)(msg->data.data[1]);
    clean_vol = (float)(msg->data.data[2]); 
    
    

    // wheel command 
    wheel_vel_cmd = (float)(wheel_vel * WHEEL_VELOCITY_MAX);
    left_wheel_motor.send_motor_velocity(-wheel_vel_cmd);   //Disable AK Motor
    right_wheel_motor.send_motor_velocity(wheel_vel_cmd);   //Disable AK Motor

    // clamp command
    clamp_vel_cmd = (float)(CLAMP_VELOCITY_MAX * clamp_dir);
    clamp_motor.send_motor_velocity(clamp_vel_cmd);         //Disable AK Motor

    // clean command
    clean_cmd = (int)(clean_vol * CLEAN_VELOCITY_MAX);
    DEBUG_SERIAL.println(clean_cmd);
    cleaning_dc_motor_pwm(clean_cmd);  


}

void led_command_subscription_callback(const void *msgin)
{
    const std_msgs__msg__UInt8*msg = (const std_msgs__msg__UInt8 *)msgin;
   
    // extract command message
    led_state = (int)(msg->data);
    // DEBUG_SERIAL.println(led_state);
    // LED state command
    led_cmd_1 = (bool)((led_state >> 0) & 1);
    led_cmd_2 = (bool)((led_state >> 1) & 1);
    led_cmd_3 = (bool)((led_state >> 2) & 1);
    led_cmd_4 = (bool)((led_state >> 3) & 1);

    // DEBUG_SERIAL.println(led_cmd_1);
    // DEBUG_SERIAL.println(led_cmd_2);
    // DEBUG_SERIAL.println(led_cmd_3);
    // DEBUG_SERIAL.println(led_cmd_4);

    TCA.write1(LED_NAV_FRONT, led_cmd_1);  
    TCA.write1(LED_NAV_REAR, led_cmd_2);  
    TCA.write1(LED_STATUS_2, led_cmd_3);  
    TCA.write1(LED_SUCCEED, led_cmd_4);  
}

void publish_clamp_wheel_module_feedback()
{
    clamp_wheel_module_state_msg.data.data[0] = (float)clamp_motor.motor_feedback.position;
    clamp_wheel_module_state_msg.data.data[1] = (float)clamp_angle;
    clamp_wheel_module_state_msg.data.data[2] = (float)clamp_motor.motor_feedback.torque;
    clamp_wheel_module_state_msg.data.data[3] = (float)right_wheel_motor.motor_feedback.velocity;
    clamp_wheel_module_state_msg.data.data[4] = (float)right_wheel_motor.motor_feedback.torque;
    clamp_wheel_module_state_msg.data.data[5] = (float)left_wheel_motor.motor_feedback.velocity;
    clamp_wheel_module_state_msg.data.data[6] = (float)left_wheel_motor.motor_feedback.torque;
    RCSOFTCHECK(rcl_publish(&clamp_wheel_module_publisher, &clamp_wheel_module_state_msg, NULL));
}

void publish_motor_temperature()
{
    motor_temperature_msg.data.data[0] = (uint8_t)clamp_motor.motor_feedback.temperature;
    motor_temperature_msg.data.data[1] = (uint8_t)left_wheel_motor.motor_feedback.temperature;
    motor_temperature_msg.data.data[2] = (uint8_t)right_wheel_motor.motor_feedback.temperature;
    RCSOFTCHECK(rcl_publish(&motor_temperature_publisher, &motor_temperature_msg, NULL));
}

void publish_motor_connection_status()
{
    motor_connection_msg.data.data[0] = (uint8_t)clamp_motor.motor_feedback.connection_status;
    motor_connection_msg.data.data[1] = (uint8_t)left_wheel_motor.motor_feedback.connection_status;
    motor_connection_msg.data.data[2] = (uint8_t)right_wheel_motor.motor_feedback.connection_status;

    RCSOFTCHECK(rcl_publish(&motor_connection_publisher, &motor_connection_msg, NULL));
}

void publish_imu_feedback()
{
    imu_msg.data.data[0] = (float)(imu_recv_data.roll);
    imu_msg.data.data[1] = (float)(imu_recv_data.pitch);
    imu_msg.data.data[2] = (float)(imu_recv_data.yaw);

    imu_msg.data.data[3] = (float)(imu_recv_data.accel_x);
    imu_msg.data.data[4] = (float)(imu_recv_data.accel_y);
    imu_msg.data.data[5] = (float)(imu_recv_data.accel_z);

    // imu_msg.data.data[3] = (float)imu_recv_data.gyro_x;
    // imu_msg.data.data[4] = (float)imu_recv_data.gyro_y;
    // imu_msg.data.data[5] = (float)imu_recv_data.gyro_z;
    // imu_msg.data.data[6] = (float)imu_recv_data.mag_x;
    // imu_msg.data.data[7] = (float)imu_recv_data.mag_y;
    // imu_msg.data.data[8] = (float)imu_recv_data.mag_z;

    RCSOFTCHECK(rcl_publish(&imu_publisher, &imu_msg, NULL));
}

void publish_force_sensors_feedback()
{
    force_sensors_msg.data.data[0] = (float)clamp_force_1;
    force_sensors_msg.data.data[1] = (float)clamp_force_2;
    force_sensors_msg.data.data[2] = (float)clamp_force_3;
    // DEBUG_SERIAL.println(clamp_force_1);
    RCSOFTCHECK(rcl_publish(&force_sensors_publisher, &force_sensors_msg, NULL));
}

void publish_motor_current_feedback()
{
    motor_current_msg.data.data[0] = (float)clamp_motor.motor_feedback.current;
    motor_current_msg.data.data[1] = (float)left_wheel_motor.motor_feedback.current;
    motor_current_msg.data.data[2] = (float)right_wheel_motor.motor_feedback.current;
    motor_current_msg.data.data[3] = (float)(dc_current_2);
    
    RCSOFTCHECK(rcl_publish(&motor_current_publisher, &motor_current_msg, NULL));
}


void publisher_callback(rcl_timer_t *timer, int64_t last_call_time)
{
  RCLC_UNUSED(last_call_time);
  
  if (timer != NULL)
  {
    // MEASURE_TIMER_FREQ(pub);
    publish_clamp_wheel_module_feedback();
    publish_motor_temperature();
    publish_motor_connection_status();
    publish_force_sensors_feedback();
    publish_motor_current_feedback();

    publish_imu_feedback();
    
    RCSOFTCHECK(rcl_publish(&board_connection_publisher, &connection_publisher_msg, NULL));
  }
  return;
}


// void publish_left_wheel_motor_feedback()
// {
//     left_wheel_motor_msg.data.data[0] = (float)left_wheel_motor.motor_feedback.id;
//     left_wheel_motor_msg.data.data[1] = (float)left_wheel_motor.motor_feedback.position;
//     left_wheel_motor_msg.data.data[2] = (float)left_wheel_motor.motor_feedback.velocity;
//     left_wheel_motor_msg.data.data[3] = (float)left_wheel_motor.motor_feedback.current;
//     left_wheel_motor_msg.data.data[4] = (float)left_wheel_motor.motor_feedback.torque;
//     left_wheel_motor_msg.data.data[5] = (float)left_wheel_motor.motor_feedback.temperature;
//     left_wheel_motor_msg.data.data[6] = (float)left_wheel_motor.motor_feedback.error;
//     left_wheel_motor_msg.data.data[7] = (float)left_wheel_motor.motor_feedback.connection_status;
//     RCSOFTCHECK(rcl_publish(&left_wheel_module_publisher, &left_wheel_motor_msg, NULL));
// }

// void publish_right_wheel_motor_feedback()
// {
//     right_wheel_motor_msg.data.data[0] = (float)right_wheel_motor.motor_feedback.id;
//     right_wheel_motor_msg.data.data[1] = (float)right_wheel_motor.motor_feedback.position;
//     right_wheel_motor_msg.data.data[2] = (float)right_wheel_motor.motor_feedback.velocity;
//     right_wheel_motor_msg.data.data[3] = (float)right_wheel_motor.motor_feedback.current;
//     right_wheel_motor_msg.data.data[4] = (float)right_wheel_motor.motor_feedback.torque;
//     right_wheel_motor_msg.data.data[5] = (float)right_wheel_motor.motor_feedback.temperature;
//     right_wheel_motor_msg.data.data[6] = (float)right_wheel_motor.motor_feedback.error;
//     right_wheel_motor_msg.data.data[7] = (float)right_wheel_motor.motor_feedback.connection_status;
//     RCSOFTCHECK(rcl_publish(&right_wheel_module_publisher, &right_wheel_motor_msg, NULL));
// }

// void publish_clamp_motor_feedback()
// {
//     clamp_motor_state_msg.data.data[0] = (float)clamp_motor.motor_feedback.id;
//     clamp_motor_state_msg.data.data[1] = (float)clamp_motor.motor_feedback.position;
//     clamp_motor_state_msg.data.data[2] = (float)clamp_motor.motor_feedback.velocity;
//     clamp_motor_state_msg.data.data[3] = (float)clamp_motor.motor_feedback.current;
//     clamp_motor_state_msg.data.data[4] = (float)clamp_motor.motor_feedback.torque;
//     clamp_motor_state_msg.data.data[5] = (float)clamp_motor.motor_feedback.temperature;
//     clamp_motor_state_msg.data.data[6] = (float)clamp_motor.motor_feedback.error;
//     clamp_motor_state_msg.data.data[7] = (float)clamp_motor.motor_feedback.connection_status;
//     RCSOFTCHECK(rcl_publish(&clamp_motor_publisher, &clamp_motor_state_msg, NULL));
// }



//----------------------------------------------------------------------------------------
//------------------                                                     -----------------
//------------------     Create Entity for node,publisher,subscriber     -----------------
//------------------                                                     -----------------
//----------------------------------------------------------------------------------------
/**
 * @brief Create the node, publisher, and subscriber entities.
 * @return true if the entities are created successfully, false otherwise.
 */
bool create_entities()
{
    allocator = rcl_get_default_allocator();

    // ========== Create init_options ==========
    RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

    // ========== Create node ==========
    char *name = create_name(String(NODE_NAME).c_str(), String(MODULE_NAME).c_str());
    RCCHECK(rclc_node_init_default(&node, name, "", &support));
    DEBUG_SERIAL.print("init node: ");
    DEBUG_SERIAL.println(name);
    delete[] name; // don't forget to free the memory allocated by create_name

    // ========== Create publishers ==========
    rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
    char topic_name[64];
    
    custom_qos.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    custom_qos.depth = 5;  
    custom_qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;

    snprintf(topic_name, sizeof(topic_name), "%s/%s/%s/module_feedback", PROJECT_NAME, MODULE_TYPE, MODULE_NAME);
    RCCHECK(rclc_publisher_init(
        &clamp_wheel_module_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        topic_name,
        &custom_qos));
    DEBUG_SERIAL.print("init publisher: ");
    DEBUG_SERIAL.println(topic_name);
    
    snprintf(topic_name, sizeof(topic_name), "%s/%s/%s/imu_feedback", PROJECT_NAME, MODULE_TYPE, MODULE_NAME);
    RCCHECK(rclc_publisher_init(
        &imu_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        topic_name,
        &custom_qos));

    snprintf(topic_name, sizeof(topic_name), "%s/%s/%s/forcefeedback", PROJECT_NAME, MODULE_TYPE, MODULE_NAME);
    RCCHECK(rclc_publisher_init(
        &force_sensors_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        topic_name,
        &custom_qos));
    DEBUG_SERIAL.print("init publisher: ");
    DEBUG_SERIAL.println(topic_name);

    snprintf(topic_name, sizeof(topic_name), "%s/%s/%s/motor_current", PROJECT_NAME, MODULE_TYPE, MODULE_NAME);
    RCCHECK(rclc_publisher_init(
        &motor_current_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        topic_name,
        &custom_qos));
    DEBUG_SERIAL.print("init publisher: ");
    DEBUG_SERIAL.println(topic_name);

    snprintf(topic_name, sizeof(topic_name), "%s/%s/%s/motor_temperature", PROJECT_NAME, MODULE_TYPE, MODULE_NAME);
    RCCHECK(rclc_publisher_init(
        &motor_temperature_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8MultiArray),
        topic_name,
        &custom_qos));
    DEBUG_SERIAL.print("init publisher: ");
    DEBUG_SERIAL.println(topic_name);

    snprintf(topic_name, sizeof(topic_name), "%s/%s/%s/motor_connection", PROJECT_NAME, MODULE_TYPE, MODULE_NAME);
    RCCHECK(rclc_publisher_init(
        &motor_connection_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8MultiArray),
        topic_name,
        &custom_qos));
    DEBUG_SERIAL.print("init publisher: ");
    DEBUG_SERIAL.println(topic_name);

      snprintf(topic_name, sizeof(topic_name), "%s/%s/%s/controller_connection", PROJECT_NAME, MODULE_TYPE, MODULE_NAME);
      RCCHECK(rclc_publisher_init_best_effort(
          &board_connection_publisher,
          &node,
          ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool),
          topic_name));
    DEBUG_SERIAL.print("init publisher: ");
    DEBUG_SERIAL.println(topic_name);

    // ========== Create subscribers ==========
    snprintf(topic_name, sizeof(topic_name), "%s/%s/%s/command", PROJECT_NAME, MODULE_TYPE, MODULE_NAME);
    RCCHECK(rclc_subscription_init_best_effort(
        &command_subscriber,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        topic_name));
  
    snprintf(topic_name, sizeof(topic_name), "%s/%s/%s/led_command", PROJECT_NAME, MODULE_TYPE, MODULE_NAME);
    RCCHECK(rclc_subscription_init_best_effort(
        &led_command_subscriber,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8),
        "/REFINE2/GUI/led_command"));

    DEBUG_SERIAL.print("init publisher: ");
    DEBUG_SERIAL.println(topic_name);

    // ========== Create timer ==========
    const unsigned int timer_timeout = 50;  // best practice 20 Hz
    RCCHECK(rclc_timer_init_default(
        &timer,
        &support,
        RCL_MS_TO_NS(timer_timeout),
        publisher_callback));
    

  // ========== create executor ==========
  RCCHECK(rclc_executor_init(&executor, &support.context, EXECUTOR_HANDLE_NUMBER, &allocator));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));

  // ========== add subscription to executor ==========
  RCCHECK(rclc_executor_add_subscription(&executor, &command_subscriber, &command_recv_msg, &command_subscription_callback, ON_NEW_DATA));
  RCCHECK(rclc_executor_add_subscription(&executor, &led_command_subscriber, &led_command_recv_msg, &led_command_subscription_callback, ON_NEW_DATA));


  return true;
}

void destroy_entities()
{
  rmw_context_t *rmw_context = rcl_context_get_rmw_context(&support.context);
  (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_context, 0);

  RCSOFTCHECK(rcl_publisher_fini(&board_connection_publisher, &node));
  RCSOFTCHECK(rcl_publisher_fini(&clamp_wheel_module_publisher, &node));
  RCSOFTCHECK(rcl_publisher_fini(&imu_publisher, &node));
  RCSOFTCHECK(rcl_publisher_fini(&motor_temperature_publisher, &node));
  RCSOFTCHECK(rcl_publisher_fini(&motor_connection_publisher, &node));
  
  RCSOFTCHECK(rcl_publisher_fini(&clamp_motor_publisher, &node));
  RCSOFTCHECK(rcl_publisher_fini(&force_sensors_publisher, &node));
  RCSOFTCHECK(rcl_publisher_fini(&clamp_angle_publisher, &node));
  RCSOFTCHECK(rcl_publisher_fini(&left_wheel_module_publisher, &node));
  RCSOFTCHECK(rcl_publisher_fini(&right_wheel_module_publisher, &node));

  RCSOFTCHECK(rcl_subscription_fini(&command_subscriber, &node));

  RCSOFTCHECK(rcl_timer_fini(&timer));
  RCSOFTCHECK(rcl_node_fini(&node));
  RCSOFTCHECK(rclc_executor_fini(&executor));
  RCSOFTCHECK(rclc_support_fini(&support));
}




bool setup_multiarray_publisher_msg()
{
    size_t data_len;
    data_len = 7;
    clamp_wheel_module_state_msg.data.data = (float_t *)malloc(sizeof(float) * data_len);
    clamp_wheel_module_state_msg.data.size = data_len;
    clamp_wheel_module_state_msg.data.capacity = data_len;

    data_len = 3;
    motor_temperature_msg.data.data = (uint8_t *)malloc(sizeof(uint8_t) * data_len);
    motor_temperature_msg.data.size = data_len;
    motor_temperature_msg.data.capacity = data_len;

    data_len = 3;
    motor_connection_msg.data.data = (uint8_t *)malloc(sizeof(uint8_t) * data_len);
    motor_connection_msg.data.size = data_len;
    motor_connection_msg.data.capacity = data_len;

    data_len = 6;
    imu_msg.data.data = (float_t *)malloc(sizeof(float) * data_len);
    imu_msg.data.size = data_len;
    imu_msg.data.capacity = data_len;

    data_len = 3;
    force_sensors_msg.data.data = (float_t *)malloc(sizeof(float) * data_len);
    force_sensors_msg.data.size = data_len;
    force_sensors_msg.data.capacity = data_len;

    data_len = 4;
    motor_current_msg.data.data = (float_t *)malloc(sizeof(float) * data_len);
    motor_current_msg.data.size = data_len;
    motor_current_msg.data.capacity = data_len;

    // ====================================================================

    data_len = 3;
    command_recv_msg.data.capacity = data_len;  // set according to expected max size
    command_recv_msg.data.size = 0;
    command_recv_msg.data.data = (float *)malloc(sizeof(float) * command_recv_msg.data.capacity);
    
    return true;
}



void reconnectMicroROS() 
{
    DEBUG_SERIAL.println("[micro-ROS] Restarting WiFi and micro-ROS connection...");

    // ========== Step 1: Disconnect WiFi ==========
    // WiFi.disconnect(true);
    // delay(100);
    // DEBUG_SERIAL.println("[micro-ROS] WiFi disconnected");

    // ========== Step 2: Reconnect WiFi ==========
    // WiFi.begin(SSID_WIFI, PASS_WIFI);

    // unsigned long startAttemptTime = millis();
    // while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 10000) {
    //     delay(500);
    // }

    // if (WiFi.status() == WL_CONNECTED) 
    // {
    // ========== Step 3: Reset micro-ROS transport ==========
        set_microros_wifi_transports(
            (char*)SSID_WIFI,
            (char*)PASS_WIFI,
            (char*)AGENT_IP,
            AGENT_PORT
        );
    // }
    // else {}


    if (create_entities()) DEBUG_SERIAL.println("create_entities() succeded");
    else DEBUG_SERIAL.println("create_entities() failed");  // Should restart the esp to reconnect to uros agent

    if (setup_multiarray_publisher_msg()) DEBUG_SERIAL.println("setup_multiarray_publisher_msg() succeded");
    else DEBUG_SERIAL.println("setup_multiarray_publisher_msg() failed");
    
    if (using_connectioncheck)
    {
      watchdogtime.restart_counter();
      watchdogtime_restartESP.restart_counter();
    }
}

void connectionCheck()
{
  if (watchdogtime.checktimeout())
  {
    DEBUG_SERIAL.println("connection timeout : robot will stop ");
    // Robot stop and show some status
  }

  if (watchdogtime_restartESP.checktimeout())
  {
    DEBUG_SERIAL.println("connection timeout : ESP will be restart");
    destroy_entities();
    // ESP.restart();
    reconnectMicroROS();
  }
  else
  {
    // Show some status
  }
}

