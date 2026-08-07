/**
 * [E][WiFiUdp.cpp:185] endPacket(): could not send data: 12
 * 1️⃣ Publishing too fast    (by far the #1 cause)
 * 2️⃣ Message size too large (Float32MultiArray is dangerous)
 */
#define PROJECT_NAME ""
#define MODULE_NAME ""
#define MODULE_TYPE "" 
#define NODE_NAME "dave_loadcell_setup_node"


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
#include <std_msgs/msg/int16.h>
#include <std_msgs/msg/u_int8.h>
#include <std_msgs/msg/u_int16.h>
#include <std_msgs/msg/bool.h>
#include <std_msgs/msg/float32.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <std_msgs/msg/int8_multi_array.h>

#include <sensor_msgs/msg/imu.h>
#include <geometry_msgs/msg/twist.h>
#include <sensor_msgs/msg/joy.h>

#define RCCHECK(fn)              {rcl_ret_t temp_rc = fn;if ((temp_rc != RCL_RET_OK)){return !ESP_OK;}}
#define RCSOFTCHECK(fn)          {rcl_ret_t temp_rc = fn;if ((temp_rc != RCL_RET_OK)){}}

#include <EEPROM.h>

#include <string>
#include <stdlib.h>
#include <cstring>

// Micro-ROS entities
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;
rcl_timer_t slow_timer;
rclc_executor_t executor;

rmw_qos_profile_t custom_qos = rmw_qos_profile_default;


// ======= Publishers =======
// 1. CONSOLIDATED FAST TOPIC (Swerve + Torque + Current)
rcl_publisher_t state_feedback_publisher;
std_msgs__msg__Float32MultiArray state_feedback_msg;

// ======= Subscription =======
// command
rcl_subscription_t command_subscriber;
std_msgs__msg__Float32MultiArray command_recv_msg;

// LED command
rcl_subscription_t led_command_subscriber;
std_msgs__msg__UInt8 led_command_msg;

// Start signal command
rcl_subscription_t start_signal_subscriber;
std_msgs__msg__UInt8 start_signal_msg;

uint8_t val_led_5 = 0;
uint8_t val_led_6 = 0;

char *create_name(const std::string &str1, const std::string &str2)
{
  std::string combined = str1 + str2;
  char *name = new char[combined.length() + 1];
  std::strcpy(name, combined.c_str());
  return name;
}

//----------------------------------------------------------------------------------------
//------------------   Callback Function for publisher and subscriber    -----------------
//----------------------------------------------------------------------------------------

void command_subscription_callback(const void *msgin)
{
    const std_msgs__msg__Float32MultiArray *msg = (const std_msgs__msg__Float32MultiArray *)msgin;
    
    #ifdef USED_CONNECTION_CHECK
        watchdog_cmd.restart_counter();
        watchdog_uros.restart_counter();
        wathcdog_trigger_value = 0;
    #endif

    swerve_driven_command[0] = (float)msg->data.data[0];  
    swerve_driven_command[1] = (float)msg->data.data[2];  
    swerve_driven_command[2] = (float)msg->data.data[4];  
    swerve_driven_command[3] = (float)msg->data.data[6]; 

    swerve_direct_command[0] = (float)msg->data.data[1];  
    swerve_direct_command[1] = (float)msg->data.data[3];  
    swerve_direct_command[2] = (float)msg->data.data[5];  
    swerve_direct_command[3] = (float)msg->data.data[7]; 

    command_mode = (float)msg->data.data[8];
}


void publish_module_feedback()
{
    static uint64_t publish_tick = 0;
    publish_tick++;

    // =========================================================
    // FAST TOPICS: Publish at 100 Hz (Every tick)
    // =========================================================
    state_feedback_msg.data.data[0] = loadcell_force_x;
    state_feedback_msg.data.data[1] = loadcell_force_y;


    RCSOFTCHECK(rcl_publish(&state_feedback_publisher, &state_feedback_msg, NULL));
}

void publisher_callback(rcl_timer_t *timer, int64_t last_call_time)
{
  RCLC_UNUSED(last_call_time);
  
  if (timer != NULL)
  {
    publish_module_feedback();
  }
}

//----------------------------------------------------------------------------------------
//------------------     Create Entity for node,publisher,subscriber     -----------------
//----------------------------------------------------------------------------------------

esp_err_t create_entities()
{
    allocator = rcl_get_default_allocator();
    RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

    char node_name[64];
    char *name = create_name(String(NODE_NAME).c_str(), String(MODULE_NAME).c_str());
    RCCHECK(rclc_node_init_default(&node, name, "", &support));
    DEBUG_SERIAL.print("init node: ");
    DEBUG_SERIAL.println(name);
    delete[] name; 

    char topic_name[64];

    // 1. Motor State
    snprintf(topic_name, sizeof(topic_name), "%s%sloadcell", PROJECT_NAME, MODULE_NAME);
    RCCHECK(rclc_publisher_init_best_effort(&state_feedback_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray), topic_name));

    // ========== Create timer ==========
    timer = rcl_get_zero_initialized_timer();
    const unsigned int timer_timeout = 10;  
    RCCHECK(rclc_timer_init_default(&timer, &support, RCL_MS_TO_NS(timer_timeout), publisher_callback));

    // ========== Create executor ==========
    executor = rclc_executor_get_zero_initialized_executor();
    RCCHECK(rclc_executor_init(&executor, &support.context, EXECUTOR_HANDLE_NUMBER, &allocator));
    RCCHECK(rclc_executor_add_timer(&executor, &timer));

    return ESP_OK;
}
esp_err_t destroy_entities()
{
    rmw_context_t *rmw_context = rcl_context_get_rmw_context(&support.context);
    (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_context, 0);

    RCSOFTCHECK(rclc_executor_fini(&executor));

    RCSOFTCHECK(rcl_publisher_fini(&state_feedback_publisher, &node));

    RCSOFTCHECK(rcl_timer_fini(&timer));
    RCSOFTCHECK(rcl_node_fini(&node));
    RCSOFTCHECK(rclc_support_fini(&support));

    free(state_feedback_msg.data.data);
    return ESP_OK;
}


esp_err_t setup_multiarray_publisher_msg()
{
    size_t data_len;

    // ====================================================================
    data_len = 2;  
    state_feedback_msg.data.data = (float_t *)malloc(sizeof(float_t) * data_len);
    if (state_feedback_msg.data.data == NULL) return ESP_FAIL;
    state_feedback_msg.data.size = data_len;  
    state_feedback_msg.data.capacity = data_len;
    
    state_feedback_msg.layout.dim.size = 0;
    state_feedback_msg.layout.dim.capacity = 0;
    state_feedback_msg.layout.dim.data = NULL;
    state_feedback_msg.layout.data_offset = 0;

    return ESP_OK;
}

bool reconnectWiFi()
{
    if (WiFi.status() != WL_CONNECTED)
    {
        DEBUG_SERIAL.println("[WiFi] Reconnecting...");
        WiFi.disconnect(true);
        delay(200);
        WiFi.begin(SSID_WIFI, PASS_WIFI);

        unsigned long start = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - start < 3000)
        {
            delay(200);
        }

        if (WiFi.status() != WL_CONNECTED)
        {
            DEBUG_SERIAL.println("[WiFi] Failed to reconnect");
            return false;
        }
        else
        {
            DEBUG_SERIAL.println("[WiFi] Reconnected successfully");
            return true;
        }
    }
    return true;
}

bool check_uros_reconnected()
{
    if (rmw_uros_ping_agent(100, 3) != RMW_RET_OK)
        return false;

    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));

    if (rmw_uros_ping_agent(100, 3) != RMW_RET_OK)
        return false;

    return true;
}

bool reconnectMicroROS()
{
    DEBUG_SERIAL.println("[micro-ROS] Reconnecting...");

    destroy_entities();

    set_microros_wifi_transports(
        (char*)SSID_WIFI,
        (char*)PASS_WIFI,
        (char*)MICRO_ROS_AGENT_IP,
        MICRO_ROS_AGENT_PORT
    );

    if (!setup_multiarray_publisher_msg())
    {
        DEBUG_SERIAL.println("[micro-ROS] message setup failed");
        return false;
    }

    if (!create_entities())
    {
        DEBUG_SERIAL.println("[micro-ROS] create_entities failed");
        return false;
    }

    watchdog_uros.restart_counter();
    watchdog_cmd.restart_counter();

    DEBUG_SERIAL.println("[micro-ROS] Reconnect successful");
    return true;
}

void connectionCheck() // This is in the main loop
{
    if (watchdog_cmd.checktimeout())
    {
        direct_motor_1.motor_stop();
        driven_motor_1.motor_stop();
        direct_motor_2.motor_stop();
        driven_motor_2.motor_stop();
        direct_motor_3.motor_stop();
        driven_motor_3.motor_stop();
        direct_motor_4.motor_stop();
        driven_motor_4.motor_stop();
        wathcdog_trigger_value = 1;
    }

    if (watchdog_uros.checktimeout())  
    {
        if (rmw_uros_ping_agent(100, 2) == RMW_RET_OK) 
        {
            watchdog_uros.restart_counter();
        }
        else 
        {
            DEBUG_SERIAL.println("micro-ROS agent unreachable → reconnect");
                        
            if (wathcdog_trigger_value == 1)
            {
                wathcdog_trigger_value = 3;
            }
            else 
            {
                wathcdog_trigger_value = 2;
            }
            watchdog_reset.restart_counter(); 
            ESP.restart();
        }
    }

    if (WiFi.status() == WL_CONNECTED)
    {
        watchdog_wifi.restart_counter();
    }
    else if (watchdog_wifi.checktimeout())
    {
        DEBUG_SERIAL.println("WiFi connection timeout : reconnect WiFi");
        
        reconnectWiFi();
        
        if (wathcdog_trigger_value == 1)
        {
            wathcdog_trigger_value = 5;
        }
        else if (wathcdog_trigger_value == 2)
        {
            wathcdog_trigger_value = 6;
        }
        else if (wathcdog_trigger_value == 3)
        {
            wathcdog_trigger_value = 7;
        }
        else 
        {
            wathcdog_trigger_value = 4;
        }
        watchdog_wifi.restart_counter();
        
        watchdog_reset.restart_counter(); 
        ESP.restart();
    }

    // ---------------------------------------------------------
    // 4. HARDWARE RESET (5000ms Timeout)
    // ---------------------------------------------------------
    if (WiFi.status() == WL_CONNECTED && !watchdog_uros.checktimeout())
    {
        watchdog_reset.restart_counter();
    }

    if (watchdog_reset.checktimeout())
    {
        DEBUG_SERIAL.println("Reset ESP32 due to catastrophic connection failure"); 
        delay(100); 
        
        if (wathcdog_trigger_value == 1)
        {
            wathcdog_trigger_value = 8;
        }
        else if (wathcdog_trigger_value == 2)
        {
            wathcdog_trigger_value = 9;
        }
        else if (wathcdog_trigger_value == 3)
        {
            wathcdog_trigger_value = 10;
        }
        else if (wathcdog_trigger_value == 4)
        {
            wathcdog_trigger_value = 11;
        }
        else if (wathcdog_trigger_value == 5)
        {
            wathcdog_trigger_value = 12;
        }
        else if (wathcdog_trigger_value == 6)
        {
            wathcdog_trigger_value = 13;
        }
        else if (wathcdog_trigger_value == 7)
        {
            wathcdog_trigger_value = 14;
        }
        else
        {
            wathcdog_trigger_value = 16;
        }
        
         ESP.restart();
    }
}

