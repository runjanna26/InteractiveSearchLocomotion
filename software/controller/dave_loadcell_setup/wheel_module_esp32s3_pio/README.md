# HERO2.1 Wheel Module 
##  Board components
- **Microcontroller**: *ESP32S3*
- **Wire communication**: *CAN* 
- **Actuators**: *RMD-X4-36*
- **IMU**: *ICM-20948*

# Topics and Messages (100 Hz)
## Publish Feedback
### 1. `swerve_feedback`
- **Type:** `float32[]`
- **Description:** Direction and velocity feedback from each swerve module and imu feedback.
- **Format:** `[vel_1, dir_1, vel_2, dir_2, vel_3, dir_3, vel_4, dir_4, roll, pitch, yaw, acc_x, acc_y, acc_z]`

---
### 2. `motor_torque_feedback`
- **Type:** `float32[]`
- **Description:** Estimated torque output of each motor.
- **Format:** `[tor_1, tor_2, tor_3, tor_4, tor_5, tor_6, tor_7, tor_8]` 
---
### 3. `motor_current_feedback`
- **Type:** `float32[]`
- **Description:** Measured motor phase current for diagnostics and protection.
- **Format:** `[cur_1, cur_2, cur_3, cur_4, cur_5, cur_6, cur_7, cur_8]`
---
### 4. `controller_connection`
- **Type:** `bool`
- **Description:** Controller connection status.
- **Values:**
- `true` → Connected
- `false` → Disconnected
---
### 5. `motor_connection`
- **Type:** `bool[]`
- **Description:** Connection status of individual motors.
- **Format:** `[con_1, con_2, con_3, con_4, con_5, con_6, con_7, con_8]`
---
### 6. `motor_temperature_feedback`
- **Type:** `uint8[]`
- **Description:** Temperature feedback from each motor (°C).
- **Format:** `[motor_temp_1, motor_temp_2, motor_temp_3, motor_temp_4, motor_temp_5, motor_temp_6, motor_temp_7, motor_temp_8]`
---
### 7. `board_temperature_feedback`
- **Type:** `uint8`
- **Description:** Main control board temperature (°C).
- **Note:** `255` indicates invalid or unavailable data.
---
### 8. `ambient_temperature_feedback`
- **Type:** `uint8`
- **Description:** Ambient temperature near the system (°C).
- **Note:** `255` indicates invalid or unavailable data.
---
### 9. `ambient_gas_feedback`
- **Type:** `uint8[]`
- **Description:** Ambient gas sensor readings.
- **Format:** `[gas_1, gas_2, gas_3]`
---
### 10. `motor_error`
- **Type:** `uint8[]`
- **Description:** Motor error codes for fault detection and recovery.
- **Format:** `[motor_err_1, motor_err_2, motor_err_3, motor_err_4, motor_err_5, motor_err_6, motor_err_7, motor_err_8]`

## Subscribe Command
### 1. `command_recv_msg`
- **Type:** `float32[]`
- **Description:** Motion command input for the swerve modules.
- **Format:** `[dir_1, vel_1, dir_2, vel_2, dir_3, vel_3, dir_4, vel_4]`
---
### 2. `motor_reset_command`
- **Type:** `uint8[]`
- **Description:** Motor reset commands.
- **Values:**
- `1` → Reset motor
- `0` → No action
- **Format:** `[motor_rst_1, motor_rst_2, motor_rst_3, motor_rst_4, motor_rst_5, motor_rst_6, motor_rst_7, motor_rst_8]`

## Bandwidth Usage

| Direction | Bandwidth |
|---------|-----------|
| Publish | **108 Kbps** |
| Subscribe | **0.14 Kbps** |


## Recovery level (For safety)
| Timer	                | Timeout (ms) | Condition| Action |
|---------------------|---------------|--|----------------|
|watchdog_cmd	            |200    |No valid command message received | Immediately stop all motors |
|watchdog_uros	|1000  |No confirmed communication with micro-ROS agent| Ping agent, if unreachable → reconnect micro-ROS|
|watchdog_wifi	|3000  |ESP32 not connected to Wi-Fi| Reconnect Wi-Fi|
|watchdog_reset	|5000  |No successful recovery at any level| ESP32 restart|

