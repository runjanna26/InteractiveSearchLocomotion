#include "ESP32CAN.h"

ESP32CAN_status_t ESP32CAN::CANInit(gpio_num_t tx_pin, gpio_num_t rx_pin, ESP32CAN_timing_t baud) {
    /* initialize configuration structures */
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(tx_pin, rx_pin, TWAI_MODE_NORMAL);
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

    twai_timing_config_t t_config;
    
    g_config.rx_queue_len = 32; 
    g_config.tx_queue_len = 32;

    switch (baud) {
        case ESP32CAN_SPEED_100KBPS:
            t_config = TWAI_TIMING_CONFIG_100KBITS();
            break;
        case ESP32CAN_SPEED_125KBPS:
            t_config = TWAI_TIMING_CONFIG_125KBITS();
            break;
        case ESP32CAN_SPEED_250KBPS:
            t_config = TWAI_TIMING_CONFIG_250KBITS();
            break;
        case ESP32CAN_SPEED_500KBPS:
            t_config = TWAI_TIMING_CONFIG_500KBITS();
            break;
        case ESP32CAN_SPEED_800KBPS:
            t_config = TWAI_TIMING_CONFIG_800KBITS();
            break;
        case ESP32CAN_SPEED_1MBPS:
            t_config = TWAI_TIMING_CONFIG_1MBITS();
            break;
        default:
            CAN_DEBUG_println("TWAI: undefined buad rate");
            return ESP32CAN_NOK;
            break;
    }

    /* install TWAI driver */
    switch (twai_driver_install(&g_config, &t_config, &f_config)) {
        case ESP_OK:
            CAN_DEBUG_println("TWAI INSTALL: ok");
            break;
        case ESP_ERR_INVALID_ARG:
            CAN_DEBUG_println("TWAI INSTALL: ESP_ERR_INVALID_ARG");
            return ESP32CAN_NOK;
            break;
        case ESP_ERR_NO_MEM:
            CAN_DEBUG_println("TWAI INSTALL: ESP_ERR_NO_MEM");
            return ESP32CAN_NOK;
            break;
        case ESP_ERR_INVALID_STATE:
            CAN_DEBUG_println("TWAI INSTALL: ESP_ERR_INVALID_STATE");
            return ESP32CAN_NOK;
            break;
        default:
            CAN_DEBUG_println("TWAI INSTALL: uknown error");
            return ESP32CAN_NOK;
            break;
    }

    /* start TWAI driver */
    switch (twai_start()) {
        case ESP_OK:
            CAN_DEBUG_println("TWAI START: ok");
            break;
        case ESP_ERR_INVALID_STATE:
            CAN_DEBUG_println("TWAI START: ESP_ERR_INVALID_STATE");
            return ESP32CAN_NOK;
            break;
        default:
            CAN_DEBUG_println("TWAI START: uknown error");
            return ESP32CAN_NOK;
            break;
    }

    return ESP32CAN_OK;
}

ESP32CAN_status_t ESP32CAN::CANStop() {
    /* stop the TWAI driver */
    switch (twai_stop()) {
        case ESP_OK:
            CAN_DEBUG_println("TWAI STOP: ok");
            break;
        case ESP_ERR_INVALID_STATE:
            CAN_DEBUG_println("TWAI STOP: ESP_ERR_INVALID_STATE");
            return ESP32CAN_NOK;
            break;
        default:
            CAN_DEBUG_println("TWAI STOP: unknow error");
            return ESP32CAN_NOK;
            break;
    }

    /* uninstall TWAI driver */
    switch (twai_driver_uninstall()) {
        case ESP_OK:
            CAN_DEBUG_println("TWAI UNINSTALL: ok");
            break;
        case ESP_ERR_INVALID_STATE:
            CAN_DEBUG_println("TWAI UNINSTALL: ESP_ERR_INVALID_STATE");
            return ESP32CAN_NOK;
            break;
        default:
            break;
    }

    return ESP32CAN_OK;
}

ESP32CAN_status_t ESP32CAN::CANWriteFrame(const twai_message_t* p_frame) {
    /* queue message for transmission */
    switch (twai_transmit(p_frame, pdMS_TO_TICKS(1))) {
        case ESP_OK:
            break;
        case ESP_ERR_INVALID_ARG:
            // CAN_DEBUG_println("TWAI TX: ESP_ERR_INVALID_ARG");
            return ESP32CAN_NOK;
            break;
        case ESP_ERR_TIMEOUT:
            // CAN_DEBUG_println("TWAI TX: ESP_ERR_TIMEOUT");
            return ESP32CAN_NOK;
            break;
        case ESP_FAIL:
            // CAN_DEBUG_println("TWAI TX: ESP_FAIL");
            return ESP32CAN_NOK;
            break;
        case ESP_ERR_INVALID_STATE:
            // CAN_DEBUG_println("TWAI TX: ESP_ERR_INVALID_STATE");
            return ESP32CAN_NOK;
            break;
        case ESP_ERR_NOT_SUPPORTED:
            // CAN_DEBUG_println("TWAI TX: ESP_ERR_NOT_SUPPORTED");
            return ESP32CAN_NOK;
            break;
        default:
            // CAN_DEBUG_println("TWAI TX: unknow error");
            return ESP32CAN_NOK;
            break;
    }

    return ESP32CAN_OK;
}

ESP32CAN_status_t ESP32CAN::CANReadFrame(twai_message_t* p_frame) {
    switch (twai_receive(p_frame, pdMS_TO_TICKS(1))) {
    case ESP_OK:
        break;
    case ESP_ERR_TIMEOUT:
        // CAN_DEBUG_println("TWAI RX: ESP_ERR_TIMEOUT");
        return ESP32CAN_NOK;
        break;
    case ESP_ERR_INVALID_ARG:
        // CAN_DEBUG_println("TWAI RX: ESP_ERR_INVALID_ARG");
        return ESP32CAN_NOK;
        break;
    case ESP_ERR_INVALID_STATE:
        // CAN_DEBUG_println("TWAI RX: ESP_ERR_INVALID_STATE");
        return ESP32CAN_NOK;
        break;
    default:
        // CAN_DEBUG_println("TWAI RX: unknow error");
        return ESP32CAN_NOK;
        break;
    }

    return ESP32CAN_OK;
}

// int ESP32CAN::CANConfigFilter(const CAN_filter_t* p_filter)
// {
//     return CAN_config_filter(p_filter);
// }

ESP32CAN ESP32Can;
