#pragma once
// Copy to config.h and set the local values. Never publish config.h or its binaries.
#define NAVGUIDE_WIFI_SSID ""
#define NAVGUIDE_WIFI_PASS ""
// Hostname only, without a scheme, path or port.
#define NAVGUIDE_SERVER_HOST ""
#define NAVGUIDE_SERVER_PORT 8081
#define NAVGUIDE_UDP_HOST NAVGUIDE_SERVER_HOST
#define NAVGUIDE_UDP_PORT 12345
// Must match NAVGUIDE_DEVICE_TOKEN on the service. Use a random hex token.
#define NAVGUIDE_DEVICE_TOKEN ""
// Use TLS for an HTTPS reverse proxy. Set port 443 and paste its trusted root CA.
#ifndef NAVGUIDE_TLS
#define NAVGUIDE_TLS 0
#endif
#define NAVGUIDE_ROOT_CA ""
// Keep UDP on a private LAN or VPN. Turn off if the cloud has no private route.
#define NAVGUIDE_IMU_ENABLED 1
#define NAVGUIDE_CAMERA_SIZE FRAMESIZE_VGA
#define NAVGUIDE_CAMERA_MAX_SIZE FRAMESIZE_QXGA
#define NAVGUIDE_CAMERA_FPS 15
#define NAVGUIDE_JPEG_QUALITY 17
// Physical left/right must be checked on the assembled glasses before navigation.
#define NAVGUIDE_CAMERA_HMIRROR 0
#define NAVGUIDE_CAMERA_VFLIP 0
// Override only when the actual wiring differs. D0..D3 map to GPIO1..GPIO4.
#define NAVGUIDE_IMU_SCK 1
#define NAVGUIDE_IMU_MOSI 2
#define NAVGUIDE_IMU_MISO 3
#define NAVGUIDE_IMU_CS 4
#define NAVGUIDE_SPEAKER_BCLK 7
#define NAVGUIDE_SPEAKER_LRCK 8
#define NAVGUIDE_SPEAKER_DIN 9
#define NAVGUIDE_SPEAKER_GAIN 0.7f
