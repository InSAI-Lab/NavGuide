# NavGuide 硬件与固件

固件面向 Seeed Studio XIAO ESP32S3 Sense，使用摄像头、板载 PDM 麦克风、外接 ICM42688-P 和 MAX98357A 采集传感器数据并传输语音。语义选择、惯性门控和导航推理在服务端运行。

## 接线

| 模块 | 模块引脚 | ESP32S3 GPIO | XIAO 标号 |
| --- | --- | --- | --- |
| ICM42688-P | SCK | 1 | D0 |
| ICM42688-P | SDI / MOSI | 2 | D1 |
| ICM42688-P | SDO / MISO | 3 | D2 |
| ICM42688-P | CS | 4 | D3 |
| MAX98357A | BCLK | 7 | D8 |
| MAX98357A | LRC / LRCK | 8 | D9 |
| MAX98357A | DIN | 9 | D10 |
| 板载麦克风 | CLK | 42 | 内部连接 |
| 板载麦克风 | DATA | 41 | 内部连接 |

IMU 使用 3.3 V 逻辑与供电并共地。MAX98357A 的电源按所用模块说明和扬声器功率选择，所有模块共地，不向 ESP32S3 GPIO 输入 5 V。GPIO3 是启动配置引脚，外接模块不能在复位时强行改变启动电平。GPIO7、8、9 同时属于 Sense 的 SD 接口，本接线使用扬声器时不插入或启用 SD 卡。

摄像头通过 Sense 扩展板连接，完整引脚定义位于 `firmware/camera_pins.h`。不同批次摄像头能力不同，OV2640 的最大尺寸是 UXGA，OV3660 可支持 QXGA。驱动初始化会按传感器能力限制最高尺寸。接口命令不能超过实际分配的帧缓冲上限。引脚依据 [Seeed 官方板卡资料](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/) 和 [麦克风资料](https://wiki.seeedstudio.com/xiao_esp32s3_sense_mic/)。

## 配置

在项目根目录执行：

```bash
cp firmware/config.example.h firmware/config.h
```

编辑 `config.h` 中 WiFi 名称、密码、服务端主机名、端口和 `NAVGUIDE_DEVICE_TOKEN`。服务端使用同名环境变量。Token 可以是最多 128 个字符的字母、数字、下划线、点或连字符，推荐随机 64 位十六进制字符串。固件按上述字符范围校验令牌，配置模板中的网络信息留空。

`config.h`、`.pio/`、`.venv-firmware/` 和 `build/` 已列入 `.gitignore`。使用私有配置构建的固件包含 WiFi 密码与设备令牌，应与配置文件一同保存在本地。

局域网开发使用 `NAVGUIDE_TLS=0`，与后端 HTTP 端口一致。通过云端 HTTPS 反向代理访问时设置：

```cpp
#define NAVGUIDE_SERVER_HOST "navguide.example.org"
#define NAVGUIDE_SERVER_PORT 443
#define NAVGUIDE_TLS 1
#define NAVGUIDE_ROOT_CA R"PEM(
-----BEGIN CERTIFICATE-----
此处粘贴服务端证书对应的可信根 CA
-----END CERTIFICATE-----
)PEM"
```

TLS 用于摄像头 WS、麦克风 WS 和语音 HTTP。配置中填写完整的可信根 CA，固件在 NTP 校时后按服务域名验证证书，校验为必选项。所有 WS 和 HTTP 请求使用 `Authorization: Bearer <token>`。

IMU 使用无 TLS 的 UDP 通道，`NAVGUIDE_UDP_HOST` 需通过同一 WiFi 私网或 VPN 可达，防火墙仅开放私网访问。设置 `NAVGUIDE_IMU_ENABLED=0` 会停用惯性数据上传。本地服务未从其他接口收到有效惯性数据时，会暂缓普通提示；紧急和目标提示仍按门控规则处理。

## 构建与烧录

固定依赖为 PlatformIO Core 6.1.18、pioarduino 平台 55.03.37、Arduino ESP32 3.3.7、ArduinoWebsockets 0.5.4。平台的发布说明标注基于 ESP-IDF 5.5.2。固件依赖 Arduino 3.x 的 `ESP_I2S.h`，Arduino 2.x 平台不提供该接口。[固定平台发布](https://github.com/pioarduino/platform-espressif32/releases/tag/55.03.37)，[Espressif I2S API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/i2s.html)。

需要 Python 3.10 至 3.14，建议 Python 3.12。首次安装会下载交叉编译工具链：

```bash
python3.12 -m venv .venv-firmware
.venv-firmware/bin/pip install platformio==6.1.18
scripts/test_firmware.sh
scripts/build_firmware.sh
```

成功后 `build/release/xiao_esp32s3/` 包含应用、启动器、分区表、ELF 调试文件和 SHA256 清单。它们是私有配置构建产物。源码仓库公开 CI 使用无凭据环境：

```bash
FIRMWARE_ENV=firmware_ci scripts/build_firmware.sh
FIRMWARE_ENV=firmware_ci_tls scripts/build_firmware.sh
```

`firmware_ci` 固定加载 `config.example.h`，用于验证源码和工具链。空配置固件启动时报告配置缺失并保持离线；联网运行使用填写配置后的 `xiao_esp32s3` 构建。

连接开发板后确认端口，再执行烧录：

```bash
.venv-firmware/bin/pio device list
.venv-firmware/bin/pio run -e xiao_esp32s3 -t upload --upload-port /dev/cu.usbmodemXXXX
.venv-firmware/bin/pio device monitor --port /dev/cu.usbmodemXXXX --baud 115200
```

Windows 端口一般是 `COM3` 等，Linux 一般是 `/dev/ttyACM0`。识别不到时按住 BOOT，按下并松开 RESET，然后松开 BOOT，再次选择端口。PlatformIO 按分区表将应用、启动器和分区数据写入对应偏移；应用 `firmware.bin` 的偏移与启动器地址不同。固件通过 USB 更新。

## 设备协议

| 通道 | 格式 | 方向 |
| --- | --- | --- |
| `/ws/camera` | 一条完整 WS 二进制消息为一张 JPEG；内部使用 4096 字节分片控制临时内存 | 设备到服务 |
| `/ws_audio` | 先发送文本 `START`，再发送每块 640 字节的 PCM s16le、16 kHz、单声道 | 设备到服务 |
| `/ws_audio` | 文本 `RESTART` 触发清空麦克风队列并重新发送 `START` | 服务到设备 |
| `/stream.wav` | PCM WAV，16 位单声道，8 至 48 kHz；支持 HTTP chunked、未知长度流和 WAV 附加块 | 服务到设备 |
| UDP 12345 | UTF-8 JSON，传感器 50 Hz；网络拥塞或丢包会降低接收频率 | 设备到服务 |

UDP 数据示例：

```json
{"schema_version":1,"seq":10,"ts":200,"temp_c":25.0,"accel":{"x":0,"y":0,"z":9.80665},"gyro":{"x":0,"y":0,"z":30},"yaw_rate_dps":30,"token":""}
```

`ts` 是设备启动以来的单调毫秒时间，使用 64 位计时，重启后归零；`seq` 为当前启动周期内的包序号。`accel` 为 m/s²，`gyro` 与 `yaw_rate_dps` 均为 deg/s。`yaw_rate_dps` 由减去缓慢估计零偏后的陀螺角速度投影至低通重力方向得到，直接供服务端的度每秒门控阈值使用。它不是绝对航向，不含磁力计校正。加速度含重力，剧烈平移时重力估计会产生暂态误差。

ICM42688-P 驱动显式配置 ±16 g、±2000 deg/s、50 Hz，检查 `WHO_AM_I=0x47`、寄存器回读和数据就绪标志。温度使用原始值 / 132.48 + 25，寄存器及转换依据 [TDK 官方数据手册](https://invensense.tdk.com/wp-content/uploads/2022/12/DS-000347-ICM-42688-P-v1.7.pdf)。

摄像头支持 `SET:FRAMESIZE=VGA`、`SET:QUALITY=17`、`SET:FPS=15` 和 `SNAP:HQ`。尺寸选项为 QVGA、VGA、SVGA、XGA、HD、SXGA、UXGA、FHD、QXGA，实际是否可用由传感器与缓冲上限决定；质量范围 5 至 40，数值越小画质越高。FPS 范围 0 至 60，0 表示不限速。`SNAP:HQ` 返回文本 `SNAP:BEGIN`、一条 JPEG 消息和 `SNAP:END`。

## 调度与设备检查

摄像头及其 WebSocket 由同一任务独占，避免取帧、快照、参数修改和发送跨任务竞态。麦克风采集与上传使用 6 块有界队列，积压时丢弃旧块，超过 120 ms 的音频不再上传。扬声器独占 I2S TX。网络写入超时关闭连接，WiFi 和三个流连接独立重试，摄像头的网络消息按小片发送以限制内存峰值。

默认分辨率为 VGA，发送帧率上限为 15 FPS。串口每 10 秒输出实际发送帧率，每 30 秒输出剩余堆、最低剩余堆、PSRAM 和麦克风丢块数。实际发送速率由摄像头、WiFi 和服务端吞吐共同决定，推理与语音输出另有各自的处理频率。

烧录后的设备检查包括：

1. 静置设备，IMU 加速度模长约为 9.8 m/s²，陀螺速度接近 0；旋转设备确认服务端 `yaw_rate_dps` 符号与变化合理。
2. 展示左右标记，确认视频与现实方向一致，再选择 `NAVGUIDE_CAMERA_HMIRROR` 和 `NAVGUIDE_CAMERA_VFLIP`。示例配置默认不镜像。
3. 发出语音并监听扬声器，确认无字节错位、变速和削波。扬声器增益可在配置中降低。
4. 断开 WiFi、重启后端、恢复网络，确认摄像头、麦克风、WAV 与 IMU 都恢复工作。
5. 持续运行并触发快照、尺寸切换，记录实际 FPS、延迟、堆最低余量和设备温度，确认没有持续内存下降。
6. 在有陪护的受控环境检查完整导航流程，并记录相机、传感器和音频参数。

主机测试使用 AddressSanitizer 和 UndefinedBehaviorSanitizer，覆盖传感器单位换算、角速度轴投影、PCM、WAV 解析及 HTTP 分块边界；交叉编译检查语法、链接与固件尺寸。帧率、功耗和运行稳定性在目标硬件上测量。实验配置与统计要求见 [测试与复现条件](architecture.md#测试与复现条件)。
