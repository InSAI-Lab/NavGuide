# 本地部署与设备接口

## 软件环境

本地服务支持 Python 3.10 至 3.12。最小运行依赖在 `requirements.txt`，视觉依赖在 `requirements-vision.txt`。性能测量时记录实际软硬件版本。

Jetson 使用与 JetPack 匹配的 L4T 环境，先安装 NVIDIA 的对应 PyTorch/torchvision，再安装视觉依赖。部署前执行 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 核对 GPU；官方安装依据见 [NVIDIA PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)。

YOLOE 文本编码器可能需要首次安装或下载依赖，应在有网络的准备阶段完成。模型和编码器缓存就绪后，再测试断网运行。YOLOE 调用依据见 [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)。盲道和交通灯检测需加载相应任务模型。

本地语音使用 espeak-ng，Ubuntu 可运行 `sudo apt-get install espeak-ng`。语音合成输出会重采样为 8 kHz mono PCM16，合成失败在状态接口报告。启用语音后务必通过真实扬声器确认音量、方向词与延迟。

## 观察输入容器

根目录 Dockerfile 提供无 GPU 的本地服务，默认只接收检测结果。

```bash
cp .env.example .env
# 在 .env 填入 NAVGUIDE_DEVICE_TOKEN
# 生成令牌: python -c "import secrets; print(secrets.token_urlsafe(32))"
docker compose config --quiet
docker compose up -d --build
curl --fail http://127.0.0.1:8081/api/ready
```

容器默认只向主机回环地址发布 HTTP，不公开 UDP。宿主机原生运行方式适合 Jetson 与硬件联调。更改监听到局域网地址必须先设置设备令牌；当前每个进程只支持一台相机，一名用户和一组上下文，应使用单个 worker。

## 接口

受保护接口需要 `Authorization: Bearer <NAVGUIDE_DEVICE_TOKEN>`。浏览器 WebSocket 使用 `navguide` 和 `auth.<token>` 两个子协议，服务确认 `navguide`。令牌不会进入 URL。UI 只在当前页面内存中保存令牌。

| 路径 | 协议 | 内容 |
| :--- | :--- | :--- |
| `/api/health` | GET | 进程存活状态；模型就绪状态见 `/api/ready` |
| `/api/ready` | GET | 当前 frontend 就绪状态，模型失败返回 503 |
| `/api/stop` | POST | 停止引导并清除待播音频 |
| `/api/status` | GET | IMU 新鲜度、相机连接、输出计数及错误状态 |
| `/api/context` | POST | task_mode、target_query、scene、user_weights |
| `/api/observations` | POST | 最多 256 个检测，像素框和可选 yaw_rate_dps |
| `/api/imu` | POST | 与 UDP 相同的 IMU 数据，不需要 JSON token 字段 |
| `/api/describe` | POST | 用户主动请求云描述，默认不附带图像 |
| `/ws/camera` | WebSocket | 每条消息为一张 JPEG，最大 2 MiB |
| `/ws/events` | WebSocket | guidance、imu、error 和 heartbeat JSON 事件 |
| `/ws_audio` | WebSocket | START、STOP、PROMPT 命令和可选 16 kHz PCM16 ASR |
| `/stream.wav` | GET | WAV 头后为 8 kHz mono PCM16 音频流 |
| `12345/udp` | UDP | 最多 2048 字节 IMU JSON |

最小观察输入：

```json
{"yaw_rate_dps":0,"detections":[{"category":"person","confidence":0.9,"bbox":[250,50,390,400],"track_id":1}]}
```

IMU 数据示例，认证令牌需替换为私有配置：

```json
{"schema_version":1,"ts":1000,"token":"device-token","accel":{"x":0,"y":0,"z":9.81},"gyro":{"x":0,"y":0,"z":30},"yaw_rate_dps":30}
```

`ts` 单位是设备启动后的毫秒，accel 为 m/s²，gyro 和 yaw_rate_dps 为 deg/s。旧包没有显式 yaw 时按重力方向投影。包解析后不会转发 token。UDP 没有加密与防重放，只用于可信私网或 VPN，不得直接开放到公网。设备令牌只用于本地设备通信，不能复用云供应商密钥。

服务端保留最新待处理帧，限制输入和音频队列。普通提示受实时 IMU 新鲜度控制。`capture_to_trigger_latency_ms` 在缺少采集时间戳时保持空值。此时服务端另行报告从帧接收到音频触发的延迟。

本地服务不落盘媒体。扩展导航应用的录制由 `NAVGUIDE_RECORDING_ENABLED=true` 启用。摄像头数据只有明确发送 `/api/describe` 且 `include_latest_image=true` 时才会进入可选云描述。

## 扩展导航的检测状态

`python -m navguide.navigation.app` 使用 `BLIND_PATH_MODEL` 分割盲道与斑马线。模型缺失或推理失败时，导航结果返回 `detection_available=false`，`detection_reason` 分别为 `model_unavailable` 或 `inference_failed`，清除历史掩码与待播提示并暂停引导。正常推理但未检出目标时，`detection_available=true`，掩码为空。运行路径不生成模拟盲道。

交通灯识别统一使用 `TRAFFIC_LIGHT_MODEL`。只有当前帧与稳定检测结果一致的绿灯信号才能触发自动过街转换；模型不可用、推理失败或失去信号时清除历史结果并保持等待。控制台显示检测不可用状态，修复模型或输入后重新检测。

## 现场验收

按静止、转身超过阈值、重新观察、目标请求、IMU 断流、WiFi 断开重连、音频掉线的顺序测试。核对 torso IMU 安装方向、时钟方向与相机镜像。系统属于研究原型，现场验证应保留使用者原有的白杖或导盲辅助。
