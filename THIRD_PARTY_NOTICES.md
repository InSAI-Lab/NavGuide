# 第三方来源与许可

NavGuide 包含源自 [AI-FanGe/OpenAIglasses_for_Navigation](https://github.com/AI-FanGe/OpenAIglasses_for_Navigation) 的代码。相关代码遵循根目录 [MIT License](LICENSE)，保留 `Copyright (c) 2025 AI-FanGe` 声明。

Ultralytics 软件与模型遵循其供应方授权，详见 [Ultralytics 许可](https://www.ultralytics.com/license)。YOLOE 权重、训练数据、MobileCLIP 文本编码器、MediaPipe 手部模型及专用检测权重需分别保留来源与许可记录，根目录许可证不替代其授权。

Arduino ESP32、esp32-camera、ArduinoWebsockets 和 espeak-ng 遵循各自许可证。固件依赖版本在 `platformio.ini` 中配置。

本地服务使用 FastAPI、Pydantic 与 Uvicorn；云客户端使用 OpenAI Python SDK 调用阿里云兼容接口，语音识别通过 Paraformer WebSocket 接口完成。各依赖的许可由其发行包提供，供应商 API 凭证不得再分发。

`voice/`、`music/` 与其他本地媒体资源须确认来源与分发许可后另行提供，默认不包含在源代码包中。参与者照片、音视频和研究记录也不包含在源代码包中。
