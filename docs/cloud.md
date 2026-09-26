# 可选云描述服务

云网关提供用户主动请求的场景描述和物品标签规范化。本地候选选择、惯性门控与导航指令独立运行，可通行判断和紧急避障不属于网关接口的功能。本地服务通过 `python -m navguide` 启动。

## 默认离线与客户端配置

`NAVGUIDE_CLOUD_ENABLED=false` 默认关闭远程描述。云连接在功能调用时建立；本地标签映射可独立于 SDK 和云凭证使用。已知中文物品和符合格式的英文标签在本地处理，其他名称通过启用的云服务解析。解析失败时返回空标签，具体接口行为见下文。

安装可选依赖：

```bash
python -m pip install -r requirements-cloud.txt
```

经网关连接时，客户端设置：

```dotenv
NAVGUIDE_CLOUD_ENABLED=true
NAVGUIDE_CLOUD_URL=https://navguide.example.org
NAVGUIDE_CLOUD_TOKEN=填写服务器生成的网关令牌
CLOUD_TIMEOUT_SECONDS=35
```

直连模式不需要网关，清空 `NAVGUIDE_CLOUD_URL`，设置 `NAVGUIDE_CLOUD_ENABLED=true`、`DASHSCOPE_API_KEY`、`DASHSCOPE_COMPAT_BASE`。密钥通过环境变量提供。只允许 HTTPS 云端地址；`http://127.0.0.1:8090` 等回环地址可用于本地开发。

| 参数 | 默认值 | 用途 |
| :--- | :--- | :--- |
| `QWEN_OMNI_MODEL` | `qwen-omni-turbo` | 与论文远程参考条件对应，需在所选区域具备访问权限 |
| `QWEN_TEXT_MODEL` | `qwen-turbo` | 物品标签规范模型 |
| `DASHSCOPE_COMPAT_BASE` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 阿里云兼容端点，可设置工作空间及区域端点 |
| `CLOUD_TIMEOUT_SECONDS` | `30` | 上游整轮生成最长时间，也约束请求读取时间 |
| `CLOUD_MAX_OUTPUT_TOKENS` | `1024` | 单次描述输出上限，范围 1 至 4096 |
| `CLOUD_MAX_REQUEST_BYTES` | `8388608` | 网关 JSON 原始请求大小上限 |
| `CLOUD_MAX_CONCURRENT_REQUESTS` | `4` | 单进程正在处理的上游请求上限 |
| `NAVGUIDE_GATEWAY_TOKEN` | 空 | 服务端令牌，至少 32 个字符 |

模型、区域、工作空间和音色需匹配阿里云账户的可用资源。默认模型名称与论文远程条件对应，切换模型后应在评估记录中使用新的模型标识。客户端与网关各有超时设置，客户端超时宜略长于网关。

## 在云主机部署

部署需要安装 Docker Engine 与 Compose 插件的 Linux 云主机、指向该主机的域名，以及具有所选模型访问权限的阿里云百炼密钥。

1. 把代码复制到云主机，进入项目目录。
2. 执行 `cp deploy/.env.cloud.example deploy/.env.cloud`。
3. 执行 `python -c "import secrets; print(secrets.token_urlsafe(48))"`，把输出保存为 `NAVGUIDE_GATEWAY_TOKEN`。填入域名、证书联系邮箱和 `DASHSCOPE_API_KEY`。本地客户端的 `NAVGUIDE_CLOUD_TOKEN` 使用相同网关令牌。
4. 执行 `chmod 600 deploy/.env.cloud`，将凭据文件限制为当前用户可读写，并保留在服务器本地。
5. 为服务器开放 TCP 80 和 443，配置域名 A/AAAA 记录。8090 仅在容器网络提供服务。
6. 校验并启动：

```bash
docker compose -f deploy/compose.cloud.yaml config --quiet
docker compose -f deploy/compose.cloud.yaml up -d --build
docker compose -f deploy/compose.cloud.yaml ps
curl --fail https://navguide.example.org/healthz
curl --fail https://navguide.example.org/readyz
```

Caddy 使用配置的域名自动获取和更新 TLS 证书，并即时转发描述流。证书数据保存在独立卷。网关容器以非 root 用户运行，只读文件系统，不存储摄像头图片、请求正文或音频。

`/healthz` 返回进程状态，`/readyz` 检查密钥存在及令牌长度。这两个接口不调用上游模型；账户额度、模型权限和外网连通性通过实际请求检查。`/v1/describe` 与 `/v1/label` 请求按供应商规则计费。

本地开发可先导出所需环境变量，再执行：

```bash
uvicorn navguide.cloud.gateway:app --host 127.0.0.1 --port 8090 --workers 1
```

并发额度按进程计数，多个 worker 的总额度为各进程额度之和。多用户部署可在入口按账户配置配额，并在供应商侧设置预算告警。供应商密钥保存在网关或启用直连功能的本地服务中，浏览器和 ESP32 使用设备或网关令牌。

## HTTP 契约与错误处理

所有 `/v1/` 接口需要 `Authorization: Bearer <网关令牌>`。

`POST /v1/describe` 请求示例：

```json
{"content":[{"type":"text","text":"请简要描述眼前场景"}],"voice":"Cherry","audio_format":"wav"}
```

可以追加 `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`。最多 8 个内容项，文本项最长 4096 字符。图片格式为内联 JPEG、PNG 或 WebP base64，外部图片 URL 会被拒绝。请求大小同时按 `Content-Length` 和实际接收字节数检查，适用于分块传输。

成功响应为逐行 JSON：

```json
{"type":"delta","text_delta":"前方有座椅","audio_b64":null}
{"type":"delta","text_delta":null,"audio_b64":"..."}
{"type":"done"}
```

`audio_b64` 使用 Qwen `audio.data` 的 24 kHz、单声道、16 bit PCM 分片。请求中的 `audio_format` 为 `wav`，流中的分片仍按 PCM 解码和重采样。使用其他音频格式的模型时，播放端需适配对应的采样率和编码。

首个有效分片产生前，上游异常返回 502，超时返回 504。响应开始后，异常通过 `{"type":"error","code":"upstream_timeout"}` 等终止事件发送，客户端据此结束描述。错误响应仅包含错误代码。`done` 表示正常完成，缺少该事件的断流按失败处理。

`POST /v1/label` 接受 `{"query":"灭火器"}`，成功返回 `{"label":"fire extinguisher"}`。标签需通过格式校验，无效上游响应返回 502。客户端在配置缺失、请求失败或标签无效时返回 `("", "fallback")`。本地服务的找物命令返回 `ERR:UNKNOWN_TARGET`；扩展导航应用在界面显示目标名称识别失败消息。两者均在更新任务目标之前结束处理。

其他响应：401 表示令牌不匹配，413 表示请求超限，422 表示输入格式错误，429 表示本进程请求额度已满，503 表示缺少服务器配置或上游限流。客户端失败时停止云描述，继续本地导航流程。并发额度在断流、取消、失败及正常结束时释放。

## 可选语音识别

本地服务通过 `navguide.cloud.asr.ASRSession` 使用 Paraformer，把用户语音转成命令文本。此通道直接连接百炼 WebSocket，不经过描述网关，默认同样禁用：

```dotenv
NAVGUIDE_ASR_ENABLED=false
DASHSCOPE_API_KEY=
DASHSCOPE_ASR_URL=wss://dashscope.aliyuncs.com/api-ws/v1/inference
ASR_MODEL=paraformer-realtime-v2
ASR_CONNECT_TIMEOUT_SECONDS=10
ASR_MAX_SESSION_SECONDS=60
```

要启用语音识别，将 `NAVGUIDE_ASR_ENABLED` 改为 `true` 并填写同区域密钥与 WSS 端点。即使描述走网关，语音识别仍需本地服务持有独立的供应商凭证。语音识别与描述分别启停；关闭云功能时仍可发送文本命令。

会话接口：`await session.start()` 启动，`session.feed(pcm)` 提交 16 kHz 单声道 PCM16，`await session.stop()` 发送结束命令并接收尾句。`await session.stop(cancel=True)` 用于设备断开或任务中断。`on_final(text)`、`on_partial(text)` 与 `on_error(code)` 在调用方的事件循环中执行，支持异步函数。每个 START 创建新会话，设备连接结束时由调用方停止。

网络操作在独立线程中执行，输入队列最多 64 帧，每帧不超过 16384 字节。队列溢出时取消当前语音会话并返回错误。会话默认最长 60 秒；正常结束时等待尾句最多 3 秒，取消操作直接中止连接。

## 测试与运维

```bash
python -m pip install pytest
python -m pytest tests/test_cloud_clients.py tests/test_cloud_gateway.py tests/test_cloud_asr.py
docker compose -f deploy/compose.cloud.yaml logs --tail=100 gateway caddy
docker compose -f deploy/compose.cloud.yaml down
```

自动测试使用模拟上游，覆盖离线导入、标签回退、异步流、认证、大小限制、并发、超时和异常处理。部署环境的检查项包括容器健康状态、DNS 解析、TLS 证书和实际模型请求。

## 官方接口依据

音频输出使用 `stream=True`、`modalities=["text","audio"]` 和 `audio` 参数，OpenAI Python SDK 最低要求为 1.52.0。区域端点与流式 PCM 示例见 [阿里云 Qwen Omni 文档](https://www.alibabacloud.com/help/en/model-studio/qwen-omni)。

异步 SDK 使用 `AsyncOpenAI`、`await` 和异步流迭代，显式关闭客户端，见 [OpenAI Python SDK 官方说明](https://github.com/openai/openai-python)。

TLS 与转发配置见 [Caddy Automatic HTTPS](https://caddyserver.com/docs/automatic-https) 和 [reverse_proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy)。FastAPI 流输出依据 [StreamingResponse 官方说明](https://fastapi.tiangolo.com/advanced/custom-response/)。

语音识别依据 [Paraformer WebSocket 接口](https://help.aliyun.com/en/model-studio/websocket-for-paraformer-real-time-service)、[客户端事件](https://help.aliyun.com/en/model-studio/paraformer-client-events) 和 [服务端事件](https://help.aliyun.com/en/model-studio/paraformer-server-events)。音频只在收到 `task-started` 后发送，尾句结束使用 `sentence_end` 字段判断。

## 代理环境

HTTP 描述客户端通过 HTTPX 支持 HTTP(S) 和 SOCKS 代理。实时 ASR 使用 websocket-client，支持直连和 HTTP 代理；其 WSS 连接需单独配置，不继承 HTTPX 的 SOCKS 处理逻辑。
