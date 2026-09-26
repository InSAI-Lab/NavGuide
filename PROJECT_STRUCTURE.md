# 项目结构

| 路径 | 内容 |
| :--- | :--- |
| `navguide/__main__.py` | 本地服务命令行入口 |
| `navguide/core/context.py` | 任务、场景平滑和用户偏好 |
| `navguide/core/selection.py` | 同类去重、五因子评分、Top 3 和语义签名 |
| `navguide/core/gating.py` | 躯干角速度门控与紧急、目标绕过 |
| `navguide/core/proximity.py` | 相机内参与类别尺寸先验的粗距离估算 |
| `navguide/core/phrasing.py` | 中文、英文行动优先短句 |
| `navguide/core/pipeline.py` | P、T、B 条件与分阶段统计 |
| `navguide/i18n.py`, `navguide/locales/` | 中文文案、语音模板与共享关键词集合 |
| `navguide/runtime/` | FastAPI 服务、环境配置、设备认证、IMU、视觉和语音 |
| `navguide/cloud/` | 云描述、标签映射、语音识别和独立网关 |
| `navguide/navigation/` | 导航应用、任务调度、盲道与过街工作流 |
| `navguide/perception/` | 视觉检测、识别与跟踪 |
| `navguide/audio/` | 音频处理、播放与命令解析 |
| `navguide/io/` | 设备通信与媒体输入输出 |
| `firmware/`, `platformio.ini` | ESP32 固件与构建配置 |
| `deploy/` | 云容器、Compose 和 HTTPS 配置 |
| `scripts/evaluate.py` | 相同检测输入下的离线回放 |
| `scripts/package_source.py` | 源代码打包和凭据检查 |
| `scripts/` | 固件测试、构建与模型校验 |
| `examples/` | 回放输入与相机标定配置示例 |
| `tests/` | 算法、协议、服务、云端和固件测试 |
| `docs/` | 算法、部署、硬件与发布说明 |
| `web/index.html` | 本地服务控制台 |
| `web/templates/`, `web/static/` | 扩展导航页面与静态资源 |
| `.github/workflows/` | Python、容器和固件持续集成 |
| `model/` | 模型说明、清单模板和本地权重 |
| `voice/`, `music/`, `recordings/` | 本地音频资产和录制输出 |

`.env`、`firmware/config.h`、模型权重、录制资料和构建产物不进入源代码包。模型与音频资源按各自许可单独分发。
