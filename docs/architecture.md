# 算法与评估

NavGuide 实现论文 Task-Aware Visual Content Selection and Inertial Gating for Wearable Spoken Guidance 中的语义内容选择与惯性门控方法。本文说明算法步骤、统计口径、回放格式与实验复现条件。

## 算法对应关系

| 论文步骤 | 代码入口 | 默认行为 |
| :--- | :--- | :--- |
| 同类空间去重 | `navguide.core.selection.same_class_dedup` | 同类框 IoU 严格大于 0.6 才去重，保留高置信度框 |
| 五因子排序 | `navguide.core.selection.SemanticMaximizationPolicy.select_cues` | `confidence × task × scene × user × proximity × hazard`，输出各因子值 |
| Top 3 | `select_cues` | 每次更新保留至多 3 个候选，分数相同时保持输入顺序 |
| 构造语义提示 | `build_cue` | 类别、时钟方向、粗距离，紧急与目标标记独立于最终分数 |
| 3 秒重复控制 | `filter_repetition` | 同签名间隔小于 3 秒时抑制，恰好 3 秒允许 |
| IMU 门控 | `navguide.core.gating.InertialMotionGate` | `abs(yaw_rate_dps) < 25` 时普通提示可通过，等于 25 时暂缓 |
| 紧急与目标绕过 | `filter_eligible_cues` | 绕过运动条件；Top 3 和重复控制仍然适用 |
| 行动优先表达 | `navguide.core.phrasing.ActionFirstPhraser` | 动作、时钟方向、粗距离、物体，支持中文和英文 |

被门控暂缓的提示在当前更新中丢弃，后续观测重新参与计算。重复历史在提示通过门控时提交，音频合成与播放由输出服务独立调度。

所有提示共享 3 秒重复窗口。紧急和目标标记仅作用于运动门控，重复抑制在此之前执行。

语义签名由类别、时钟方向和距离档位组成。红绿灯检测显式提供 `signal_color` 时，颜色也纳入签名，因此绿灯转为红灯会产生新的内容。信号状态仅按约定的交通灯类别和颜色字段解析。

## 参数来源

论文第 2 页 Algorithm 1 明确给出同类去重 IoU 阈值 0.6、Top 3、重复窗口 3 秒和惯性门限 25 度每秒；公开实现使用这些数值。第 2.2 节说明五因子权重经过 pilot 调节并在评估期间固定，但未列出各项权重值或配置版本。

以下为公开实现的默认权重，尚未与论文评估使用的 pilot 配置核实。源码中的定义是本发布版本的参数依据。

| 参数 | 公开实现默认值 | 定义位置 |
| :--- | :--- | :--- |
| 任务因子 | 目标匹配 4.5；路径类导航目标 2.4；完整分支取值 0.35 至 4.5 | [`compute_task_weight`](../navguide/core/selection.py) |
| 场景因子 | 人行道静态障碍 1.8；室内动态危险类别 0.4；完整分支取值 0.4 至 2.0 | [`compute_scene_weight`](../navguide/core/selection.py) |
| 用户因子 | 未设置的类别为 1.0，由 `user_weights` 覆盖 | [`compute_user_weight`](../navguide/core/selection.py) |
| 距离因子 | 小于 1.5 米为 2.5；1.5 至小于 2.5 米为 1.8；2.5 至小于 4.5 米为 1.2；其余为 0.6 | [`ProximityEstimator`](../navguide/core/proximity.py) |
| 危险因子 | 动态类别或运动目标距离小于 3 米为 2.5，其余为 1.8；静态障碍或显式危险目标距离小于 2 米为 1.8，其余为 1.3；普通目标为 1.0 | [`compute_hazard_weight`](../navguide/core/selection.py) |
| 危险敏感度 | 默认 1.0，乘入动态或静态危险目标的危险因子 | [`GuidanceContext`](../navguide/core/context.py) |

发布包内的 `RELEASE_MANIFEST.json` 记录上述源码文件的 SHA256，用于标识实际使用的公开参数版本。将其对应到论文实验需要固定的 pilot 权重表及其与评估运行的版本关联记录。

## 权重与距离估算

任务和场景权重、动态类别集合、距离分档及类别尺寸先验均有默认配置。每个输出候选的 `relevance_factors` 包含五个值，`relevance_score` 为五个因子与置信度的乘积，用于候选间的相对排序。

默认距离估算使用 60 度垂直视场角和类别高度先验。`CameraIntrinsics.calibrated_focal_length_y` 可指定参考图像尺寸下的实测像素焦距，运行时按图像高度缩放。裁剪、更换镜头或调整相机安装后，需要对应的标定参数。

`examples/camera_calibration.example.json` 展示配置格式，焦距由默认视场角计算。输出距离限定在 0.3 至 20 米，用于粗距离档位和近似措辞，不作为标定深度或碰撞剩余时间。核心策略对障碍物给出避让提示，对绿灯给出通行条件核实提示。

类别匹配采用完整词组，避免 `car` 命中 `carpet`。常见找物请求使用内置中文别名，其他中文名称由上游映射到检测器的实际类别名。

输入校验会拒绝无效框、非有限数值、负权重、错误配置和倒退的回放时间，并向调用方抛出异常。

## 统计口径

| 字段 | 定义 |
| :--- | :--- |
| `raw_item_count` | 当前帧输入检测数量 |
| `retained_item_count` | 同类去重和选择后的语义缓冲区数量，位于重复控制与运动门控之前 |
| `critical_detected_count` | 前端已经检测出的关键类别或显式危险标记数量 |
| `critical_retained_count` | 被选中候选与原始关键检测身份的交集数量 |
| `repetition_suppressed_count` | P 中因重复控制被抑制的数量；T 中未产生轨迹事件的数量 |
| `eligible_cues` | 通过选择、重复控制与门控的内容，处于音频调度之前 |
| `deferred_cues` | 当前帧被运动门控暂缓的诊断信息，不作为下一帧输入 |
| `processing_latency_ms` | `process()` 执行时间，包含本地策略和文字生成 |
| `capture_to_output_ready_latency_ms` | 仅传入真实采集时间时计算，到文本准备完成的延迟 |
| `capture_to_trigger_latency_ms` | 默认空值，仅显式记录实际音频触发后设置 |

累计 reduction 为 `1 - total_retained / total_raw`，累计 retention 为 `total_critical_retained / total_critical_detected`，均由累计计数计算。分母为零时返回 `null`。

关键项由车辆类、静态障碍类、红灯状态和显式危险标记定义。统计对象限于前端输出，分母不含漏检物体。同类重复框按原始检测身份分别计数，去除重复框也会降低该检测层面的保留率。真实场景召回率需另有标注数据。

采集与音频触发时间处于同一时钟域时，通过以下接口记录延迟：

```python
pipeline.record_audio_trigger(capture_timestamp, trigger_timestamp, result)
```

JPEG 接口未附带采集时间戳时，服务使用 `receive_to_audio_trigger_ms` 记录接收至音频触发的延迟，`capture_to_trigger_latency_ms` 保持空值。

## 基线

P 实现上述完整策略。B 每帧输出所有有效检测，不使用空间去重、时间抑制、Top 3 或运动门控。T 使用同类空间去重后出现的新轨迹、重现轨迹或方向及距离档位变化事件，使用检测描述语句，不使用任务与场景排序、Top K 或运动门控。

T 优先使用 `track_id`。缺少 ID 时采用同类 IoU 至少 0.3 的贪心关联，3 秒未更新后过期。

REMOTE 由独立云客户端运行，本地流水线收到该条件时抛出异常。其检测、网络和生成路径与本地条件不同；共享前端检测输入的选择策略比较使用 P、T、B。

## 离线回放

核心算法及回放只依赖 Python 标准库，不需要摄像头、GPU、模型文件或云密钥。在项目目录执行：

```sh
python3 scripts/evaluate.py \
  --input examples/synthetic_replay.jsonl \
  --conditions P T B \
  --output /tmp/navguide-report.json \
  --frames-output /tmp/navguide-trace.jsonl
```

示例为合成序列，覆盖重复、转身、暂缓后重新观察、紧急提示、目标搜索、Top 3 和空帧。

报告保存输入 SHA256、阈值、统计分母、计数阶段与指标适用范围。逐帧 JSONL 保存候选分数、五因子、语义签名、可输出与暂缓内容。相同输入的策略决策与计数可重复，实际处理耗时会随机器和负载变化。

每行输入一个 JSON 对象，最小输入为：

```json
{"timestamp":0.0,"yaw_rate_dps":0.0,"detections":[]}
```

`timestamp` 为同一回放时间线上的秒数，必须非递减。`yaw_rate_dps` 必须是躯干 yaw 角速度的度每秒值。`frame_width`、`frame_height` 默认 640 与 480，应与框坐标所属图像一致。框采用像素坐标 `[x1,y1,x2,y2]`。

检测对象必须包含 `category`、`confidence`、`bbox`，可包含 `track_id`、`is_hazard`、`is_moving`、`urgency_override`、`signal_color` 和 `raw_data`。类别和分数必须来自实际前端，危险与运动标记应记录赋值依据。可选上下文包括 `task_mode`、`target_query`、`scene` 和 `user_weights`。切换任务或目标会清空重复历史。

```json
{"timestamp":1.0,"yaw_rate_dps":30.0,"frame_width":640,"frame_height":480,"task_mode":"target_search","target_query":"cup","scene":"indoor","detections":[{"category":"cup","confidence":0.92,"bbox":[250,100,310,140],"track_id":5}]}
```

使用 `--calibration examples/camera_calibration.example.json` 加载标定配置。`eligible_items_per_minute_over_timestamp_span` 按可输出物体项计数，并以首末输入时间跨度换算每分钟数量；单帧或零跨度返回 `null`。

## 测试与复现条件

```sh
python3 -m unittest discover -s tests -p 'test_algorithm_contract.py' -v
python3 -m unittest discover -s tests -p 'test_navigation.py' -v
```

算法测试仅依赖标准库，覆盖阈值边界、错误输入、各阶段计数和回放。导航集成测试需要项目运行依赖。

源代码包提供默认参数和合成回放示例，用于功能验证。论文数值结果复现需使用对应版本的研究数据、固定实验参数、相机标定、音频触发日志和硬件环境。

惯性门控实验使用实时躯干角速度输入，并记录检测数据来源、采集条件与 IMU 安装坐标系。模型精度使用标注数据评估，可听延迟在目标设备上测量。比较 items/min 时，采用相同的计数阶段和时间窗口。
