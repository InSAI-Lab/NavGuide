# 发布说明

## 检查与打包

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m pytest -q
bash scripts/test_firmware.sh
FIRMWARE_ENV=firmware_ci bash scripts/build_firmware.sh
python scripts/package_source.py
```

输出文件为 `dist/NavGuide-source.zip`。包内 `RELEASE_MANIFEST.json` 记录源文件大小与 SHA256，便于核对分发内容。

源代码包排除 Git 历史、环境变量文件、固件私有配置、模型权重、录制资料及媒体资源。设备固件二进制可能包含 WiFi 密码与设备令牌，发布源代码时不应附带个人构建的固件。`firmware_ci` 使用空凭据模板，仅用于验证构建。

## 发布要求

1. 审查包内文件与校验清单，确认不包含凭据或私人数据；撤销曾暴露的密钥。
2. 保留 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。单独分发的模型和媒体需有明确许可、下载地址及 SHA256。
3. 从审查后的源代码包建立公开版本。需要保留 Git 历史时，先检查历史提交中的凭据和私人资料。
4. 标明支持的 Python、运行平台和硬件配置。部署测试应包含设备连接、传感器、音频输出及云端 DNS、TLS 和模型权限。

算法评估见 [测试与复现条件](architecture.md#测试与复现条件)，设备部署见 [现场验收](deployment.md#现场验收)。
