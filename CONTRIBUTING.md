# 参与贡献

使用 Python 3.10 至 3.12，在虚拟环境中安装开发依赖：

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m pytest -q
```

算法变更应说明对应步骤、参数来源和对候选选择、重复抑制、运动门控的影响。修复应包含能覆盖问题的测试。设备或云端测试结果需说明硬件、依赖版本、测量时钟与输入来源。

固件变更运行：

```bash
bash scripts/test_firmware.sh
FIRMWARE_ENV=firmware_ci bash scripts/build_firmware.sh
```

提交应聚焦功能、修复或必要文档。密钥、私有固件配置、个人录制、模型二进制和未授权资产不得提交。保留第三方版权与许可证。

算法评估的数据、参数和测量要求见 [测试与复现条件](docs/architecture.md#测试与复现条件)。
