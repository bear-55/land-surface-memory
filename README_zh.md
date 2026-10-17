# 使用说明

本仓库提供论文对应的研究代码，包含 B0、A1、M1 的模型与训练代码、
固定的 370/50/107 流域划分、十个配对种子、数据准备、门控和跨时期分析，以及
可比较的参考结果。

## 先验证，再运行完整实验

1. 按 [英文 README](README.md) 配置 Python 3.12 和依赖。
2. 在本目录运行以下三个命令，无须下载 CAMELS 即可检查代码：

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

3. 按 [数据说明](docs/data.md) 获取 CAMELS 原始数据，将自己的路径传给
   `scripts/prepare_data.py`。随后依次执行训练、两时期评价、门控诊断和分析。
   完整命令见英文 README；不需要修改 Python 源码中的路径。

数据准备的预期输出为 527 个流域、1,341,831 个训练窗口和 73,050 个验证窗口。
`scripts/verify_release.py --cache` 可核对生成的标准化参数与固定参考值。
完整实验运行后，使用 `scripts/analysis_verify.py` 比较分析结果；各项检查的用途
见 [验证说明](docs/verification.md)。

原始 CAMELS 数据与模型权重不随源代码分发。原始数据获取方式见数据说明，
权重可通过训练命令生成。程序的 MIT 许可证不替代原数据提供者的使用条款。
