# Retinal Identity Experiments

本项目实现 Vistel 与 FIRE 眼底图像身份检索实验，包含数据审计、固定协议、
SIFT 局部匹配、HOG 全局匹配、阈值标定、正式评测和错误案例分析。

## 环境

本机未检测到 `conda`、`mamba` 或 `micromamba`，因此使用项目内 Python
虚拟环境。当前验证环境为：

- Python 3.13.2（`.venv`）
- 精确依赖版本见 `requirements.lock.txt`
- 计划建议 Python 3.11；当前实现及测试在 Python 3.13.2 下通过

重新创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
```

## 数据布局

原始压缩包保持只读，解压、清单、协议和缓存写入 `artifacts/`：

```text
dataset/
  FIRE.zip
  vistel_staff_fundus.zip
artifacts/
  datasets/
  manifests/
  protocols/
  cache/features/
  runs/
  reports/
```

## 运行

```bash
# 生成受控解压目录、数据清单和固定评测协议
.venv/bin/retina-auth prepare

# 仅在 Vistel 33/125 代理开发协议上搜索并冻结 SIFT/HOG 参数
.venv/bin/retina-auth calibrate

# 使用冻结参数评测 Vistel 和 FIRE 两种协议
.venv/bin/retina-auth evaluate

# 生成汇总表、S/P/A 分组、FIRE 配准误差和错误案例
.venv/bin/retina-auth report

# 对任意查询图执行全参考库检索
.venv/bin/retina-auth predict \
  --query "/path/to/query.png" \
  --dataset vistel \
  --method sift
```

`predict` 只读取查询图像内容和已注册参考库，不使用文件名中的身份、眼别或
配对编号筛选候选。

## 本地可视化控制台

```bash
.venv/bin/retina-auth serve
```

打开 [本地控制台](http://127.0.0.1:8765)。控制台包含四个模块：

- 光学识别台：协议查询图或 JPG/PNG 上传，支持 SIFT、双向 SIFT、HOG 单独或同台比对，点击候选图查看参考图。
- 协议基准对比：从现有报告读取准确率、FIRE 同眼命中率、分组结果与配准误差，可切换 Top-1 和带拒识统计口径。
- 阈值标定与改进：查看代理开发集标定结果、双向匹配对照和参数搜索记录。
- 错误分析与审计：按运行和类别筛选代表案例，载入识别台复现，并查看数据集与模型完整性审计。

协议图像支持 RGB、绿通道、FOV 掩膜和 CLAHE 预处理预览；真实几何内点连线仅在局部算法 Top-1 的 RGB / 绿通道下显示。上传查询显示原图。
页面兼容手机布局，支持键盘切换页签与分段选项。上传限制为 20 MB，所有检索请求均发往本地服务。

## 评测口径

- Vistel 查询固定 40 张、参考固定 158 张，同人同眼任意参考图均算正确。
- FIRE 同时保留原始 134 x 134 协议和排除查询自身副本的协议。
- FIRE 同时报告指定配对参考命中率与已确认同眼命中率。跨关系组的返回只表示
  “同眼关系未确认”，不被当作确定的异眼真值。
- 拒绝、无有效候选和匹配失败均保留在查询分母中。
- FIRE 对应点只用于附加配准误差分析，不参与阈值选择或身份识别。

## SuperRetina 状态

本地没有 SuperRetina 官方源码和预训练权重。当前版本不会加载随机初始化模型，
也不会伪造深度模型结果；`artifacts/model_audit.json` 会记录缺失状态。SIFT 与
HOG 仍可独立完成全流程。

## 测试

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m compileall -q src tests
```
