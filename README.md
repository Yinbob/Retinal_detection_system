# Retinal Identity Experiments

本项目实现 Vistel 与 FIRE 眼底图像身份检索实验，包含数据审计、固定协议、
SIFT 局部匹配、HOG 全局匹配、阈值标定、正式评测和错误案例分析。

## 使用要求

- Python 3.11 或更高版本；项目曾在 Python 3.13.2 下验证。
- 需要自行获取有使用权限的 Vistel 和 FIRE 数据集。仓库不包含完整数据集，`prepare` 要求两个压缩包同时存在，且内容符合本项目的固定评测协议。
- 需要足够的磁盘空间保存原始压缩包、解压文件、特征缓存和实验结果；完整标定和评测可能需要较长时间。
- 安装依赖需要网络连接。精确依赖版本见 `requirements.lock.txt`。

## 安装

```bash
git clone https://github.com/Yinbob/Retinal_detection_system.git
cd Retinal_detection_system
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
```

以下命令均在项目根目录执行，并假定已激活虚拟环境。

## 数据集放置位置

在项目根目录创建 `dataset/`，将两个原始 ZIP 文件按以下名称放入；无需手动解压：

```text
Retinal_detection_system/
├── dataset/
│   ├── FIRE.zip
│   └── vistel_staff_fundus.zip
├── configs/default.yaml
└── README.md
```

文件名区分大小写。请自行从数据提供方获取对应数据并遵守其使用许可；项目不会自动下载数据集。`prepare` 会读取压缩包，在 `artifacts/datasets/` 下解压，并在 `artifacts/` 下生成清单、协议、缓存和结果，不会修改原始压缩包。

## 运行

按顺序执行完整流程：

```bash
# 解压数据并生成清单和固定评测协议
retina-auth prepare

# 在 Vistel 代理开发协议上标定并冻结 SIFT/HOG 参数
retina-auth calibrate

# 使用冻结参数评测 Vistel 和 FIRE
retina-auth evaluate

# 生成汇总表和错误案例报告
retina-auth report
```

运行完上述流程后，可对查询图进行检索：

```bash
retina-auth predict --query "/path/to/query.png" --dataset vistel --method sift
```

`--dataset` 可选 `vistel` 或 `fire`；`--method` 可选 `sift`、`sift_bidirectional` 或 `hog`。`predict` 需要先完成数据准备和参数标定，只读取查询图像内容和已注册参考库，不使用文件名中的身份、眼别或配对编号筛选候选。

## 本地可视化控制台

```bash
retina-auth serve
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
python -m pytest -q
python -m compileall -q src tests
```
