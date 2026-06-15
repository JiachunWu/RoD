# Ultralytics YOLO11/12 Dental Minimal Release

这是一个面向 GitHub 公开的精简版本，来源于本地项目 `/data/wjc/ultralytics-yolo11-main`，只保留了与牙科全景片检测实验直接相关的代码、模型配置和评估脚本。

本仓库不包含以下内容：

- 数据集图片与标签
- 训练权重
- `runs/`、`outputs/`、`result/` 等实验产物
- 示例工程、测试缓存和无关大文件

## 保留内容

- `ultralytics/`
  自定义过的 Ultralytics 源码，包含当前实验依赖的自定义模块注册与实现。
- `scripts/train.py`
  参数化训练入口。
- `scripts/val.py`
  参数化验证入口。
- `tools/artifact_experiment/`
  `Dental_X` 数据集的 artifact-source 分层评估脚本。
- `tools/artifact_experiment_v2_30class/`
  `30class` 数据集的分层评估、公平评估和 tuned 推理脚本。
- `docs/artifact-experimen.md`
  `Dental_X` 版本实验方案。
- `docs/artifact-experimen-v2.md`
  `30class` 版本实验方案。

## 环境

推荐 Python 3.10+，并先安装依赖：

```bash
pip install -r requirements.txt
```

如果你使用的是干净环境，通常还需要手动确认这些包可用：

```bash
pip install ultralytics prettytable pywavelets dill timm
```

## 训练示例

```bash
python scripts/train.py \
  --model ultralytics/cfg/models/12/our_WTconv_DYT.yaml \
  --data path/to/30class/data.yaml \
  --name class30_our_WTconv_DYT
```

## 验证示例

```bash
python scripts/val.py \
  --model path/to/best_fp32.pt \
  --data path/to/30class/data.yaml \
  --split test \
  --save-json
```

## 分层评估

两套分层评估脚本都保留了：

- `tools/artifact_experiment/`: 对 `Dental_X` 使用两模型分层评估。
- `tools/artifact_experiment_v2_30class/`: 对 `30class` 使用三模型公平评估与 tuned 推理补测。

这些脚本默认需要你自己提供本地数据集路径和权重路径。

## 数据集说明

本项目实验中使用了两个 YOLO 格式数据集：

- `Dental_X`
  31 类牙科全景片检测数据集，目录结构为 `images/{train,val,test}` 与 `labels/{train,val,test}`。
- `30class`
  10 类牙科全景片检测数据集，目录结构同样为标准 YOLO 检测格式。

仓库不提供数据本体。如果你已有对应数据，只需准备自己的 `data.yaml`。如果没有，需要从原始标注来源或你的项目数据管理位置自行获取后，整理为标准 YOLO 检测格式。

更具体的使用方式见 [docs/DATASET.md](docs/DATASET.md)。
