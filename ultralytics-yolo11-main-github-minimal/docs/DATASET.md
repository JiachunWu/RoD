# Dataset Notes

本仓库不上传任何数据集文件，只说明本项目使用过的数据组织方式。

## 1. Dental_X

- 任务类型：YOLO 目标检测
- 类别数：31
- 目录结构：

```text
Dental_X/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

- `data.yaml` 示例：

```yaml
path: /path/to/Dental_X
train: images/train
val: images/val
test: images/test
names:
  0: Bone Loss
  1: Caries
  2: Crown
  ...
  30: wire
```

## 2. 30class

- 任务类型：YOLO 目标检测
- 类别数：10
- 目录结构：

```text
30class/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

- `data.yaml` 示例：

```yaml
path: /path/to/30class
train: images/train
val: images/val
test: images/test
names:
  0: Crown
  1: Filling
  2: Fracture teeth
  3: Implant
  4: Retained root
  5: Root Canal Treatment
  6: Root Piece
  7: abutment
  8: gingival former
  9: impacted tooth
```

## 获取方式

- 如果你已经在本地项目或实验平台中维护这两个数据集，直接复用原始数据并准备自己的 `data.yaml` 即可。
- 如果你准备公开复现，请只公开类别说明、目录结构和数据整理脚本，不要直接上传原始影像和标签，除非你明确拥有再分发权限。
- 如果数据来自第三方平台、合作医院、人工标注项目或内部私有数据源，请按原始授权方式获取。
