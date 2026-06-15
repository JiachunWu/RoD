# Execution Context

## 2026-06-15 15:25 - T1: 初始化打包任务

### 当前状态

已创建新的独立目录 `/data/wjc/ultralytics-yolo11-main-github-minimal`，本次所有 GitHub 打包工作都将在该目录中完成，不触碰原项目文件。

### 关键发现

- 用户要求先规划，再执行实际打包。
- 用户明确要求只上传最简代码版本，不上传数据集。
- 原项目的 `context.md` 需要作为打包范围判定的重要依据。

### 已修改文件

- `task.md`: 新建本次打包任务计划。
- `context.md`: 新建本次打包执行上下文。

### 决策记录

- 决策：新建独立发布目录承载全部待上传文件。
- 原因：满足“不直接修改原始项目”的约束。

### 下一步

读取原项目上下文和关键源码，确认最简可公开文件范围。

## 2026-06-15 15:33 - T2: 最简公开范围分析中

### 当前状态

已完成原项目 `context.md` 的通读，并开始追踪当前实验实际依赖的训练入口、模型 YAML、评估脚本和自定义 `ultralytics` 模块。

### 关键发现

- 原项目目录本身不是一个已初始化的 Git 仓库，因此新目录需要作为可直接上传 GitHub 的独立发布目录来组织。
- 与当前 30class / Dental_X 实验直接相关的脚本主要集中在 `tools/artifact_experiment*`、`train.py`、`val.py` 和 `ultralytics/cfg/models/12/`。
- 目标模型 YAML 并不只依赖官方 `ultralytics`，还依赖自定义模块注册与实现，尤其是 `C3k2_WTConv`、`A2C2f_DFFN`、`A2C2f_DYT`。
- 这些核心改动最终都落在 `ultralytics/nn/tasks.py`、`ultralytics/nn/extra_modules/block.py`、`wtconv2d.py`、`transformer.py`、`EVSSM.py` 一带，但当前实验实际用到的能力范围远小于整个大仓库。

### 已修改文件

- `context.md`: 追加 T2 分析进展。

### 决策记录

- 决策：优先构造“基于官方 ultralytics 的最小扩展包”，而不是复制整个原仓库。
- 原因：这样更符合“最简上传版本”的要求，同时保留实验复现所需核心代码。

### 下一步

提取当前实验真正需要的最小自定义模块、配置和脚本，落到新目录中。

## 2026-06-15 15:38 - T3/T4/T5: 精简发布目录完成

### 当前状态

已在新目录中完成 GitHub 精简版打包。当前目录可作为独立上传目录使用，且未包含数据集、权重、训练输出或测试结果。

### 关键发现

- 仅复制 YAML 和脚本不足以保证可运行，因此最终保留了自定义过的 `ultralytics/` 代码包，以覆盖当前实验依赖的模块注册与实现。
- 已将 `scripts/train.py` 和 `scripts/val.py` 改为参数化入口，不再依赖本机绝对路径。
- 已补充 `README.md`、`docs/DATASET.md` 和 `.gitignore`，明确说明不上传数据集、权重和输出目录。
- `artifact_experiment` 与 `artifact_experiment_v2_30class` 两套分层评估脚本都已保留。
- `python3 -m compileall` 已通过，说明新目录中的关键脚本语法正常。

### 已修改文件

- `README.md`: 新增精简版仓库说明、安装方式、训练/验证示例和数据集说明入口。
- `docs/DATASET.md`: 新增 `Dental_X` 与 `30class` 的目录结构和获取说明。
- `.gitignore`: 新增忽略数据、权重、运行结果和缓存的规则。
- `scripts/train.py`: 改为参数化训练入口。
- `scripts/val.py`: 改为参数化验证入口。
- `tools/artifact_experiment/`: 复制 `Dental_X` 分层评估脚本。
- `tools/artifact_experiment_v2_30class/`: 复制 `30class` 分层评估、公平评估和 tuned 脚本。
- `configs/models12/`: 复制当前实验相关的模型 YAML。
- `ultralytics/`: 复制当前实验运行所需的自定义源码包。
- `requirements.txt`, `pyproject.toml`, `LICENSE`: 复制基础依赖和许可文件。
- `task.md`: 标记全部任务完成。
- `context.md`: 追加完成记录。

### 决策记录

- 决策：最终保留精简后的完整 `ultralytics/` 包，而不是只保留极小补丁。
- 原因：这样更稳妥，能确保自定义 YAML、模块注册和实验脚本在公开目录中保持一致可运行。
- 决策：不复制 `runs/`、`outputs/`、`result/`、数据集目录和权重文件。
- 原因：用户明确要求只上传最简代码版本，不上传数据集。

### 下一步

向用户汇报新目录位置、已保留内容和验证结果。
