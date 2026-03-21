# VisionNDVI 脚本使用指南

## 快速开始

现在你只需要使用一个脚本：`./run.sh`

```bash
# 查看帮助
./run.sh help

# 查看任务状态
./run.sh status
```

---

## 常用操作对照表

| 旧方式 | 新方式 |
|--------|--------|
| `bash run_inference_no_finetune.sh` | `./run.sh inference nofinetune` |
| `bash run_inference_background.sh AWI 201501 202012` | `./run.sh inference nofinetune --start 201501 --end 202012 --bg` |
| `bash run_all_models.sh 201501 202012` | `./run.sh batch 201501 202012` |
| `bash train_adapter.sh` | `./run.sh train adapter` |
| `bash train_adapter_bg.sh` | `./run.sh train adapter --bg` |
| `bash manage_tasks.sh status` | `./run.sh status` |
| `bash manage_tasks.sh stop` | `./run.sh stop` |
| `bash manage_tasks.sh logs` | `./run.sh logs` |

---

## 具体用法示例

### 1. 推理

```bash
# 滚动推理（支持在线微调）
./run.sh inference rolling --start 201501 --end 205012 --labeled-end 202012

# 无微调推理
./run.sh inference nofinetune --start 201501 --end 202012

# 后台运行
./run.sh inference rolling --start 201501 --end 202012 --bg

# 使用自定义模型/数据
BASE_MODEL=./checkpoints/BCC_prediction_model.pth \
DATASET_DIR=./datasets/BCC-CM-1-1-MR/ \
./run.sh inference nofinetune --start 201501 --end 202012
```

### 2. 批量推理

```bash
# 推理所有模型（串行，避免显存冲突）
./run.sh batch 201501 202012
```

### 3. 训练

```bash
# 前台训练 Adapter
./run.sh train adapter

# 后台训练，自定义参数
./run.sh train adapter --epochs 10 --lr 1e-3 --bg
```

### 4. 任务管理

```bash
# 查看状态
./run.sh status

# 查看日志
./run.sh logs              # 查看最新日志
./run.sh logs inference    # 查看推理日志

# 停止任务
./run.sh stop              # 停止所有任务
./run.sh stop inference    # 停止特定任务

# 查看结果
./run.sh results

# 清理旧日志
./run.sh clean
```

---

## 文件变更说明

### Python 脚本

| 文件 | 状态 | 说明 |
|------|------|------|
| `inference_rolling.py` | ✅ 保留 | 功能最全的推理脚本 |
| `inference_no_finetune.py` | ✅ 保留 | 无微调推理 |
| `inference_legacy.py` | ⏳ 备份 | 原 `inference.py`，功能已合并到 rolling |
| `utils/inference.py` | ✅ 保留 | 工具函数，不是可直接运行的脚本 |

### Shell 脚本

| 文件 | 状态 | 说明 |
|------|------|------|
| `run.sh` | ✅ **新建** | **统一入口脚本，推荐用这个** |
| `run_inference_no_finetune.sh` | ⏳ 可选删除 | 功能合并到 `run.sh` |
| `run_inference_background.sh` | ⏳ 可选删除 | 功能合并到 `run.sh` |
| `run_all_models.sh` | ⏳ 可选删除 | 功能合并到 `run.sh` |
| `manage_tasks.sh` | ⏳ 可选删除 | 功能合并到 `run.sh` |
| `train_adapter.sh` | ⏳ 可选删除 | 功能合并到 `run.sh` |
| `train_adapter_bg.sh` | ⏳ 可选删除 | 功能合并到 `run.sh` |

---

## 如果你想保留旧脚本

旧脚本仍然可以使用，不会互相干扰。如果你适应了新脚本，可以删除旧脚本：

```bash
# 删除旧脚本（可选）
rm run_inference_*.sh run_all_models.sh manage_tasks.sh train_adapter*.sh
```

或者保留它们作为备用。

---

## 环境变量

可以通过环境变量自定义默认路径：

```bash
export DATASET_DIR="./datasets/AWI-CM-1-1-MR/"
export BASE_MODEL="./checkpoints/AWI_prediction_model.pth"
export ADAPTER="./checkpoints/adapter_v3/DeepMultiTimeAdapter_best.pth"
export OUTPUT_DIR="./results"
```
