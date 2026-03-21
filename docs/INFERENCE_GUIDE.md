# 无微调推理脚本使用指南

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `inference_no_finetune.py` | 核心推理脚本（严格只读模式） |
| `run_inference_background.sh` | 单模型后台推理 |
| `run_all_models.sh` | 批量运行所有模型 |
| `manage_tasks.sh` | 任务管理（查看状态、停止、日志等） |

---

## 快速开始

### 1. 单模型后台推理

```bash
# 使用默认参数 (AWI模型, 201501-202012)
./run_inference_background.sh

# 指定模型和日期范围
./run_inference_background.sh AWI 201501 202012

# 其他模型
./run_inference_background.sh BCC 201501 202012
./run_inference_background.sh MPI 201501 202012
```

**日志位置**: `logs/inference_模型名_日期范围_时间戳.log`

---

### 2. 批量运行所有模型

```bash
# 运行所有模型（串行，避免显存不足）
./run_all_models.sh 201501 202012
```

---

### 3. 任务管理

```bash
# 查看所有任务状态
./manage_tasks.sh status

# 停止所有任务
./manage_tasks.sh stop

# 查看最新日志
./manage_tasks.sh logs

# 查看特定模型日志
./manage_tasks.sh logs AWI

# 查看结果汇总
./manage_tasks.sh results

# 清理旧日志
./manage_tasks.sh clean
```

---

## 支持的模型

- `AWI` - AWI-CM-1-1-MR
- `BCC` - BCC-CSM2-MR
- `CMCC_CM2_HR4` - CMCC-CM2-HR4
- `CMCC_CM2_SR5` - CMCC-CM2-SR5
- `CMCC_ESM2` - CMCC-ESM2
- `FIO` - FIO-ESM-2-0
- `MPI` - MPI-ESM1-2-HR

---

## 输出结构

```
results/no_finetune_模型名_开始日期_结束日期/
├── predictions/          # .npy预测文件
├── visualizations/       # 前20个样本的可视化
└── metrics.csv          # 整体指标统计
```

---

## 示例工作流程

```bash
# 1. 启动AWI模型推理（后台）
./run_inference_background.sh AWI 201501 202012

# 2. 查看运行状态
./manage_tasks.sh status

# 3. 实时查看日志
./manage_tasks.sh logs AWI

# 4. 完成后查看结果
./manage_tasks.sh results
```

---

## 注意事项

1. **显存管理**: 批量运行时会串行执行，避免GPU显存不足
2. **日志保留**: 自动清理7天前的日志文件
3. **任务冲突**: 同一模型同时只能运行一个任务，启动新任务会自动提示终止旧任务
