# Adapter 预训练与滚动预测工作流

本文档描述如何使用 `train_adapter.py` 和 `inference_rolling.py` 实现完整的 Adapter 预训练与滚动预测流程。

## 架构概述

```
阶段1: 基础模型训练（已完成）
    ├── 输入：有标签数据（1982-2014）
    ├── 输出：预训练基础模型（UNet/CNN/3D UNet）
    └── 脚本：train_OF_*.py

阶段2: Adapter预训练（新增）
    ├── 输入：基础模型 + 有标签数据（1982-2014）
    ├── 冻结基础模型，训练Adapter
    └── 脚本：train_adapter.py

阶段3: 滚动预测（新增）
    ├── 有标签期间（2015-2020）：在线微调Adapter，阶段保存权重
    ├── 无标签期间（2021+）：直接使用预训练Adapter
    └── 脚本：inference_rolling.py
```

## 快速开始

### 1. Adapter 预训练

```bash
python train_adapter.py \
    --base_model ./checkpoints/best_unet_model.pth \
    --dataset_dir ./datasets/AWI-CM-1-1-MR/ \
    --output_dir ./checkpoints/adapter \
    --adapter_type DeepMultiTimeAdapter \
    --window_size 3 \
    --start_date 198201 \
    --end_date 201412 \
    --epochs 50 \
    --lr 1e-3
```

**输出文件：**
```
checkpoints/adapter/
├── DeepMultiTimeAdapter_best.pth       # 验证损失最低的模型
├── DeepMultiTimeAdapter_latest.pth     # 最后一个epoch的模型
└── DeepMultiTimeAdapter_history.json   # 训练历史（损失曲线）
```

### 2. 滚动预测

#### 场景A：验证集（2015-2020有标签，在线微调）

```bash
python inference_rolling.py \
    --base_model ./checkpoints/best_unet_model.pth \
    --adapter ./checkpoints/adapter/DeepMultiTimeAdapter_best.pth \
    --dataset_dir ./datasets/AWI-CM-1-1-MR/ \
    --start_date 201501 \
    --end_date 202012 \
    --online_finetune \
    --finetune_iterations 50 \
    --output_dir ./results/validation
```

#### 场景B：纯预测（2021-2050无标签，无微调）

```bash
python inference_rolling.py \
    --base_model ./checkpoints/best_unet_model.pth \
    --adapter ./checkpoints/adapter/DeepMultiTimeAdapter_best.pth \
    --dataset_dir ./datasets/AWI-CM-1-1-MR/ \
    --start_date 202101 \
    --end_date 205012 \
    --no_online_finetune \
    --output_dir ./results/prediction
```

#### 场景C：混合（2015-2020有标签微调，2021+无标签预测）

```bash
python inference_rolling.py \
    --base_model ./checkpoints/best_unet_model.pth \
    --adapter ./checkpoints/adapter/DeepMultiTimeAdapter_best.pth \
    --dataset_dir ./datasets/AWI-CM-1-1-MR/ \
    --start_date 201501 \
    --end_date 205012 \
    --labeled_end_date 202012 \
    --online_finetune \
    --save_adapter_stages \
    --reference_geotiff ./reference.tif \
    --output_dir ./results/mixed
```

#### 场景D：高质量可视化输出（论文用图）

```bash
python inference_rolling.py \
    --base_model ./checkpoints/best_unet_model.pth \
    --adapter ./checkpoints/adapter/DeepMultiTimeAdapter_best.pth \
    --dataset_dir ./datasets/AWI-CM-1-1-MR/ \
    --start_date 201501 \
    --end_date 202012 \
    --online_finetune \
    --viz_formats png,svg,pdf \              # 多格式输出
    --residual_bin_width 0.001 \             # 精细的残差分布
    --num_viz 20 \                           # 可视化更多样本
    --output_dir ./results/paper_quality
```

## 输出目录结构（规范化）

运行 `inference_rolling.py` 后会生成以下**规范化**的目录结构：

```
output_dir/
├── predictions/                     # 预测结果
│   ├── npy/                         # NumPy格式
│   │   ├── 201501_pred.npy
│   │   ├── 201502_pred.npy
│   │   └── ...
│   └── geotiff/                     # GeoTIFF格式
│       ├── 201501_pred.tif
│       ├── 201502_pred.tif
│       └── ...
├── adapters/                        # Adapter权重（阶段保存）
│   ├── adapter_pretrained.pth       # 初始预训练权重
│   └── adapter_after_finetune.pth   # 有标签期微调后的权重
├── checkpoints/                     # 进度检查点（支持断点续传）
│   ├── checkpoint_201512.pth        # 定期保存的检查点
│   ├── checkpoint_201612.pth
│   └── ...
├── visualizations/                  # 可视化对比图
│   ├── 201501.png                   # 预测 vs 真值 vs 残差
│   ├── 201502.png
│   └── ...
├── residuals/                       # 残差分析
│   ├── 201501_residual.npy          # 残差数值
│   ├── 201501_residual.png          # 残差空间图
│   ├── 201501_distribution.png      # 残差分布图
│   └── ...
└── logs/                            # 日志和配置
    ├── config.json                  # 运行配置
    └── metrics.csv                  # 评估指标
```

### 各目录说明

| 目录 | 内容 | 用途 |
|------|------|------|
| `predictions/npy/` | `.npy` 格式的预测数组 | Python后续处理 |
| `predictions/geotiff/` | `.tif` 格式的地理数据 | GIS软件（QGIS, ArcGIS）可视化 |
| `adapters/` | 不同阶段的Adapter权重 | 断点续传、对比分析 |
| `checkpoints/` | 进度检查点 | 支持断点续传 |
| `visualizations/` | 预测对比图 | 快速查看预测质量 |
| `residuals/` | 残差分析 | 误差空间分布、统计特征 |
| `logs/` | 配置、指标和日志 | 实验记录、结果复现、调试 |

### 进度检查点（断点续传）

#### 自动保存检查点

默认每12个月自动保存一次检查点（可通过 `--checkpoint_freq` 调整）：

```bash
python inference_rolling.py \
    --checkpoint_freq 12 \      # 每12个月保存一次
    --output_dir ./results
```

检查点包含：
- 当前批次索引和日期
- Adapter模型状态
- 优化器状态
- 历史队列（时序信息）
- 已计算的指标
- 阶段状态（是否处于微调期）

#### 从检查点恢复

如果推理中断，可以从任意检查点恢复：

```bash
# 查看可用的检查点
ls ./results/checkpoints/
# checkpoint_202012.pth  checkpoint_202112.pth  checkpoint_202212.pth

# 从特定检查点恢复
python inference_rolling.py \
    --resume ./results/checkpoints/checkpoint_202012.pth \
    --output_dir ./results
```

恢复后会：
1. 加载Adapter和优化器状态
2. 恢复历史队列（用于时序建模）
3. 跳过已处理的批次
4. 继续从断点处推理

**注意**：恢复时其他参数（`--base_model`, `--adapter`, `--dataset_dir` 等）应与原命令一致。

## 核心机制

### 1. Adapter 阶段保存

当启用 `--save_adapter_stages` 时，会在关键时刻保存Adapter权重：

```
[Stage] Starting online finetune at 201501
  ✓ Adapter saved to: adapters/adapter_pretrained.pth
  
...（在线微调中）...

[Stage] Ending online finetune at 202012
  ✓ Finetuned adapter saved to: adapters/adapter_after_finetune.pth
```

**用途：**
- `adapter_pretrained.pth`: 预训练权重（用于纯预测场景）
- `adapter_after_finetune.pth`: 微调后权重（包含有标签期的学习成果）

### 2. 历史窗口维护（策略C：观测优先）

```python
if has_ground_truth:
    current_value = ground_truth      # 用观测值
else:
    current_value = prediction        # 用预测值

history_queue.append(current_residual)
```

### 3. 滚动预测流程

```
时间步1: [2013_obs, 2014_obs, 2015_obs] → 预测 2016 → 在线微调 → 保存
时间步2: [2014_obs, 2015_obs, 2016_pred] → 预测 2017 → 在线微调 → 保存
...
时间步N: [2019_obs, 2020_obs, 2021_pred] → 预测 2022 → 直接推理 → 保存
```

## GeoTIFF 导出

### 方法一：自动地理信息（默认）

如果没有提供参考GeoTIFF，将使用默认的全球范围：
- CRS: EPSG:4326 (WGS84)
- 范围: -180°~180°E, -90°~90°N
- 分辨率: 0.083333° (约 10km)

```bash
python inference_rolling.py \
    --save_geotiff \
    --resolution 0.083333 \
    ...
```

### 方法二：参考已有GeoTIFF（推荐）

复制参考文件的地理信息（坐标系、变换矩阵等）：

```bash
python inference_rolling.py \
    --save_geotiff \
    --reference_geotiff ./data/reference_ndvi.tif \
    ...
```

### 在QGIS中查看

1. 打开 QGIS
2. Layer → Add Layer → Add Raster Layer
3. 选择 `predictions/geotiff/201501_pred.tif`
4. 右键图层 → Properties → Symbology
5. Render type: Singleband pseudocolor
6. Color ramp: RdYlGn（红-黄-绿）
7. Min: -0.1, Max: 0.9

## 参数说明

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_model` | 预训练基础模型路径 | 必需 |
| `--adapter` | 预训练Adapter路径 | 必需 |
| `--adapter_type` | Adapter类型 | DeepMultiTimeAdapter |
| `--window_size` | 历史窗口大小 | 3 |
| `--start_date` | 推理起始日期（YYYYMM） | 必需 |
| `--end_date` | 推理结束日期（YYYYMM） | 必需 |
| `--labeled_end_date` | 有标签数据截止日期 | None |

### 在线微调参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--online_finetune` | 启用在线微调 | True |
| `--finetune_iterations` | 每样本微调迭代数 | 50 |
| `--finetune_lr` | 微调学习率 | 2e-3 |
| `--save_adapter_stages` | 阶段保存Adapter权重 | True |

### 进度保存参数（断点续传）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint_freq` | 每N个月保存检查点（0=禁用） | 12 |
| `--resume` | 从检查点文件恢复 | None |

### 输出控制参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--save_npy` | 保存NumPy格式 | True |
| `--save_geotiff` | 保存GeoTIFF格式 | True |
| `--save_visualizations` | 保存可视化图像 | True |
| `--save_residuals` | 保存残差分析 | True |
| `--num_viz` | 可视化样本数（0=全部） | 10 |
| `--viz_formats` | 可视化格式（逗号分隔） | png |
| `--residual_bin_width` | 残差分布bin宽度 | 0.001 |

### GeoTIFF参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--reference_geotiff` | 参考GeoTIFF文件 | None |
| `--crs` | 坐标参考系统 | EPSG:4326 |
| `--resolution` | 空间分辨率（度） | 0.083333 |

## 可视化输出格式控制

### 多格式输出

通过 `--viz_formats` 参数可以同时输出多种格式的图像：

```bash
python inference_rolling.py \
    --viz_formats png,svg,pdf \
    --output_dir ./results
```

**支持的格式：**
- `png` - 默认格式，适合快速查看
- `jpg/jpeg` - 压缩格式，文件较小
- `svg` - 矢量格式，适合论文插图（可无损缩放）
- `pdf` - 矢量格式，适合打印和发表论文

**输出示例：**
```
visualizations/
├── 201501.png          # PNG格式
├── 201501.svg          # SVG矢量格式
└── 201501.pdf          # PDF格式

residuals/
├── 201501_residual.png
├── 201501_residual.svg
├── 201501_distribution.png
└── 201501_distribution.svg
```

### 残差分布图 Bin 宽度控制

通过 `--residual_bin_width` 参数控制残差分布直方图的精细程度：

```bash
python inference_rolling.py \
    --residual_bin_width 0.001 \
    --output_dir ./results
```

**参数说明：**
- 默认值：`0.001`
- 范围：`-0.2 ~ 0.2`（共400个bin）
- 计算公式：`bin数量 = 0.4 / bin_width`

**使用建议：**
| 数据量 | 建议 bin_width | 说明 |
|--------|---------------|------|
| 小 (< 1000像素) | 0.01 | 避免bin内样本过少 |
| 中 (1000-10000像素) | 0.005 | 平衡精细度和可读性 |
| 大 (> 10000像素) | 0.001 | 默认，展现细节 |

**分布图包含信息：**
- 残差均值 ($\mu_{land}$)
- 残差标准差 ($\sigma_{land}$)
- 使用的 bin_width 值
- 零误差参考线（红色虚线）

---

## 日志记录

推理过程中的所有输出会自动保存到日志文件：

```
logs/
├── config.json          # 运行配置参数
├── metrics.csv          # 评估指标汇总
└── inference.log        # 完整运行日志（时间戳+信息）
```

**日志内容包括：**
- 设备信息（CUDA/CPU）
- 模型加载状态
- 数据处理进度
- 每个样本的 RMSE、R² 指标
- 检查点保存记录
- 错误和警告信息

---

## 支持的 Adapter 类型

| 类型 | 描述 | 适用场景 |
|------|------|---------|
| `FineTuningAdapter` | MLP Adapter | 简单修正 |
| `ResFineTuningAdapter` | 带残差MLP | 深层修正 |
| `ConvResAdapter` | 卷积残差 | 局部空间修正 |
| `TimeSpaceAdapter` | 单步时序 | 利用上月残差 |
| `DeepMultiTimeAdapter` | 多步时序 | **推荐**，利用多历史窗口 |

## 故障排查

### 问题1: "No .npy files found"

检查 `--dataset_dir` 路径是否正确，确保目录结构为：
```
dataset_dir/
├── Evapotranspiration/
├── Precipitation/
├── SOV/
├── Temperature/
├── NDVI_Monthly/
├── slope.npy
├── elevation.npy
└── mask.npy
```

### 问题2: GeoTIFF保存失败

检查是否安装了 `rasterio`：
```bash
pip install rasterio
```

### 问题3: 在线微调后效果变差

尝试减少 `--finetune_lr` 或 `--finetune_iterations`。

### 问题4: 残差分布图太粗糙或太精细

调整 `--residual_bin_width` 参数（默认 0.001）：
```bash
# 更粗的bin（适合数据量小的情况）
--residual_bin_width 0.01

# 更细的bin（默认，适合数据量大的情况）
--residual_bin_width 0.001
```

### 问题5: 需要SVG格式用于论文插图

使用 `--viz_formats` 指定输出格式：
```bash
# 仅SVG
--viz_formats svg

# 多种格式（PNG用于查看，SVG用于论文，PDF用于打印）
--viz_formats png,svg,pdf
```

### 问题4: 显存不足

减少 `--finetune_iterations` 或使用更小的 `--grid_size`。

## 示例完整流程

```bash
# 1. 预训练Adapter
python train_adapter.py \
    --base_model ./checkpoints/unet_best.pth \
    --dataset_dir ./data/AWI/ \
    --output_dir ./checkpoints/adapter_awi \
    --start_date 198201 --end_date 201412 \
    --epochs 50

# 2. 混合推理（有标签期微调 + 无标签期预测）
python inference_rolling.py \
    --base_model ./checkpoints/unet_best.pth \
    --adapter ./checkpoints/adapter_awi/DeepMultiTimeAdapter_best.pth \
    --dataset_dir ./data/AWI/ \
    --start_date 201501 --end_date 205012 \
    --labeled_end_date 202012 \
    --online_finetune \
    --save_adapter_stages \
    --save_geotiff \
    --reference_geotiff ./data/AWI/reference.tif \
    --output_dir ./results/awi_2015_2050

# 3. 查看结果
ls ./results/awi_2015_2050/
#  predictions/npy/          - NumPy预测
#  predictions/geotiff/      - GeoTIFF预测
#  adapters/                 - 阶段权重
#  checkpoints/              - 进度检查点
#  visualizations/           - 对比图 (支持多格式)
#  residuals/                - 误差分析 (含分布图)
#  logs/                     - 日志和指标
#    ├── config.json
#    ├── metrics.csv
#    └── inference.log
```

### 断点续传示例

```bash
# 4. 长时段推理（自动保存检查点）
python inference_rolling.py \
    --base_model ./checkpoints/unet_best.pth \
    --adapter ./checkpoints/adapter_awi/DeepMultiTimeAdapter_best.pth \
    --dataset_dir ./data/AWI/ \
    --start_date 201501 --end_date 210012 \
    --checkpoint_freq 12 \                    # 每年保存一次检查点
    --output_dir ./results/longterm

# 如果中断，查看检查点
ls ./results/longterm/checkpoints/
# checkpoint_202012.pth  checkpoint_203012.pth  ...

# 5. 从检查点恢复（继续推理）
python inference_rolling.py \
    --resume ./results/longterm/checkpoints/checkpoint_203012.pth \
    --base_model ./checkpoints/unet_best.pth \
    --adapter ./checkpoints/adapter_awi/DeepMultiTimeAdapter_best.pth \
    --dataset_dir ./data/AWI/ \
    --checkpoint_freq 12 \
    --output_dir ./results/longterm
# 会自动跳过 201501-203012，从 203101 继续
```
