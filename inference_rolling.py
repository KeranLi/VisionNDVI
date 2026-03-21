"""
滚动预测推理脚本（支持在线微调和预训练Adapter）

功能：
1. 加载预训练的基础模型和Adapter
2. 在[有标签期间]：在线微调Adapter（使用GT），阶段保存权重
3. 在[无标签期间]：直接使用预训练Adapter（无微调）
4. 滚动预测：维护历史窗口，观测优先策略
5. 规范化输出：predictions/ adapters/ visualizations/ residuals/
6. 支持GeoTIFF导出

用法示例：
    # 场景1：验证集（2015-2020有标签）
    python inference_rolling.py \
        --base_model ./checkpoints/best_model.pth \
        --adapter ./checkpoints/DeepMultiTimeAdapter_best.pth \
        --dataset_dir ./datasets/AWI-CM-1-1-MR/ \
        --start_date 201501 \
        --end_date 202012 \
        --online_finetune \
        --output_dir ./results/validation
    
    # 场景2：纯预测（2021-2050无标签）
    python inference_rolling.py \
        --base_model ./checkpoints/best_model.pth \
        --adapter ./checkpoints/DeepMultiTimeAdapter_best.pth \
        --dataset_dir ./datasets/AWI-CM-1-1-MR/ \
        --start_date 202101 \
        --end_date 205012 \
        --no_online_finetune \
        --output_dir ./results/prediction
    
    # 场景3：混合（2015-2020有标签，2021+无标签）
    python inference_rolling.py \
        --base_model ./checkpoints/best_model.pth \
        --adapter ./checkpoints/DeepMultiTimeAdapter_best.pth \
        --dataset_dir ./datasets/AWI-CM-1-1-MR/ \
        --start_date 201501 \
        --end_date 205012 \
        --labeled_end_date 202012 \
        --online_finetune \
        --save_adapter_stages \
        --output_dir ./results/mixed
"""

import os
import argparse
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# GeoTIFF support
import rasterio
from rasterio.transform import from_origin

from models.models import load_model
from models.adapter import (
    FineTuningAdapter, 
    ResFineTuningAdapter, 
    ConvResAdapter, 
    TimeSpaceAdapter, 
    DeepMultiTimeAdapter
)
from utils.helpers import (
    set_random_seeds, 
    get_npy_files, 
    filter_files_by_date,
    TARGET_SHAPE,
    CATEGORIES
)
from utils.datasets import NDVIDataset
from utils.inference import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Rolling Inference with Adapter')
    
    # 模型路径
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to pretrained base model')
    parser.add_argument('--adapter', type=str, required=True,
                        help='Path to pretrained adapter checkpoint')
    
    # 数据路径
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing dataset')
    parser.add_argument('--stats_file', type=str, default='training_stats.json',
                        help='Path to training statistics JSON')
    parser.add_argument('--mask_path', type=str, default=None,
                        help='Path to mask file (default: dataset_dir/mask.npy)')
    
    # GeoTIFF参考（用于获取地理信息）
    parser.add_argument('--reference_geotiff', type=str, default=None,
                        help='Reference GeoTIFF file to copy geo metadata from')
    parser.add_argument('--crs', type=str, default='EPSG:4326',
                        help='Coordinate reference system (default: EPSG:4326)')
    parser.add_argument('--resolution', type=float, default=0.083333,
                        help='Spatial resolution in degrees (default: 0.083333 = 1/12 degree)')
    
    # 时间范围
    parser.add_argument('--start_date', type=str, required=True,
                        help='Start date for inference (YYYYMM)')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date for inference (YYYYMM)')
    parser.add_argument('--labeled_end_date', type=str, default=None,
                        help='Last date with labels (YYYYMM). After this, no online finetuning.')
    
    # 推理模式
    parser.add_argument('--online_finetune', action='store_true', default=True,
                        help='Enable online finetuning during labeled period')
    parser.add_argument('--no_online_finetune', dest='online_finetune', action='store_false',
                        help='Disable online finetuning')
    parser.add_argument('--finetune_iterations', type=int, default=50,
                        help='Number of iterations for online finetuning per sample')
    parser.add_argument('--finetune_lr', type=float, default=2e-3,
                        help='Learning rate for online finetuning')
    
    # Adapter阶段保存
    parser.add_argument('--save_adapter_stages', action='store_true', default=True,
                        help='Save adapter weights at stage transitions')
    parser.add_argument('--no_save_adapter_stages', dest='save_adapter_stages', action='store_false',
                        help='Do not save adapter stages')
    
    # Adapter参数
    parser.add_argument('--adapter_type', type=str, default='DeepMultiTimeAdapter',
                        choices=['FineTuningAdapter', 'ResFineTuningAdapter', 
                                'ConvResAdapter', 'TimeSpaceAdapter', 'DeepMultiTimeAdapter'])
    parser.add_argument('--window_size', type=int, default=3,
                        help='History window size')
    parser.add_argument('--grid_size', type=int, default=30,
                        help='Grid size for patch-based processing')
    
    # 输出控制
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory')
    parser.add_argument('--save_npy', action='store_true', default=True,
                        help='Save predictions as .npy files')
    parser.add_argument('--save_geotiff', action='store_true', default=True,
                        help='Save predictions as GeoTIFF files')
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--save_residuals', action='store_true', default=True,
                        help='Save residual maps and distributions')
    parser.add_argument('--num_viz', type=int, default=10,
                        help='Number of samples to visualize (0 = all)')
    parser.add_argument('--viz_formats', type=str, default='png',
                        help='Visualization format(s), comma-separated (e.g., png,svg,pdf,jpg). Default: png')
    parser.add_argument('--residual_bin_width', type=float, default=0.001,
                        help='Bin width for residual distribution histogram. Default: 0.001')
    
    # 进度保存
    parser.add_argument('--checkpoint_freq', type=int, default=12,
                        help='Save checkpoint every N months (0 = disable)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (must be 1 for sequential processing)')
    
    return parser.parse_args()


def create_output_directories(output_dir):
    """创建规范化的输出目录结构"""
    dirs = {
        'predictions': os.path.join(output_dir, 'predictions'),
        'predictions_npy': os.path.join(output_dir, 'predictions', 'npy'),
        'predictions_geotiff': os.path.join(output_dir, 'predictions', 'geotiff'),
        'adapters': os.path.join(output_dir, 'adapters'),
        'visualizations': os.path.join(output_dir, 'visualizations'),
        'residuals': os.path.join(output_dir, 'residuals'),
        'checkpoints': os.path.join(output_dir, 'checkpoints'),
        'logs': os.path.join(output_dir, 'logs'),
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    return dirs


def save_adapter_checkpoint(adapter, optimizer, epoch, output_path, extra_info=None):
    """保存Adapter检查点"""
    checkpoint = {
        'model_state_dict': adapter.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'adapter_type': adapter.__class__.__name__,
    }
    if extra_info:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, output_path)
    print(f"  ✓ Adapter saved to: {output_path}")


def save_progress_checkpoint(output_path, batch_idx, date_str, adapter, optimizer, 
                             history_queue, results, finetune_active, finetune_ended, args):
    """
    保存进度检查点（支持断点续传）
    
    Args:
        output_path: 检查点保存路径
        batch_idx: 当前批次索引
        date_str: 当前日期字符串
        adapter: Adapter模型
        optimizer: 优化器
        history_queue: 历史队列
        results: 结果字典
        finetune_active: 是否处于微调阶段
        finetune_ended: 微调是否已结束
        args: 命令行参数
    """
    checkpoint = {
        # 进度信息
        'batch_idx': batch_idx,
        'date_str': date_str,
        'timestamp': pd.Timestamp.now().isoformat(),
        
        # 模型状态
        'adapter_state_dict': adapter.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        
        # 时序状态
        'history_queue': [r.cpu().numpy() for r in history_queue] if history_queue else [],
        
        # 结果
        'results': results,
        
        # 阶段状态
        'finetune_active': finetune_active,
        'finetune_ended': finetune_ended,
        
        # 配置
        'args': vars(args)
    }
    
    torch.save(checkpoint, output_path)
    return output_path


def load_progress_checkpoint(checkpoint_path, adapter, optimizer, device):
    """
    加载进度检查点
    
    Returns:
        checkpoint: 检查点字典
        history_queue: 恢复后的历史队列
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 恢复Adapter状态
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    
    # 恢复优化器状态
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 恢复历史队列
    history_queue = deque(maxlen=checkpoint['args'].get('window_size', 3))
    for residual_np in checkpoint.get('history_queue', []):
        history_queue.append(torch.from_numpy(residual_np).to(device))
    
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    print(f"  Resume from: {checkpoint['date_str']} (batch {checkpoint['batch_idx']})")
    print(f"  History queue size: {len(history_queue)}")
    
    return checkpoint, history_queue


def load_adapter(checkpoint_path, adapter_type, window_size, grid_size, device):
    """加载预训练的Adapter"""
    if adapter_type == 'FineTuningAdapter':
        adapter = FineTuningAdapter(input_size=grid_size * grid_size)
    elif adapter_type == 'ResFineTuningAdapter':
        adapter = ResFineTuningAdapter(input_size=grid_size * grid_size)
    elif adapter_type == 'ConvResAdapter':
        adapter = ConvResAdapter(in_channels=1, hidden_dim=32)
    elif adapter_type == 'TimeSpaceAdapter':
        adapter = TimeSpaceAdapter(in_channels=2, hidden_channels=64)
    elif adapter_type == 'DeepMultiTimeAdapter':
        adapter = DeepMultiTimeAdapter(history_window=window_size, hidden_channels=64)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        adapter.load_state_dict(checkpoint['model_state_dict'])
    else:
        adapter.load_state_dict(checkpoint)
    
    adapter = adapter.to(device)
    adapter.eval()
    
    print(f"Loaded adapter from {checkpoint_path}")
    return adapter, checkpoint


def extract_patches(tensor, grid_size, stride=None):
    """提取patches"""
    if stride is None:
        stride = grid_size // 2
    
    B, C, H, W = tensor.shape
    patches = []
    coords = []
    
    for b in range(B):
        for y in range(0, H - grid_size + 1, stride):
            for x in range(0, W - grid_size + 1, stride):
                patch = tensor[b:b+1, :, y:y+grid_size, x:x+grid_size]
                patches.append(patch)
                coords.append((b, y, x))
    
    return torch.cat(patches, dim=0), coords


def online_finetune_adapter(adapter, base_output, target, mask, optimizer, 
                            num_iterations=50, grid_size=30):
    """在线微调Adapter"""
    adapter.train()
    
    stride = grid_size // 2
    base_patches, coords = extract_patches(base_output, grid_size, stride)
    target_patches, _ = extract_patches(target, grid_size, stride)
    mask_patches, _ = extract_patches(mask, grid_size, stride)
    
    num_patches = base_patches.size(0)
    
    for iteration in range(num_iterations):
        indices = torch.randperm(num_patches)
        batch_size = 128
        
        for start_idx in range(0, num_patches, batch_size):
            end_idx = min(start_idx + batch_size, num_patches)
            batch_indices = indices[start_idx:end_idx]
            
            b_base = base_patches[batch_indices]
            b_target = target_patches[batch_indices]
            b_mask = mask_patches[batch_indices]
            
            optimizer.zero_grad()
            adjusted_patches = adapter(b_base)
            
            mask_bool = (b_mask > 0.5)
            if mask_bool.sum() > 0:
                loss_mse = F.mse_loss(adjusted_patches[mask_bool], b_target[mask_bool])
                loss_l1 = F.l1_loss(adjusted_patches[mask_bool], b_target[mask_bool])
                penalty_under = torch.mean(torch.relu(-adjusted_patches[mask_bool]) ** 2)
                penalty_over = torch.mean(torch.relu(adjusted_patches[mask_bool] - 1.0) ** 2)
                
                total_loss = 10.0 * loss_mse + 2.0 * loss_l1 + 10.0 * (penalty_under + penalty_over)
                total_loss.backward()
                optimizer.step()
    
    # 推理模式
    adapter.eval()
    with torch.no_grad():
        B, C, H, W = base_output.shape
        adjusted_output = torch.zeros_like(base_output)
        weight_map = torch.zeros_like(base_output)
        
        fusion_weight = torch.ones((1, 1, grid_size, grid_size), device=base_output.device)
        for i in range(grid_size):
            dist = min(i, grid_size - 1 - i) / (grid_size // 2)
            fusion_weight[:, :, i, :] *= dist
            fusion_weight[:, :, :, i] *= dist
        fusion_weight = torch.clamp(fusion_weight, min=0.1)
        
        refined_patches = []
        for i in range(0, num_patches, 128):
            batch = base_patches[i:i+128]
            refined = adapter(batch)
            refined_patches.append(refined)
        refined_patches = torch.cat(refined_patches, dim=0)
        
        for i, (b, y, x) in enumerate(coords):
            adjusted_output[b:b+1, :, y:y+grid_size, x:x+grid_size] += refined_patches[i:i+1] * fusion_weight
            weight_map[b:b+1, :, y:y+grid_size, x:x+grid_size] += fusion_weight
        
        adjusted_output = torch.where(weight_map > 0, adjusted_output / weight_map, base_output)
        adjusted_output = torch.clamp(adjusted_output, 0.0, 1.0)
    
    return adjusted_output


def inference_with_adapter(adapter, base_output, history=None, grid_size=30):
    """使用Adapter进行推理（无在线微调）"""
    adapter.eval()
    
    with torch.no_grad():
        if isinstance(adapter, (TimeSpaceAdapter, DeepMultiTimeAdapter)):
            if isinstance(adapter, TimeSpaceAdapter):
                if history is not None and len(history) > 0:
                    last_residual = history[-1]
                    if last_residual.shape != base_output.shape:
                        last_residual = F.interpolate(
                            last_residual, 
                            size=base_output.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                else:
                    last_residual = torch.zeros_like(base_output)
                adjusted = adapter(base_output, last_residual)
            else:
                window_size = adapter.history_window
                if history is not None and len(history) > 0:
                    # 如果历史长度不足 window_size，用零填充
                    if len(history) < window_size:
                        needed = window_size - len(history)
                        zeros = [torch.zeros_like(base_output) for _ in range(needed)]
                        combined_history = torch.cat(list(history) + zeros, dim=1)
                    else:
                        # 如果历史长度超过 window_size，只取最近的 window_size 个
                        combined_history = torch.cat(list(history)[-window_size:], dim=1)
                else:
                    zeros = [torch.zeros_like(base_output) for _ in range(window_size)]
                    combined_history = torch.cat(zeros, dim=1)
                adjusted = adapter(base_output, combined_history)
        else:
            stride = grid_size // 2
            patches, coords = extract_patches(base_output, grid_size, stride)
            
            refined_patches = []
            for i in range(0, len(patches), 128):
                batch = patches[i:i+128]
                refined = adapter(batch)
                refined_patches.append(refined)
            refined_patches = torch.cat(refined_patches, dim=0)
            
            B, C, H, W = base_output.shape
            adjusted = torch.zeros_like(base_output)
            weight_map = torch.zeros_like(base_output)
            
            fusion_weight = torch.ones((1, 1, grid_size, grid_size), device=base_output.device)
            for i in range(grid_size):
                dist = min(i, grid_size - 1 - i) / (grid_size // 2)
                fusion_weight[:, :, i, :] *= dist
                fusion_weight[:, :, :, i] *= dist
            fusion_weight = torch.clamp(fusion_weight, min=0.1)
            
            for i, (b, y, x) in enumerate(coords):
                adjusted[b:b+1, :, y:y+grid_size, x:x+grid_size] += refined_patches[i:i+1] * fusion_weight
                weight_map[b:b+1, :, y:y+grid_size, x:x+grid_size] += fusion_weight
            
            adjusted = torch.where(weight_map > 0, adjusted / weight_map, base_output)
        
        adjusted = torch.clamp(adjusted, 0.0, 1.0)
    
    return adjusted


def save_as_geotiff(data, output_path, reference_file=None, crs='EPSG:4326', resolution=0.083333):
    """
    保存numpy数组为GeoTIFF
    
    Args:
        data: 2D numpy array
        output_path: 输出文件路径
        reference_file: 参考GeoTIFF文件（复制地理信息）
        crs: 坐标参考系统
        resolution: 空间分辨率（度）
    """
    height, width = data.shape
    
    if reference_file and os.path.exists(reference_file):
        # 从参考文件复制地理信息
        with rasterio.open(reference_file) as src:
            transform = src.transform
            crs = src.crs
            profile = src.profile
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw',
                nodata=-9999
            )
    else:
        # 创建默认地理信息（全球范围）
        # 假设数据覆盖 -180~180°E, -90~90°N
        left = -180
        top = 90
        transform = from_origin(left, top, resolution, resolution)
        
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.float32,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
            'nodata': -9999
        }
    
    # 写入文件
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)
    
    return output_path


def save_residual_map(prediction, ground_truth, output_dir, base_name, formats=None):
    """
    保存残差图
    
    Args:
        prediction: 预测结果
        ground_truth: 真实值
        output_dir: 输出目录
        base_name: 基础文件名
        formats: 格式列表，如 ['png', 'svg', 'pdf']，默认为 ['png']
    """
    if formats is None:
        formats = ['png']
    
    pred = prediction.squeeze()
    gt = ground_truth.squeeze()
    residual = pred - gt
    
    # 保存数值
    res_path = os.path.join(output_dir, f"{base_name}_residual.npy")
    np.save(res_path, residual)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    im = plt.imshow(residual, cmap='coolwarm', vmin=-0.1, vmax=0.1)
    plt.colorbar(im, fraction=0.025, pad=0.04)
    plt.title(f"Residual Map: {base_name}")
    plt.axis('off')
    
    # 为每种格式保存文件
    for fmt in formats:
        fmt = fmt.strip().lower()
        if fmt in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
            viz_path = os.path.join(output_dir, f"{base_name}_residual.{fmt}")
            if fmt == 'jpg':
                fmt = 'jpeg'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight', format=fmt)
    
    plt.close()


def save_residual_distribution(prediction, ground_truth, mask, output_dir, base_name, bin_width=0.001, formats=None):
    """
    保存残差分布
    
    Args:
        prediction: 预测结果
        ground_truth: 真实值
        mask: 掩码
        output_dir: 输出目录
        base_name: 基础文件名
        bin_width: bin宽度，默认0.001
        formats: 格式列表，如 ['png', 'svg', 'pdf']，默认为 ['png']
    """
    if formats is None:
        formats = ['png']
    
    residual_full = (prediction - ground_truth).flatten()
    land_mask = mask.flatten()
    
    land_residuals = residual_full[land_mask == 1]
    
    if len(land_residuals) == 0:
        return
    
    mu = np.mean(land_residuals)
    sigma = np.std(land_residuals)
    
    # 根据bin宽度和数据范围计算bin数量
    range_min, range_max = -0.2, 0.2
    num_bins = int((range_max - range_min) / bin_width)
    num_bins = max(num_bins, 10)  # 至少10个bin
    
    plt.figure(figsize=(10, 6))
    plt.hist(land_residuals, bins=num_bins, color='teal', edgecolor='white', alpha=0.7, density=True)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    
    stats_text = f'$\\mu_{{land}} = {mu:.6f}$\n$\\sigma_{{land}} = {sigma:.6f}$\nbin_width = {bin_width}'
    plt.gca().text(0.95, 0.90, stats_text, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.title(f'Land-Only Residual Distribution: {base_name}')
    plt.xlabel('Residual (Pred - GT)')
    plt.ylabel('Density')
    plt.xlim(range_min, range_max)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # 为每种格式保存文件
    for fmt in formats:
        fmt = fmt.strip().lower()
        if fmt in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
            dist_path = os.path.join(output_dir, f"{base_name}_distribution.{fmt}")
            if fmt == 'jpg':
                fmt = 'jpeg'
            plt.savefig(dist_path, dpi=150, bbox_inches='tight', format=fmt)
    
    plt.close()


def visualize_prediction(prediction, target, mask, output_path, title="", formats=None):
    """
    可视化预测结果
    
    Args:
        prediction: 预测结果
        target: 真实值
        mask: 掩码
        output_path: 输出路径（不含扩展名）
        title: 图表标题
        formats: 格式列表，如 ['png', 'svg', 'pdf']，默认为 ['png']
    """
    if formats is None:
        formats = ['png']
    
    # 确保所有tensor都在CPU上
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu()
    if isinstance(target, torch.Tensor):
        target = target.cpu()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu()
    
    pred_np = prediction.squeeze().numpy() if isinstance(prediction, torch.Tensor) else prediction.squeeze()
    target_np = target.squeeze().numpy() if isinstance(target, torch.Tensor) else target.squeeze()
    mask_np = mask.squeeze().numpy() if isinstance(mask, torch.Tensor) else mask.squeeze()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin, vmax = -0.1, 0.9
    
    im0 = axes[0].imshow(pred_np, cmap='RdYlGn', vmin=vmin, vmax=vmax)
    axes[0].set_title('Prediction')
    plt.colorbar(im0, ax=axes[0], fraction=0.025)
    
    im1 = axes[1].imshow(target_np, cmap='RdYlGn', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground Truth')
    plt.colorbar(im1, ax=axes[1], fraction=0.025)
    
    residual = pred_np - target_np
    im2 = axes[2].imshow(residual, cmap='coolwarm', vmin=-0.1, vmax=0.1)
    axes[2].set_title('Residual (Pred - GT)')
    plt.colorbar(im2, ax=axes[2], fraction=0.025)
    
    for ax in axes:
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # 为每种格式保存文件
    base_path = output_path.replace('.png', '').replace('.jpg', '').replace('.svg', '').replace('.pdf', '')
    for fmt in formats:
        fmt = fmt.strip().lower()
        if fmt in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
            save_path = f"{base_path}.{fmt}"
            if fmt == 'jpg':
                fmt = 'jpeg'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', format=fmt)
    
    plt.close()


def main():
    args = parse_args()
    
    # 设置随机种子
    set_random_seeds(args.seed)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    dirs = create_output_directories(args.output_dir)
    print(f"Output directories created:")
    for name, path in dirs.items():
        print(f"  {name}: {path}")
    
    # 设置日志记录（同时输出到控制台和文件）
    import logging
    log_file = os.path.join(dirs['logs'], 'inference.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    
    # 加载统计信息
    stats = {}
    if os.path.exists(args.stats_file):
        with open(args.stats_file, 'r') as f:
            stats = json.load(f)
        print(f"Loaded stats from {args.stats_file}")
    
    # 加载掩码
    mask_path = args.mask_path or os.path.join(args.dataset_dir, "mask.npy")
    if os.path.exists(mask_path):
        global_mask_np = np.load(mask_path)
        global_mask = torch.from_numpy(global_mask_np).float().to(device).unsqueeze(0).unsqueeze(0)
        print(f"Loaded mask from {mask_path}")
    else:
        print(f"Warning: Mask not found, using all ones")
        global_mask = torch.ones((1, 1, *TARGET_SHAPE), device=device)
    
    # 加载模型
    print(f"\nLoading base model from {args.base_model}")
    base_model = load_model(args.base_model, device)
    
    print(f"Loading adapter from {args.adapter}")
    adapter, adapter_ckpt = load_adapter(
        args.adapter, 
        args.adapter_type, 
        args.window_size, 
        args.grid_size, 
        device
    )
    
    # 保存初始Adapter（预训练权重）
    if args.save_adapter_stages:
        initial_adapter_path = os.path.join(dirs['adapters'], 'adapter_pretrained.pth')
        save_adapter_checkpoint(
            adapter, None, 0, initial_adapter_path,
            extra_info={'stage': 'pretrained', 'date': args.start_date}
        )
    
    # 获取数据文件
    npy_files = get_npy_files(args.dataset_dir)
    print(f"\nFound files: { {k: len(v) for k, v in npy_files.items()} }")
    
    # 按日期过滤
    filtered_files = filter_files_by_date(
        npy_files,
        start_date=args.start_date,
        end_date=args.end_date,
        mode='between'
    )
    print(f"Filtered files ({args.start_date}-{args.end_date}): { {k: len(v) for k, v in filtered_files.items()} }")
    
    # 准备数据路径
    feature_files = {cat: filtered_files.get(cat, []) for cat in CATEGORIES}
    label_files = {'NDVI_Monthly': filtered_files.get('NDVI_Monthly', [])}
    slope_path = os.path.join(args.dataset_dir, "slope.npy")
    elevation_path = os.path.join(args.dataset_dir, "elevation.npy")
    
    # 创建数据集
    has_labels = len(label_files['NDVI_Monthly']) > 0
    dataset = NDVIDataset(
        feature_files,
        label_files,
        slope_path,
        elevation_path,
        mask_path,
        stats,
        mode='train' if has_labels else 'inference'
    )
    print(f"Dataset size: {len(dataset)}")
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 判断在线微调的截止日期
    labeled_end_date = args.labeled_end_date
    if labeled_end_date is None and args.online_finetune:
        labeled_end_date = args.end_date
    
    # 历史窗口
    history_queue = deque(maxlen=args.window_size)
    
    # 在线微调的优化器
    finetune_optimizer = torch.optim.Adam(adapter.parameters(), lr=args.finetune_lr) if args.online_finetune else None
    
    # 断点续传支持
    start_batch_idx = 0
    resume_date = None
    
    if args.resume and os.path.exists(args.resume):
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {args.resume}")
        print(f"{'='*60}")
        checkpoint, history_queue = load_progress_checkpoint(
            args.resume, adapter, finetune_optimizer, device
        )
        start_batch_idx = checkpoint['batch_idx'] + 1
        resume_date = checkpoint['date_str']
        results = checkpoint.get('results', {'metrics': []})
        finetune_active = checkpoint.get('finetune_active', False)
        finetune_ended = checkpoint.get('finetune_ended', False)
        print(f"Will skip first {start_batch_idx} batches (up to {resume_date})")
    else:
        # 结果记录
        results = {
            'metrics': []
        }
        
        # 跟踪状态
        finetune_active = False
        finetune_ended = False
    
    print(f"\n{'='*60}")
    print(f"Starting rolling inference")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Online finetune: {args.online_finetune}")
    if args.online_finetune:
        print(f"  Finetune until: {labeled_end_date}")
    if args.checkpoint_freq > 0:
        print(f"  Checkpoint every: {args.checkpoint_freq} months")
    print(f"{'='*60}\n")
    
    pbar = tqdm(dataloader, desc="Inference", initial=start_batch_idx, total=len(dataloader))
    
    for batch_idx, batch_data in enumerate(pbar):
        # 跳过已处理的批次（断点续传）
        if batch_idx < start_batch_idx:
            continue
        
        # 获取数据
        # 处理不同情况：
        # - 有标签时: (features, target) 或 (features, target, file_path)
        # - 无标签时: (features, file_path)
        if len(batch_data) == 3:
            features, target, file_path = batch_data
            has_gt = True
        elif len(batch_data) == 2:
            # 判断第二个元素是tensor还是str
            if isinstance(batch_data[1], str):
                # (features, file_path) - 无标签
                features, file_path = batch_data
                target = None
                has_gt = False
            else:
                # (features, target) - 有标签但无file_path
                features, target = batch_data
                has_gt = True
                file_path = None
        else:
            raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
        
        file_path = file_path[0] if isinstance(file_path, (list, tuple)) else file_path
        
        # 提取日期
        try:
            date_str = os.path.basename(file_path).replace('.npy', '').replace('NDVI_Processed_', '')
        except:
            date_str = f"sample_{batch_idx:04d}"
        
        # 判断是否在线微调
        should_finetune = args.online_finetune and (labeled_end_date is None or date_str <= labeled_end_date)
        
        # 跟踪状态变化
        if should_finetune and not finetune_active:
            finetune_active = True
            print(f"\n[Stage] Starting online finetune at {date_str}")
        
        if finetune_active and not should_finetune and not finetune_ended:
            finetune_ended = True
            print(f"\n[Stage] Ending online finetune at {date_str}")
            
            # 阶段保存：有标签期结束后的Adapter
            if args.save_adapter_stages:
                finetuned_adapter_path = os.path.join(dirs['adapters'], 'adapter_after_finetune.pth')
                save_adapter_checkpoint(
                    adapter, finetune_optimizer, batch_idx, finetuned_adapter_path,
                    extra_info={
                        'stage': 'after_finetune',
                        'start_date': args.start_date,
                        'end_date': date_str,
                        'last_labeled_date': labeled_end_date
                    }
                )
                print(f"  Finetuned adapter saved (trained up to {labeled_end_date})")
        
        features = features.to(device)
        if has_gt:
            target = target.to(device).unsqueeze(1) if target.dim() == 3 else target.to(device)
        
        H, W = TARGET_SHAPE
        curr_mask = global_mask[:, :, :H, :W]
        
        with torch.no_grad():
            base_output = base_model(features)
            if base_output.dim() == 3:
                base_output = base_output.unsqueeze(1)
        
        # 准备历史
        history = list(history_queue) if len(history_queue) > 0 else None
        
        # Adapter推理
        if should_finetune and has_gt:
            adjusted = online_finetune_adapter(
                adapter, base_output, target, curr_mask,
                finetune_optimizer, args.finetune_iterations, args.grid_size
            )
            mode_str = "finetune"
        else:
            adjusted = inference_with_adapter(adapter, base_output, history, args.grid_size)
            mode_str = "inference"
        
        adjusted = adjusted * curr_mask
        
        # 更新历史队列（策略C：观测优先）
        if has_gt:
            current_value = target
        else:
            current_value = adjusted.detach()
        
        if has_gt:
            current_residual = (adjusted - target).detach()
        else:
            current_residual = torch.zeros_like(adjusted)
        
        history_queue.append(current_residual)
        
        # 保存预测结果
        pred_np = adjusted.squeeze().cpu().numpy()
        
        # 1. 保存为 .npy
        if args.save_npy:
            npy_path = os.path.join(dirs['predictions_npy'], f"{date_str}_pred.npy")
            np.save(npy_path, pred_np)
        
        # 2. 保存为 GeoTIFF
        if args.save_geotiff:
            geotiff_path = os.path.join(dirs['predictions_geotiff'], f"{date_str}_pred.tif")
            try:
                save_as_geotiff(
                    pred_np, geotiff_path, 
                    reference_file=args.reference_geotiff,
                    crs=args.crs,
                    resolution=args.resolution
                )
            except Exception as e:
                print(f"Warning: Failed to save GeoTIFF for {date_str}: {e}")
        
        # 如果有标签，计算指标并保存分析
        if has_gt:
            target_np = target.squeeze().cpu().numpy()
            
            # 计算指标
            land_mask = (curr_mask.squeeze().cpu().numpy() > 0.5)
            if land_mask.any():
                metrics = calculate_metrics(pred_np.flatten(), target_np.flatten(), land_mask.flatten())
                metrics['date'] = date_str
                metrics['mode'] = mode_str
                results['metrics'].append(metrics)
                pbar.set_postfix({
                    'RMSE': f"{metrics['rmse']:.4f}", 
                    'R2': f"{metrics['r2']:.4f}", 
                    'mode': mode_str
                })
            
            # 保存可视化
            if args.save_visualizations:
                should_viz = (args.num_viz == 0) or (batch_idx < args.num_viz)
                if should_viz:
                    viz_path = os.path.join(dirs['visualizations'], f"{date_str}.png")
                    viz_formats = [f.strip() for f in args.viz_formats.split(',')]
                    visualize_prediction(adjusted, target, curr_mask, viz_path, f"{date_str} ({mode_str})", formats=viz_formats)
            
            # 保存残差分析
            if args.save_residuals:
                viz_formats = [f.strip() for f in args.viz_formats.split(',')]
                save_residual_map(
                    pred_np, 
                    target_np * curr_mask.squeeze().cpu().numpy(), 
                    dirs['residuals'], 
                    date_str,
                    formats=viz_formats
                )
                save_residual_distribution(
                    pred_np, 
                    target_np * curr_mask.squeeze().cpu().numpy(),
                    land_mask.astype(np.float32),
                    dirs['residuals'], 
                    date_str,
                    bin_width=args.residual_bin_width,
                    formats=viz_formats
                )
        
        # 定期保存进度检查点
        if args.checkpoint_freq > 0 and (batch_idx + 1) % args.checkpoint_freq == 0:
            ckpt_path = os.path.join(dirs['checkpoints'], f'checkpoint_{date_str}.pth')
            save_progress_checkpoint(
                ckpt_path, batch_idx, date_str, adapter, finetune_optimizer,
                history_queue, results, finetune_active, finetune_ended, args
            )
            print(f"\n  ✓ Progress checkpoint saved: {ckpt_path}")
    
    # 保存最终Adapter（如果一直处于微调模式）
    if args.save_adapter_stages and finetune_active and not finetune_ended:
        final_adapter_path = os.path.join(dirs['adapters'], 'adapter_after_finetune.pth')
        save_adapter_checkpoint(
            adapter, finetune_optimizer, len(dataloader), final_adapter_path,
            extra_info={
                'stage': 'after_finetune',
                'start_date': args.start_date,
                'end_date': args.end_date,
                'note': 'Full period was labeled'
            }
        )
    
    # 保存汇总结果
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Results saved to: {args.output_dir}")
    
    # 保存指标
    if results['metrics']:
        df = pd.DataFrame(results['metrics'])
        metrics_path = os.path.join(dirs['logs'], 'metrics.csv')
        df.to_csv(metrics_path, index=False)
        
        # 按阶段统计
        print(f"\nMetrics Summary:")
        for mode in ['finetune', 'inference']:
            mode_df = df[df['mode'] == mode]
            if len(mode_df) > 0:
                print(f"\n  [{mode.upper()}] ({len(mode_df)} samples)")
                print(f"    Mean RMSE: {mode_df['rmse'].mean():.6f}")
                print(f"    Mean MAE:  {mode_df['mae'].mean():.6f}")
                print(f"    Mean R²:   {mode_df['r2'].mean():.6f}")
        
        print(f"\nOverall metrics saved to: {metrics_path}")
    
    # 保存配置
    config_path = os.path.join(dirs['logs'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Config saved to: {config_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
