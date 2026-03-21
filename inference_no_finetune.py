"""
无微调推理脚本 - 直接使用预训练Adapter权重

用法：
    python inference_no_finetune.py \
        --base_model ./checkpoints/AWI_prediction_model.pth \
        --adapter ./checkpoints/adapter_v3/DeepMultiTimeAdapter_best.pth \
        --dataset_dir ./datasets/AWI-CM-1-1-MR/ \
        --start_date 201501 \
        --end_date 202012 \
        --output_dir ./results/no_finetune
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from collections import deque
import warnings
warnings.filterwarnings("ignore")

from models.models import load_model
from models.adapter import DeepMultiTimeAdapter
from utils.helpers import set_random_seeds, get_npy_files, filter_files_by_date, TARGET_SHAPE, CATEGORIES
from utils.datasets import NDVIDataset
from utils.inference import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Inference WITHOUT Online Finetuning')
    
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
    
    # 时间范围
    parser.add_argument('--start_date', type=str, required=True,
                        help='Start date for inference (YYYYMM)')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date for inference (YYYYMM)')
    
    # Adapter参数
    parser.add_argument('--window_size', type=int, default=3,
                        help='History window size')
    parser.add_argument('--grid_size', type=int, default=30,
                        help='Grid size for patch-based processing')
    
    # 输出控制
    parser.add_argument('--output_dir', type=str, default='./results/no_finetune',
                        help='Output directory')
    parser.add_argument('--save_npy', action='store_true', default=True,
                        help='Save predictions as .npy files')
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                        help='Save visualization images')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def load_adapter(checkpoint_path, window_size, device):
    """加载预训练的Adapter（严格只读模式）"""
    adapter = DeepMultiTimeAdapter(history_window=window_size, hidden_channels=64)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        adapter.load_state_dict(checkpoint['model_state_dict'])
    else:
        adapter.load_state_dict(checkpoint)
    
    adapter = adapter.to(device)
    adapter.eval()  # 严格推理模式
    
    # 冻结所有参数，确保不会更新
    for param in adapter.parameters():
        param.requires_grad = False
    
    print(f"✓ Loaded adapter from {checkpoint_path}")
    print(f"  Parameters frozen: {sum(p.numel() for p in adapter.parameters())} params")
    
    return adapter


def inference_with_adapter(adapter, base_output, history, window_size):
    """
    纯推理模式 - 不更新任何权重
    """
    with torch.no_grad():  # 确保不计算梯度
        # 处理历史窗口
        if history is not None and len(history) > 0:
            if len(history) < window_size:
                needed = window_size - len(history)
                zeros = [torch.zeros_like(base_output) for _ in range(needed)]
                combined_history = torch.cat(list(history) + zeros, dim=1)
            else:
                combined_history = torch.cat(list(history)[-window_size:], dim=1)
        else:
            zeros = [torch.zeros_like(base_output) for _ in range(window_size)]
            combined_history = torch.cat(zeros, dim=1)
        
        # Adapter前向传播
        adjusted = adapter(base_output, combined_history)
        adjusted = torch.clamp(adjusted, 0.0, 1.0)
    
    return adjusted


def save_visualization(prediction, target, mask, output_path, title=""):
    """保存可视化结果"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    pred_np = prediction.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy() if target is not None else None
    mask_np = mask.squeeze().cpu().numpy()
    
    if target_np is not None:
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
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(pred_np, cmap='RdYlGn', vmin=-0.1, vmax=0.9)
        ax.set_title('Prediction (No GT Available)')
        plt.colorbar(im, ax=ax, fraction=0.025)
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # 设置随机种子
    set_random_seeds(args.seed)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print("PURE INFERENCE MODE (NO FINETUNING)")
    print(f"{'='*60}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, 'predictions')
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # 加载统计信息
    stats = {}
    if os.path.exists(args.stats_file):
        with open(args.stats_file, 'r') as f:
            stats = json.load(f)
    
    # 加载掩码
    mask_path = args.mask_path or os.path.join(args.dataset_dir, "mask.npy")
    if os.path.exists(mask_path):
        global_mask_np = np.load(mask_path)
        global_mask = torch.from_numpy(global_mask_np).float().to(device).unsqueeze(0).unsqueeze(0)
        print(f"✓ Loaded mask from {mask_path}")
    else:
        global_mask = torch.ones((1, 1, *TARGET_SHAPE), device=device)
    
    # 加载基础模型和Adapter
    print(f"\nLoading models...")
    base_model = load_model(args.base_model, device)
    adapter = load_adapter(args.adapter, args.window_size, device)
    
    # 获取数据文件
    npy_files = get_npy_files(args.dataset_dir)
    print(f"\nFound files: {{k: len(v) for k, v in npy_files.items()}}")
    
    # 按日期过滤
    filtered_files = filter_files_by_date(
        npy_files,
        start_date=args.start_date,
        end_date=args.end_date,
        mode='between'
    )
    print(f"Filtered files ({args.start_date}-{args.end_date}): {sum(len(v) for v in filtered_files.values())} files")
    
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 历史窗口（用于DeepMultiTimeAdapter）
    history_queue = deque(maxlen=args.window_size)
    
    # 结果记录
    results = {'metrics': []}
    
    print(f"\n{'='*60}")
    print(f"Starting inference: {args.start_date} to {args.end_date}")
    print(f"Mode: NO FINETUNING - Using pretrained adapter only")
    print(f"{'='*60}\n")
    
    # 推理循环
    for batch_idx, batch_data in enumerate(dataloader):
        # 解析数据
        if len(batch_data) == 3:
            features, target, file_path = batch_data
            has_gt = True
        elif len(batch_data) == 2:
            if isinstance(batch_data[1], str):
                features, file_path = batch_data
                target = None
                has_gt = False
            else:
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
        
        # 推理
        features = features.to(device)
        if has_gt:
            target = target.to(device).unsqueeze(1) if target.dim() == 3 else target.to(device)
        
        H, W = TARGET_SHAPE
        curr_mask = global_mask[:, :, :H, :W]
        
        with torch.no_grad():
            base_output = base_model(features)
            if base_output.dim() == 3:
                base_output = base_output.unsqueeze(1)
        
        # Adapter推理（无微调）
        history = list(history_queue) if len(history_queue) > 0 else None
        adjusted = inference_with_adapter(adapter, base_output, history, args.window_size)
        adjusted = adjusted * curr_mask
        
        # 更新历史（使用预测值作为下一帧的历史）
        if has_gt:
            current_residual = (adjusted - target).detach()
        else:
            current_residual = torch.zeros_like(adjusted)
        history_queue.append(current_residual)
        
        # 保存预测结果
        pred_np = adjusted.squeeze().cpu().numpy()
        
        if args.save_npy:
            npy_path = os.path.join(pred_dir, f"{date_str}_pred.npy")
            np.save(npy_path, pred_np)
        
        # 如果有标签，计算指标
        if has_gt:
            target_np = target.squeeze().cpu().numpy()
            land_mask = (curr_mask.squeeze().cpu().numpy() > 0.5)
            
            if land_mask.any():
                metrics = calculate_metrics(pred_np.flatten(), target_np.flatten(), land_mask.flatten())
                metrics['date'] = date_str
                results['metrics'].append(metrics)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"[{batch_idx+1}/{len(dataloader)}] {date_str} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        else:
            if (batch_idx + 1) % 10 == 0:
                print(f"[{batch_idx+1}/{len(dataloader)}] {date_str} - (no GT)")
        
        # 保存可视化
        if args.save_visualizations and batch_idx < 20:  # 只保存前20个
            viz_path = os.path.join(viz_dir, f"{date_str}.png")
            save_visualization(adjusted, target if has_gt else None, curr_mask, viz_path, date_str)
    
    # 保存汇总结果
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Results saved to: {args.output_dir}")
    
    if results['metrics']:
        import pandas as pd
        df = pd.DataFrame(results['metrics'])
        metrics_path = os.path.join(args.output_dir, 'metrics.csv')
        df.to_csv(metrics_path, index=False)
        
        print(f"\nOverall Metrics:")
        print(f"  Mean RMSE: {df['rmse'].mean():.6f}")
        print(f"  Mean MAE:  {df['mae'].mean():.6f}")
        print(f"  Mean R²:   {df['r2'].mean():.6f}")
        print(f"\nMetrics saved to: {metrics_path}")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
