"""
Adapter预训练脚本 - 完全模仿 inference.py 的 patches 处理方式

关键：Adapter 接收 30x30 patches，不是 full image！
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from collections import deque

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
from utils.inference import calculate_metrics
from utils.datasets import NDVIDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Adapter - Patch-based like inference.py')
    
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='./datasets/AWI-CM-1-1-MR/')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--stats_file', type=str, default='training_stats.json')
    
    parser.add_argument('--start_date', type=str, default='198201')
    parser.add_argument('--end_date', type=str, default='201412')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    
    parser.add_argument('--adapter_type', type=str, default='DeepMultiTimeAdapter',
                        choices=['FineTuningAdapter', 'ResFineTuningAdapter', 
                                'ConvResAdapter', 'TimeSpaceAdapter', 'DeepMultiTimeAdapter'])
    
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--grid_size', type=int, default=30)
    
    # 与 inference.py 一致
    parser.add_argument('--iterations_per_sample', type=int, default=50)
    parser.add_argument('--inner_batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--epochs', type=int, default=5)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 可视化
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Enable visualization during training')
    parser.add_argument('--viz_dir', type=str, default=None,
                        help='Visualization output directory (default: output_dir/visualizations)')
    parser.add_argument('--viz_freq', type=int, default=10,
                        help='Visualize every N batches (default: 10)')
    parser.add_argument('--num_viz_samples', type=int, default=5,
                        help='Number of samples to visualize per epoch (default: 5)')
    
    return parser.parse_args()


def create_adapter(adapter_type, window_size, grid_size):
    if adapter_type == 'FineTuningAdapter':
        return FineTuningAdapter(input_size=grid_size * grid_size)
    elif adapter_type == 'ResFineTuningAdapter':
        return ResFineTuningAdapter(input_size=grid_size * grid_size)
    elif adapter_type == 'ConvResAdapter':
        return ConvResAdapter(in_channels=1, hidden_dim=32)
    elif adapter_type == 'TimeSpaceAdapter':
        return TimeSpaceAdapter(in_channels=2, hidden_channels=64)
    elif adapter_type == 'DeepMultiTimeAdapter':
        return DeepMultiTimeAdapter(history_window=window_size, hidden_channels=64)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


def compute_loss(adjusted, target, mask):
    """与 inference.py 一致"""
    mask_bool = (mask > 0.5)
    if mask_bool.sum() == 0:
        return torch.tensor(0.0, device=adjusted.device)
    
    loss_mse = F.mse_loss(adjusted[mask_bool], target[mask_bool])
    loss_l1 = F.l1_loss(adjusted[mask_bool], target[mask_bool])
    penalty_under = torch.mean(torch.relu(-adjusted[mask_bool])**2)
    penalty_over = torch.mean(torch.relu(adjusted[mask_bool] - 1.0)**2)
    
    return 10.0 * loss_mse + 2.0 * loss_l1 + 10.0 * (penalty_under + penalty_over)


def train_on_sample(base_model, adapter, features, targets, global_mask_np, 
                    optimizer, args, device, history_residuals=None):
    """
    完全模仿 inference.py 的 run_inference_with_multi_history
    关键：Adapter 处理 30x30 patches，不是 full image！
    """
    # 基础模型前向（无梯度）
    with torch.no_grad():
        base_output = base_model(features)
        if base_output.dim() == 3:
            base_output = base_output.unsqueeze(1)
        base_output = base_output.detach()
    
    # 删除 features，释放内存
    del features
    torch.cuda.empty_cache()
    
    B, C, H, W = base_output.shape
    
    # 加载掩码到 GPU
    curr_mask = torch.from_numpy(global_mask_np).float().to(device).unsqueeze(0).unsqueeze(0)[:, :, :H, :W]
    
    # 准备融合权重
    stride = args.grid_size // 2
    fusion_weight = torch.ones((1, 1, args.grid_size, args.grid_size), device=device)
    for i in range(args.grid_size):
        dist = min(i, args.grid_size - 1 - i) / (args.grid_size // 2)
        fusion_weight[:, :, i, :] *= dist
        fusion_weight[:, :, :, i] *= dist
    fusion_weight = torch.clamp(fusion_weight, min=0.1)
    
    # 准备历史（GPU tensors，与 inference.py 一致）
    if args.adapter_type in ['TimeSpaceAdapter', 'DeepMultiTimeAdapter']:
        history_list = []
        for i in range(args.window_size):
            if history_residuals and i < len(history_residuals):
                r = history_residuals[-(i+1)]
                # 调整尺寸
                if r.shape[-2:] != (H, W):
                    r = F.interpolate(r, size=(H, W), mode='bilinear', align_corners=False)
                history_list.append(r)
            else:
                history_list.append(torch.zeros((B, 1, H, W), device=device))
        combined_history = torch.cat(history_list, dim=1)  # [B, window_size, H, W]
    
    # ====== 提取 patches（关键！与 inference.py 一致）======
    base_patches, gt_patches, mask_patches, hist_patches, coords = [], [], [], [], []
    for b in range(B):
        for y in range(0, H - args.grid_size + 1, stride):
            for x in range(0, W - args.grid_size + 1, stride):
                base_patches.append(base_output[b:b+1, :, y:y+args.grid_size, x:x+args.grid_size])
                gt_patches.append(targets[b:b+1, :, y:y+args.grid_size, x:x+args.grid_size])
                mask_patches.append(curr_mask[b:b+1, :, y:y+args.grid_size, x:x+args.grid_size])
                if args.adapter_type in ['TimeSpaceAdapter', 'DeepMultiTimeAdapter']:
                    hist_patches.append(combined_history[b:b+1, :, y:y+args.grid_size, x:x+args.grid_size])
                coords.append((b, y, x))
    
    # Cat 所有 patches（与 inference.py 一致）
    all_base = torch.cat(base_patches, dim=0)  # [num_patches, 1, 30, 30]
    all_gt = torch.cat(gt_patches, dim=0)
    all_mask = torch.cat(mask_patches, dim=0)
    if args.adapter_type in ['TimeSpaceAdapter', 'DeepMultiTimeAdapter']:
        all_hist = torch.cat(hist_patches, dim=0)  # [num_patches, window_size, 30, 30]
    
    # 删除原始 tensors，只保留 patches
    del base_output, targets, curr_mask
    if args.adapter_type in ['TimeSpaceAdapter', 'DeepMultiTimeAdapter']:
        del combined_history
    torch.cuda.empty_cache()
    
    num_patches = all_base.size(0)
    
    # ====== 多轮迭代训练（与 inference.py 一致）======
    adapter.train()
    final_loss = 0.0
    
    for iteration in range(args.iterations_per_sample):
        indices = torch.randperm(num_patches)
        
        for start_idx in range(0, num_patches, args.inner_batch_size):
            end_idx = min(start_idx + args.inner_batch_size, num_patches)
            idx = indices[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # Adapter 处理 patches（关键！）
            if args.adapter_type == 'TimeSpaceAdapter':
                # TimeSpaceAdapter: 输入当前预测 + 上期残差（在 patch 中）
                adjusted = adapter(all_base[idx], all_hist[idx][:, -1:, :, :])  # 只取最后一个月
            elif args.adapter_type == 'DeepMultiTimeAdapter':
                adjusted = adapter(all_base[idx], all_hist[idx])
            else:
                adjusted = adapter(all_base[idx])
            
            loss = compute_loss(adjusted, all_gt[idx], all_mask[idx])
            
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
            
            del adjusted, loss
        
        # 定期清理
        if iteration % 10 == 0:
            torch.cuda.empty_cache()
    
    # ====== 融合 patches（与 inference.py 一致）======
    adapter.eval()
    with torch.no_grad():
        combined_output = torch.zeros((B, 1, H, W), device=device)
        weight_sum = torch.zeros((B, 1, H, W), device=device)
        
        for i in range(0, num_patches, args.inner_batch_size):
            end_i = min(i + args.inner_batch_size, num_patches)
            
            if args.adapter_type == 'TimeSpaceAdapter':
                refined = adapter(all_base[i:end_i], all_hist[i:end_i][:, -1:, :, :])
            elif args.adapter_type == 'DeepMultiTimeAdapter':
                refined = adapter(all_base[i:end_i], all_hist[i:end_i])
            else:
                refined = adapter(all_base[i:end_i])
            
            patch_coords = coords[i:end_i]
            for j in range(len(refined)):
                if j >= len(patch_coords):
                    break
                b, y, x = patch_coords[j]
                combined_output[b:b+1, :, y:y+args.grid_size, x:x+args.grid_size] += refined[j:j+1] * fusion_weight
                weight_sum[b:b+1, :, y:y+args.grid_size, x:x+args.grid_size] += fusion_weight
        
        final_output = torch.where(weight_sum > 0, combined_output / weight_sum, torch.zeros_like(combined_output))
    
    # 裁剪并应用掩码（使用原始的 full-size mask）
    final_output = torch.clamp(final_output, 0.0, 1.0)
    
    # 计算残差（需要 full-size targets 和 mask）
    # 重新加载 targets 到 GPU（或者从 all_gt 重建）
    # 简化：直接从 all_gt 重建 full image
    with torch.no_grad():
        full_gt = torch.zeros((B, 1, H, W), device=device)
        full_mask = torch.zeros((B, 1, H, W), device=device)
        for i, (b, y, x) in enumerate(coords):
            full_gt[b:b+1, :, y:y+args.grid_size, x:x+args.grid_size] += all_gt[i:i+1] * fusion_weight
            full_mask[b:b+1, :, y:y+args.grid_size, x:x+args.grid_size] += fusion_weight
        full_gt = full_gt / weight_sum.clamp(min=1e-8)
        
        # 计算残差
        residual = (final_output - full_gt).detach()
    
    # 清理
    del all_base, all_gt, all_mask
    if args.adapter_type in ['TimeSpaceAdapter', 'DeepMultiTimeAdapter']:
        del all_hist
    del combined_output, weight_sum, final_output
    torch.cuda.empty_cache()
    
    return final_loss, residual


def main():
    args = parse_args()
    set_random_seeds(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载统计信息
    stats = {}
    if os.path.exists(args.stats_file):
        with open(args.stats_file, 'r') as f:
            stats = json.load(f)
    
    # 加载掩码（numpy）
    mask_path = os.path.join(args.dataset_dir, "mask.npy")
    if os.path.exists(mask_path):
        global_mask_np = np.load(mask_path)
        print(f"Loaded mask: {global_mask_np.shape}")
    else:
        global_mask_np = np.ones(TARGET_SHAPE, dtype=np.float32)
    
    # 加载基础模型
    print(f"\nLoading base model from {args.base_model}")
    base_model = load_model(args.base_model, device)
    for param in base_model.parameters():
        param.requires_grad = False
    print("Base model frozen")
    
    # 创建 Adapter
    adapter = create_adapter(args.adapter_type, args.window_size, args.grid_size)
    adapter = adapter.to(device)
    print(f"Adapter: {args.adapter_type}, params: {sum(p.numel() for p in adapter.parameters()):,}")
    
    # 获取数据
    npy_files = get_npy_files(args.dataset_dir)
    filtered_files = filter_files_by_date(
        npy_files, start_date=args.start_date, end_date=args.end_date, mode='between'
    )
    
    feature_files = {cat: filtered_files.get(cat, []) for cat in CATEGORIES}
    label_files = {'NDVI_Monthly': filtered_files.get('NDVI_Monthly', [])}
    slope_path = os.path.join(args.dataset_dir, "slope.npy")
    elevation_path = os.path.join(args.dataset_dir, "elevation.npy")
    
    dataset = NDVIDataset(
        feature_files, label_files, slope_path, elevation_path, 
        mask_path, stats, mode='train'
    )
    print(f"Dataset size: {len(dataset)}")
    
    # 按时序划分
    val_size = int(args.val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    
    print(f"\nSplit: Train {train_size} | Val {val_size}")
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, 
                              num_workers=args.num_workers, pin_memory=True)
    
    # 优化器
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    
    print(f"\nConfig: iter={args.iterations_per_sample}, lr={args.lr}, epochs={args.epochs}")
    
    # 训练循环
    best_loss = float('inf')
    history = {'train_losses': [], 'val_losses': [], 'metrics': []}
    
    # 历史队列（跨epochs保持，与 inference.py 一致）
    res_queue = deque(maxlen=args.window_size)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        adapter.train()
        train_losses = []
        all_metrics = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_data in pbar:
            if len(batch_data) == 3:
                features, targets, _ = batch_data
            else:
                features, targets = batch_data
            
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1) if targets.dim() == 3 else targets.to(device)
            
            # 准备历史（GPU list）
            history_list = list(res_queue) if len(res_queue) > 0 else None
            
            # 训练（patches 方式）
            loss, residual = train_on_sample(
                base_model, adapter, features, targets, 
                global_mask_np, optimizer, args, device, history_list
            )
            
            train_losses.append(loss)
            
            # 计算指标（基于残差和真实值）
            with torch.no_grad():
                # 获取调整后的预测
                adjusted_pred = targets + residual
                pred_np = adjusted_pred.squeeze().cpu().numpy()
                target_np = targets.squeeze().cpu().numpy()
                
                # 计算指标
                land_mask = (global_mask_np > 0.5)
                if land_mask.any():
                    metrics = calculate_metrics(pred_np.flatten(), target_np.flatten(), land_mask.flatten())
                    all_metrics.append(metrics)
                    pbar.set_postfix({
                        'loss': f"{loss:.6f}", 
                        'RMSE': f"{metrics['rmse']:.4f}",
                        'R2': f"{metrics['r2']:.4f}"
                    })
                else:
                    pbar.set_postfix({'loss': f"{loss:.6f}"})
            
            # 更新历史队列（GPU tensor）- 跨epoch保持
            res_queue.append(residual)
        
        avg_train_loss = np.mean(train_losses)
        history['train_losses'].append(avg_train_loss)
        
        # 计算平均指标
        if all_metrics:
            avg_metrics = {
                'mse': float(np.mean([m['mse'] for m in all_metrics])),
                'mae': float(np.mean([m['mae'] for m in all_metrics])),
                'rmse': float(np.mean([m['rmse'] for m in all_metrics])),
                'r2': float(np.mean([m['r2'] for m in all_metrics]))
            }
            history['metrics'].append(avg_metrics)
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {avg_train_loss:.6f}")
            print(f"  MSE:  {avg_metrics['mse']:.6f}")
            print(f"  MAE:  {avg_metrics['mae']:.6f}")
            print(f"  RMSE: {avg_metrics['rmse']:.6f}")
            print(f"  R2:   {avg_metrics['r2']:.6f}")
        else:
            print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.6f}")
        
        # 保存模型
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'args': vars(args)
            }, os.path.join(args.output_dir, f'{args.adapter_type}_best.pth'))
            print(f"  Saved best model")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'args': vars(args)
        }, os.path.join(args.output_dir, f'{args.adapter_type}_latest.pth'))
    
    # 保存历史
    with open(os.path.join(args.output_dir, f'{args.adapter_type}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 打印最终总结
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best Loss: {best_loss:.6f}")
    if history['metrics']:
        final_metrics = history['metrics'][-1]
        print(f"Final Metrics (Epoch {args.epochs}):")
        print(f"  MSE:  {final_metrics['mse']:.6f}")
        print(f"  MAE:  {final_metrics['mae']:.6f}")
        print(f"  RMSE: {final_metrics['rmse']:.6f}")
        print(f"  R2:   {final_metrics['r2']:.6f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
