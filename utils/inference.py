# inference.py
import os
import numpy as np
import torch
import torch.nn.functional as F
import random
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# ==================== Configuration ====================
TARGET_SHAPE = (2154, 4320)
CATEGORIES = ['Evapotranspiration', 'Precipitation', 'SOV', 'Temperature']

# ==================== Data I/O Functions ====================
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_npy_files(directory):
    """Get all .npy files organized by category"""
    npy_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                category = os.path.basename(root)
                if category not in npy_files:
                    npy_files[category] = []
                npy_files[category].append(os.path.join(root, file))
    return npy_files

def filter_files_by_date(file_dict, date_format="%Y%m", start_date="201501", end_date=None, mode='after'):
    """
    Filter files by date
    mode: 'after' - files after start_date, 'before' - files before end_date, 'between' - between both dates
    """
    if mode == 'after':
        cutoff = datetime.strptime(start_date, date_format)
        filtered = {}
        for category, files in file_dict.items():
            filtered[category] = []
            for fp in files:
                date_str = os.path.basename(fp).split('.')[0][-6:]
                try:
                    if datetime.strptime(date_str, date_format) >= cutoff:
                        filtered[category].append(fp)
                except ValueError:
                    continue
    elif mode == 'before':
        cutoff = datetime.strptime(end_date, date_format)
        filtered = {}
        for category, files in file_dict.items():
            filtered[category] = []
            for fp in files:
                date_str = os.path.basename(fp).split('.')[0][-6:]
                try:
                    if datetime.strptime(date_str, date_format) <= cutoff:
                        filtered[category].append(fp)
                except ValueError:
                    continue
    else:
        raise ValueError("mode must be 'after' or 'before'")
    
    return filtered

# ==================== Statistics & Normalization ====================
def load_stats(stats_path):
    """Load normalization statistics from JSON file"""
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            print("Loaded stats:", stats)  # Debug print the loaded stats
            return stats
    else:
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")

def save_stats(stats, filename="training_stats.json"):
    """Save normalization statistics to JSON"""
    stats_serializable = {}
    for cat, values in stats.items():
        stats_serializable[cat] = {
            'min': float(values['min']) if values['min'] is not None else None,
            'max': float(values['max']) if values['max'] is not None else None
        }
    with open(filename, 'w') as f:
        json.dump(stats_serializable, f, indent=2)

def denormalize(tensor, cat, stats):
    """Denormalize predictions to original scale"""
    d_min = stats[cat]['min']
    d_max = stats[cat]['max']
    #print(f"Denormalizing for {cat}: Min = {d_min}, Max = {d_max}")
    denorm_tensor = tensor * (d_max - d_min) + d_min
    #print(f"Denormalized tensor min: {np.min(denorm_tensor)}, max: {np.max(denorm_tensor)}")
    return denorm_tensor

def read_npy(file_path, category, stats_dict=None, normalize=True):
    """
    Read and process .npy file
    Args:
        stats_dict: Precomputed statistics dictionary. If None, will compute on-the-fly
        normalize: Whether to normalize the data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if os.path.getsize(file_path) == 0:
        print(f"[Warning] Empty file skipped: {file_path}")
        return np.zeros(TARGET_SHAPE, dtype=np.float32)
    
    data = np.load(file_path)
    data = np.nan_to_num(data, nan=-100.0)
    data[data == -9999.0] = -100.0
    data[np.isinf(data)] = -100.0
    data[data < -999] = -100.0
    
    if normalize and stats_dict is not None:
        # Use precomputed stats
        if category in stats_dict and stats_dict[category]['min'] is not None:
            d_min = stats_dict[category]['min']
            d_max = stats_dict[category]['max']
        else:
            # Compute on-the-fly (for inference if stats not available)
            valid = data != -100
            if valid.any():
                d_min = float(data[valid].min())
                d_max = float(data[valid].max())
            else:
                d_min, d_max = 0.0, 1.0
    elif normalize:
        # No stats provided, compute from data
        valid = data != -100
        if valid.any():
            d_min = float(data[valid].min())
            d_max = float(data[valid].max())
        else:
            d_min, d_max = 0.0, 1.0
    else:
        # No normalization
        d_min, d_max = 0.0, 1.0
    
    # Apply normalization if requested
    if normalize and (d_max - d_min) > 0:
        valid = data != -100
        data[valid] = (data[valid] - d_min) / (d_max - d_min + 1e-8)
        data[~valid] = 0.0
    elif not normalize:
        # For labels, just mask invalid values
        data[data == -100] = 0.0
    
    return data.astype(np.float32)

def seed_worker(worker_id):
    """Set seed for data loader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==================== Evaluation Metrics ====================
def calculate_metrics(pred, actual, mask=None):
    """Calculate evaluation metrics"""
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    else:
        mask = mask > 0
    
    pred_valid = pred[mask]
    actual_valid = actual[mask]
    
    # Avoid log warnings
    pred_valid = np.clip(pred_valid, 1e-8, None)
    
    mse = np.mean((pred_valid - actual_valid) ** 2)
    mae = np.mean(np.abs(pred_valid - actual_valid))
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((actual_valid - pred_valid) ** 2)
    ss_tot = np.sum((actual_valid - np.mean(actual_valid)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def run_inference(model, dataloader, device, output_dir, denormalize_output=True, stats=None, adapter=None):
    """Run inference and save predictions with optional adapter"""
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    file_paths = []
    
    with torch.no_grad():
        for batch_idx, (features, _, file_path) in enumerate(tqdm(dataloader, desc="Inference")):
            features = features.to(device)
            
            # Forward pass through the model
            with torch.cuda.amp.autocast():
                outputs = model(features)
            
            # If adapter is provided, pass the output through the adapter
            if adapter:
                outputs = adapter(outputs)
            
            # Process each sample
            for i in range(outputs.shape[0]):
                pred = outputs[i].cpu().numpy()
                
                # Denormalize if requested
                if denormalize_output and stats and 'NDVI_Monthly' in stats:
                    pred = denormalize(torch.from_numpy(pred), 'NDVI_Monthly', stats).numpy()
                
                predictions.append(pred)
                
                # Save prediction
                base_name = os.path.basename(file_path[i]).replace('.npy', '_pred.npy')
                save_path = os.path.join(output_dir, base_name)
                np.save(save_path, pred)
                file_paths.append(file_path[i])
    
    print(f"Saved {len(predictions)} predictions to {output_dir}")
    return predictions, file_paths

def visualize_and_evaluate(predictions, file_paths, dataset, output_dir, stats, num_viz=10):
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    metrics = []
    
    for i in tqdm(range(min(num_viz, len(predictions))), desc="Visualization"):
        # 1. 获取真实标签
        _, actual, fp = dataset[i]
        actual_np = actual.numpy().squeeze() # 确保是 (H, W)
        
        # 2. 获取预测值 (处理可能存在的 Batch 维度)
        pred_np = predictions[i]
        if isinstance(pred_np, np.ndarray):
            # 如果 pred_np 是 (Batch, C, H, W)，取第一个样本并去掉多余维度
            if pred_np.ndim == 4:
                pred_np = pred_np[0].squeeze()
            elif pred_np.ndim == 3:
                pred_np = pred_np.squeeze()
        
        # --- 检查点：确保形状完全一致 ---
        if pred_np.shape != actual_np.shape:
            # 如果形状不匹配，强制调整或报错
            from skimage.transform import resize
            pred_np = resize(pred_np, actual_np.shape, preserve_range=True)

        # 3. 计算指标 (只针对有效值)
        mask = np.isfinite(actual_np) & (actual_np != -100.0)
        metric = calculate_metrics(pred_np.flatten(), actual_np.flatten(), mask.flatten())
        metric['file'] = os.path.basename(fp)
        metrics.append(metric)

        # 4. 绘图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 设置统一的色阶范围，方便对比
        vmin, vmax = -0.1, 0.9 # NDVI 的典型范围
        
        im0 = axes[0].imshow(pred_np, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Refined Prediction\n(ROI Modified)')
        plt.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.04)
        
        im1 = axes[1].imshow(actual_np, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        axes[1].set_title('Ground Truth')
        plt.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.04)

        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'eval_{os.path.basename(fp)}.png'))
        plt.close()
    
    # Save metrics CSV
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, 'inference_metricsgit.csv'), index=False)
    print(f"Metrics saved. Mean RMSE: {df['rmse'].mean():.4f}, Mean R²: {df['r2'].mean():.4f}")

def split_image_into_blocks(batch_image, block_size=256):
    """
    Split a batch of images into smaller blocks.
    Args:
        batch_image (torch.Tensor): The input batch of images tensor of shape (batch_size, C, H, W).
        block_size (int): The size of each block (e.g., 256 for 256x256 blocks).
    Returns:
        List of blocks (each block is a tensor of shape (batch_size, C, block_size, block_size)).
    """
    batch_size, C, H, W = batch_image.shape  # Ensure the input is a batch of images
    blocks = []

    # Iterate through the batch
    for i in range(batch_size):
        image = batch_image[i]
        # Split each image into blocks
        for h in range(0, H, block_size):
            for w in range(0, W, block_size):
                block = image[:, h:h+block_size, w:w+block_size]
                blocks.append(block)

    return blocks

def save_residual_map(prediction, ground_truth, save_path, filename):
    """
    计算并保存残差图：Residual = Prediction - Ground Truth
    """
    # 确保维度一致 [H, W]
    pred = prediction.squeeze()
    gt = ground_truth.squeeze()
    
    # 计算残差
    residual = pred - gt
    
    # 1. 保存为原始数值文件 (.npy)，便于后续定量统计指标分析
    np.save(os.path.join(save_path, f"{filename}_residual.npy"), residual)
    
    # 2. (可选) 保存为可视化图片，直观观察空间误差分布
    # 红色代表预测偏高，蓝色代表预测偏低，白色代表准确
    plt.figure(figsize=(10, 8))
    plt.imshow(residual, cmap='coolwarm', vmin=-0.2, vmax=0.2) # NDVI残差通常在这个范围
    plt.colorbar(label='Residual (Pred - GT)')
    plt.title(f'Residual Map: {filename}')
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"{filename}_residual_viz.png"), bbox_inches='tight')
    plt.close()

    return residual.mean(), np.abs(residual).mean() # 返回偏置和MAE供参考

def run_inference_with_adapter(model, dataloader, device, output_dir, denormalize_output=True, stats=None, adapter=None, grid_size=30, num_iterations=500):
    os.makedirs(output_dir, exist_ok=True)
    predictions, file_paths = [], []
    optimizer = torch.optim.Adam(adapter.parameters(), lr=2e-3)

    stride = grid_size // 2  
    mask = torch.ones((1, 1, grid_size, grid_size)).to(device)
    for i in range(grid_size):
        dist = min(i, grid_size - 1 - i) / (grid_size // 2)
        mask[:, :, i, :] *= dist
        mask[:, :, :, i] *= dist
    mask = torch.clamp(mask, min=0.1) 

    model.eval()
    
    for batch_idx, (features, targets, file_path) in enumerate(tqdm(dataloader, desc="Refining")):
        features = features.to(device)
        targets = targets.to(device).unsqueeze(1) if targets.dim() == 3 else targets.to(device)
        
        with torch.no_grad():
            base_output = model(features)
            if base_output.dim() == 3: base_output = base_output.unsqueeze(1)
            base_output = base_output.detach().float()

        B, C, H, W = base_output.shape
        
        # --- 1. 提取所有 Patches ---
        base_patches, gt_patches, coords = [], [], []
        for y in range(0, H - grid_size + 1, stride):
            for x in range(0, W - grid_size + 1, stride):
                base_patches.append(base_output[:, :, y:y+grid_size, x:x+grid_size])
                gt_patches.append(targets[:, :, y:y+grid_size, x:x+grid_size])
                coords.append((y, x))
        
        all_base = torch.cat(base_patches, dim=0) 
        all_gt = torch.cat(gt_patches, dim=0)

        # --- 2. 核心改进：小批量迭代微调 ---
        # 4万个块太多了，分批处理防止 OOM
        adapter.train()
        inner_batch_size = 128 # 根据你的显存调整，128-512 比较稳妥
        num_patches = all_base.size(0)

        for iteration in range(num_iterations):
            # 每一轮随机打乱 Patch 顺序有助于更好过拟合
            indices = torch.randperm(num_patches)
            epoch_loss = 0
            
            for start_idx in range(0, num_patches, inner_batch_size):
                end_idx = min(start_idx + inner_batch_size, num_patches)
                batch_indices = indices[start_idx:end_idx]
                
                # 获取当前小批次的 Patch
                b_in = all_base[batch_indices]
                b_gt = all_gt[batch_indices]

                optimizer.zero_grad()
                adjusted = adapter(b_in)
                
                loss_mse = torch.nn.functional.mse_loss(adjusted, b_gt)
                loss_l1 = torch.nn.functional.l1_loss(adjusted, b_gt)
                
                # 组合 Loss
                total_loss = 10.0 * loss_mse + 2.0 * loss_l1
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            #if iteration % 10 == 0:
                #print(f"Iter {iteration} | Avg Loss: {epoch_loss / (num_patches/inner_batch_size):.6f}")

        # --- 3. 融合贴回 (推理时分批以节省显存) ---
        adapter.eval()
        with torch.no_grad():
            combined_output = torch.zeros_like(base_output)
            weight_sum = torch.zeros_like(base_output)
            
            # 推理也建议分批
            refined_list = []
            for i in range(0, num_patches, inner_batch_size):
                end_i = min(i + inner_batch_size, num_patches)
                refined_list.append(adapter(all_base[i:end_i]))
            refined_patches = torch.cat(refined_list, dim=0)
            
            for i, (y, x) in enumerate(coords):
                patch_to_add = refined_patches[i:i+1] 
                combined_output[0:1, :, y:y+grid_size, x:x+grid_size] += patch_to_add * mask
                weight_sum[0:1, :, y:y+grid_size, x:x+grid_size] += mask

            final_output = torch.where(weight_sum > 0, combined_output / weight_sum, base_output)
            
            #rmse = torch.sqrt(torch.nn.functional.mse_loss(final_output, targets)).item()
            #print(f"Batch {batch_idx} | RMSE: {rmse:.6f}")

            # 打印与 GT 的对齐程度
            #final_rmse = torch.sqrt(torch.nn.functional.mse_loss(final_output, targets)).item()
            #print(f"Batch {batch_idx} | Current Sample RMSE: {final_rmse:.6f}")

            # --- 5. 保存结果 ---
            final_output_np = final_output.cpu().numpy()
            predictions.append(final_output_np)
            
            for i in range(final_output_np.shape[0]):
                base_name = os.path.basename(file_path[i]).replace('.npy', '')
                
                # 保存预测结果
                save_path_pred = os.path.join(output_dir, f"{base_name}_pred.npy")
                np.save(save_path_pred, final_output_np[i])
                
                # --- 新增：残差计算与保存 ---
                # 这里的 targets 是你在微调时用的 GT
                gt_np = targets[i].cpu().numpy()
                residual = final_output_np[i] - gt_np
                
                save_path_res = os.path.join(output_dir, f"{base_name}_residual.npy")
                np.save(save_path_res, residual)
                
                # 统计一下当前样本的平均残差，看看有没有系统性偏差
                print(f"Sample {base_name} | Bias: {residual.mean():.6f}")

    return predictions, file_paths