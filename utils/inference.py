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
import seaborn as sns

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

def save_residual_map(
        prediction,
        ground_truth,
        output_dir,
        base_name
    ):
    """
    计算残差并保存。残差 = 预测 - 真值
    """
    # 确保是 2D 形状 [H, W]
    pred = prediction.squeeze()
    gt = ground_truth.squeeze()
    residual = pred - gt
    
    # 1. 保存数值，用于后续定量分析 RMSE, R2
    res_path = os.path.join(output_dir, f"{base_name}_residual.npy")
    np.save(res_path, residual)
    
    # 2. 自动化可视化：这对研究空间模式非常有帮助
    plt.figure(figsize=(8, 6))
    # 使用 coolwarm 色调，0值对应白色，正值为红，负值为蓝
    # vmin/vmax 设置为 0.1 左右，因为 NDVI 细微偏差更有意义
    im = plt.imshow(residual, cmap='coolwarm', vmin=-0.1, vmax=0.1)
    ax = plt.gca()
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    #plt.colorbar(im, label='Residual (Pred - GT)')
    plt.title(f"Residual Map: {base_name}")
    plt.axis('off')
    
    viz_path = os.path.join(output_dir, f"{base_name}_residual_viz.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_residual_distribution(
        prediction,
        ground_truth,
        output_dir,
        base_name,
        mask_path='datasets/AWI-CM-1-1-MR/mask.npy'
        ): # 新增 current_mask 参数

        """
        专门针对陆地像素统计残差分布，标注 μ 和 σ
        """
        # 1. 计算原始残差
        residual_full = (prediction - ground_truth).flatten()
        
        # 2. 加载海洋掩码 (假设 1 为陆地, 0 为海洋)
        # 如果 mask 很大，建议在外部加载好传进来，这里为了演示逻辑闭环写在内部
        land_mask = np.load(mask_path).flatten()
        
        # 3. 核心步骤：只筛选出陆地部分的残差
        # 确保 mask 长度与展平后的图像一致
        land_residuals = residual_full[land_mask == 1]
        
        if len(land_residuals) == 0:
            print(f"Warning: No land pixels found for {base_name}")
            return

        # 4. 计算陆地统计量
        mu = np.mean(land_residuals)
        sigma = np.std(land_residuals)
        
        # 5. 绘图
        plt.figure(figsize=(10, 6))
        
        # 绘制直方图 (只包含陆地)
        sns.histplot(land_residuals, bins=100, kde=True, color='teal', edgecolor='white', alpha=0.7)
        
        # 绘制 0 基准线
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
        
        # 在图中标注统计量
        stats_text = f'$\mu_{{land}} = {mu:.6f}$\n$\sigma_{{land}} = {sigma:.6f}$'
        plt.gca().text(0.95, 0.90, stats_text, transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.title(f'Land-Only Residual Distribution: {base_name}')
        plt.xlabel('Residual (Pred - GT)')
        plt.ylabel('Pixel Count')
        
        # 设置合理的 X 轴范围，过滤掉 0 值的海洋干扰后，
        # 陆地分布的“宽度”会展现得非常清晰
        plt.xlim(-0.2, 0.2) 
        plt.grid(axis='y', alpha=0.3)
        
        dist_path = os.path.join(output_dir, f"{base_name}_residual_dist.png")
        plt.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close()

def run_inference_with_adapter(
        model,
        dataloader,
        device,
        output_dir,
        denormalize_output=True,
        stats=None, adapter=None,
        grid_size=30,
        num_iterations=100,
        mask_path='datasets/AWI-CM-1-1-MR/mask.npy'
        ):
    
    os.makedirs(output_dir, exist_ok=True)
    predictions, file_paths = [], []
    optimizer = torch.optim.Adam(adapter.parameters(), lr=2e-3)

    # --- 加载全局掩码 ---
    # 确保掩码在 GPU 上，形状为 [1, 1, H, W]
    global_mask_np = np.load(mask_path)
    global_mask = torch.from_numpy(global_mask_np).float().to(device).unsqueeze(0).unsqueeze(0)

    stride = grid_size // 2  
    fusion_mask = torch.ones((1, 1, grid_size, grid_size)).to(device)
    for i in range(grid_size):
        dist = min(i, grid_size - 1 - i) / (grid_size // 2)
        fusion_mask[:, :, i, :] *= dist
        fusion_mask[:, :, :, i] *= dist
    fusion_mask = torch.clamp(fusion_mask, min=0.1) 

    model.eval()
    
    for batch_idx, (features, targets, file_path) in enumerate(tqdm(dataloader, desc="Refining")):
        features = features.to(device)
        targets = targets.to(device).unsqueeze(1) if targets.dim() == 3 else targets.to(device)
        
        with torch.no_grad():
            base_output = model(features)
            if base_output.dim() == 3: base_output = base_output.unsqueeze(1)
            base_output = base_output.detach().float()

        B, C, H, W = base_output.shape
        # 裁剪 mask 以匹配当前输出尺寸
        curr_mask = global_mask[:, :, :H, :W]
        
        # --- 1. 提取所有 Patches ---
        base_patches, gt_patches, mask_patches, coords = [], [], [], []
        for y in range(0, H - grid_size + 1, stride):
            for x in range(0, W - grid_size + 1, stride):
                base_patches.append(base_output[:, :, y:y+grid_size, x:x+grid_size])
                gt_patches.append(targets[:, :, y:y+grid_size, x:x+grid_size])
                mask_patches.append(curr_mask[:, :, y:y+grid_size, x:x+grid_size]) # 提取掩码块
                coords.append((y, x))
        
        all_base = torch.cat(base_patches, dim=0) 
        all_gt = torch.cat(gt_patches, dim=0)
        all_mask = torch.cat(mask_patches, dim=0) # [N, 1, 30, 30]

        # --- 2. 带掩码的小批量迭代微调 ---
        adapter.train()
        inner_batch_size = 128 
        num_patches = all_base.size(0)

        for iteration in range(num_iterations):
            indices = torch.randperm(num_patches)
            for start_idx in range(0, num_patches, inner_batch_size):
                end_idx = min(start_idx + inner_batch_size, num_patches)
                batch_indices = indices[start_idx:end_idx]
                
                b_in = all_base[batch_indices]
                b_gt = all_gt[batch_indices]
                b_mask = all_mask[batch_indices] # 获取当前批次的掩码

                optimizer.zero_grad()
                adjusted = adapter(b_in)
                
                # --- Loss 屏蔽海洋区域 ---
                # 只计算 b_mask == 1 的位置
                mask_bool = (b_mask > 0.5) 
                if mask_bool.sum() > 0: # 确保当前批次有陆地
                    loss_mse = torch.nn.functional.mse_loss(adjusted[mask_bool], b_gt[mask_bool])
                    loss_l1 = torch.nn.functional.l1_loss(adjusted[mask_bool], b_gt[mask_bool])
                    penalty_under = torch.mean(torch.relu(-adjusted[mask_bool])**2) # 针对陆地像素，惩罚小于 0 的值
                    penalty_over = torch.mean(torch.relu(adjusted[mask_bool] - 1.0)**2) # 针对陆地像素，惩罚大于 1 的值
                    
                    total_loss = 10.0 * loss_mse + 2.0 * loss_l1 + 10.0 * (penalty_under + penalty_over)
                    total_loss.backward()
                    optimizer.step()

        # --- 3. 融合 ---
        adapter.eval()
        with torch.no_grad():
            combined_output = torch.zeros_like(base_output)
            weight_sum = torch.zeros_like(base_output)
            
            refined_list = []
            for i in range(0, num_patches, inner_batch_size):
                end_i = min(i + inner_batch_size, num_patches)
                refined_list.append(adapter(all_base[i:end_i]))
            refined_patches = torch.cat(refined_list, dim=0)
            
            for i, (y, x) in enumerate(coords):
                patch_to_add = refined_patches[i:i+1] 
                combined_output[0:1, :, y:y+grid_size, x:x+grid_size] += patch_to_add * fusion_mask
                weight_sum[0:1, :, y:y+grid_size, x:x+grid_size] += fusion_mask

            final_output = torch.where(weight_sum > 0, combined_output / weight_sum, base_output)
            
            # --- 4. 预测后处理：强制归零海洋（防止微小抖动） ---
            final_output = final_output * curr_mask 

            # --- 5. 保存结果 ---
            final_output_np = final_output.cpu().numpy()
            targets_np = targets.cpu().numpy()
            # 同样对 targets 做掩码，确保计算 MAE 时也是陆地
            targets_masked_np = targets_np * curr_mask.cpu().numpy()
            
            for i in range(final_output_np.shape[0]):
                base_name = os.path.basename(file_path[i]).replace('.npy', '')
                
                save_path_pred = os.path.join(output_dir, f"{base_name}_pred.npy")
                np.save(save_path_pred, final_output_np[i])
                
                # 传入 mask_path 进行绘图过滤
                save_residual_map(
                    prediction=final_output_np[i], 
                    ground_truth=targets_masked_np[i], 
                    output_dir=output_dir, 
                    base_name=base_name
                )

                save_residual_distribution(
                    prediction=final_output_np[i], 
                    ground_truth=targets_masked_np[i], 
                    output_dir=output_dir, 
                    base_name=base_name,
                    mask_path=mask_path
                )
                
                file_paths.append(file_path[i])
                predictions.append(final_output_np[i])

            # 打印陆地 MAE
            land_mask_idx = (curr_mask.cpu().numpy() > 0.5)
            if land_mask_idx.any():
                batch_mae = np.abs(final_output_np[land_mask_idx] - targets_np[land_mask_idx]).mean()
                print(f"Batch {batch_idx} | Land MAE: {batch_mae:.6f}")

    return predictions, file_paths

def run_inference_with_time_adapter(
        model,
        dataloader,
        device,
        output_dir,
        denormalize_output=True,
        stats=None, 
        adapter=None,
        grid_size=30,
        num_iterations=50, # 有了时序先验，迭代次数可以适当减少
        mask_path='datasets/AWI-CM-1-1-MR/mask.npy'
        ):
    
    os.makedirs(output_dir, exist_ok=True)
    predictions, file_paths = [], []
    optimizer = torch.optim.Adam(adapter.parameters(), lr=2e-3)

    # --- 1. 初始化资源 ---
    global_mask_np = np.load(mask_path)
    global_mask = torch.from_numpy(global_mask_np).float().to(device).unsqueeze(0).unsqueeze(0)
    
    # 这一步很关键：初始化“前一月残差”为全 0
    prev_residual = None 

    stride = grid_size // 2  
    fusion_mask = torch.ones((1, 1, grid_size, grid_size)).to(device)
    for i in range(grid_size):
        dist = min(i, grid_size - 1 - i) / (grid_size // 2)
        fusion_mask[:, :, i, :] *= dist
        fusion_mask[:, :, :, i] *= dist
    fusion_mask = torch.clamp(fusion_mask, min=0.1) 

    model.eval()
    
    # 注意：dataloader 必须 shuffle=False 以保证时间连续
    for batch_idx, (features, targets, file_path) in enumerate(tqdm(dataloader, desc="Time-Linked Refining")):
        features = features.to(device)
        targets = targets.to(device).unsqueeze(1) if targets.dim() == 3 else targets.to(device)
        
        with torch.no_grad():
            base_output = model(features)
            if base_output.dim() == 3: base_output = base_output.unsqueeze(1)
            base_output = base_output.detach().float()

        B, C, H, W = base_output.shape
        curr_mask = global_mask[:, :, :H, :W]
        
        # 如果是第一个月，初始化 prev_residual 为全 0
        if prev_residual is None:
            prev_residual = torch.zeros_like(base_output)
        
        # --- 2. 提取 Patches (增加 prev_res_patches) ---
        base_patches, gt_patches, mask_patches, res_patches, coords = [], [], [], [], []
        for y in range(0, H - grid_size + 1, stride):
            for x in range(0, W - grid_size + 1, stride):
                base_patches.append(base_output[:, :, y:y+grid_size, x:x+grid_size])
                gt_patches.append(targets[:, :, y:y+grid_size, x:x+grid_size])
                mask_patches.append(curr_mask[:, :, y:y+grid_size, x:x+grid_size])
                # 提取上个月在该位置的残差块
                res_patches.append(prev_residual[:, :, y:y+grid_size, x:x+grid_size])
                coords.append((y, x))
        
        all_base = torch.cat(base_patches, dim=0) 
        all_gt = torch.cat(gt_patches, dim=0)
        all_mask = torch.cat(mask_patches, dim=0)
        all_prev_res = torch.cat(res_patches, dim=0)

        # --- 3. 迭代微调 (Adapter 输入：当前预测 + 上月残差) ---
        adapter.train()
        inner_batch_size = 128 
        num_patches = all_base.size(0)

        for iteration in range(num_iterations):
            indices = torch.randperm(num_patches)
            for start_idx in range(0, num_patches, inner_batch_size):
                batch_indices = indices[start_idx : min(start_idx + inner_batch_size, num_patches)]
                
                b_in = all_base[batch_indices]
                b_gt = all_gt[batch_indices]
                b_mask = all_mask[batch_indices]
                b_prev_res = all_prev_res[batch_indices]

                optimizer.zero_grad()
                
                # --- 核心改变：Adapter 接收两个输入 ---
                # 假设你的 Adapter forward 已经改为接收 (x, residual)
                # 或者你将它们 cat 在一起：adapter(torch.cat([b_in, b_prev_res], dim=1))
                adjusted = adapter(b_in, b_prev_res)
                
                mask_bool = (b_mask > 0.5) 
                if mask_bool.sum() > 0:
                    loss_mse = torch.nn.functional.mse_loss(adjusted[mask_bool], b_gt[mask_bool])
                    loss_l1 = torch.nn.functional.l1_loss(adjusted[mask_bool], b_gt[mask_bool])
                    penalty_under = torch.mean(torch.relu(-adjusted[mask_bool])**2)
                    penalty_over = torch.mean(torch.relu(adjusted[mask_bool] - 1.0)**2)
                    
                    total_loss = 10.0 * loss_mse + 2.0 * loss_l1 + 10.0 * (penalty_under + penalty_over)
                    total_loss.backward()
                    optimizer.step()

        # --- 4. 推理与融合 ---
        adapter.eval()
        with torch.no_grad():
            combined_output = torch.zeros_like(base_output)
            weight_sum = torch.zeros_like(base_output)
            
            refined_list = []
            for i in range(0, num_patches, inner_batch_size):
                end_i = min(i + inner_batch_size, num_patches)
                # 推理时也要输入 prev_residual
                refined_list.append(adapter(all_base[i:end_i], all_prev_res[i:end_i]))
            
            refined_patches = torch.cat(refined_list, dim=0)
            for i, (y, x) in enumerate(coords):
                combined_output[0:1, :, y:y+grid_size, x:x+grid_size] += refined_patches[i:i+1] * fusion_mask
                weight_sum[0:1, :, y:y+grid_size, x:x+grid_size] += fusion_mask

            final_output = torch.where(weight_sum > 0, combined_output / weight_sum, base_output)
            final_output = final_output * curr_mask # 掩码过滤

            # --- 5. 更新时序状态：保存当前残差供下个月使用 ---
            # 残差 = 预测 - 真值
            # 必须使用 .detach() 彻底切断计算图，防止内存随时间累积
            prev_residual = (final_output - targets).detach()

            # --- 6. 保存与绘图 (逻辑同前) ---
            final_output_np = final_output.cpu().numpy()
            targets_masked_np = (targets * curr_mask).cpu().numpy()
            
            base_name = os.path.basename(file_path[0]).replace('.npy', '')
            np.save(os.path.join(output_dir, f"{base_name}_pred.npy"), final_output_np[0])
            
            save_residual_map(final_output_np[0], targets_masked_np[0], output_dir, base_name)
            save_residual_distribution(final_output_np[0], targets_masked_np[0], output_dir, base_name, mask_path)
            
            file_paths.append(file_path[0])
            predictions.append(final_output_np[0])

    return predictions, file_paths