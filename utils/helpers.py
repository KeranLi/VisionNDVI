# utils/helpers.py
import os
import json
import numpy as np
import torch
from datetime import datetime
import random

# Constants
TARGET_SHAPE = (2154, 4320)
CATEGORIES = ['Evapotranspiration', 'Precipitation', 'SOV', 'Temperature']

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    """Set seed for data loader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_npy_files(directory):
    """
    获取所有 .npy 文件，按类别组织
    如果目录结构是 dataset/dataset/category/*.npy，自动修正
    """
    npy_files = {}
    
    # 检查是否存在多余的 'dataset' 子文件夹
    possible_subdir = os.path.join(directory, 'dataset')
    if os.path.exists(possible_subdir) and os.path.isdir(possible_subdir):
        print(f"[Warning] Detected redundant 'dataset' folder layer. Using subfolder: {possible_subdir}")
        directory = possible_subdir
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                # 获取倒数第二级目录名作为类别（跳过最后一级）
                parts = root.split(os.sep)
                if len(parts) >= 2:
                    category = parts[-1]  # 直接使用当前文件夹名作为类别
                else:
                    category = os.path.basename(root)
                
                if category not in npy_files:
                    npy_files[category] = []
                npy_files[category].append(os.path.join(root, file))
    
    # Debug print
    print(f"Scan result: { {k: len(v) for k, v in npy_files.items()} }")
    return npy_files

def filter_files_by_date(file_dict, date_format="%Y%m", start_date="201501", end_date=None, mode='after'):
    """
    Filter files by date
    mode: 'after' - files after start_date, 'before' - files before end_date
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
        print(f"[Warning] Empty file: {file_path}, returning zeros")
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

def load_stats(stats_path):
    """Load normalization statistics from JSON file"""
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
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
    return tensor * (d_max - d_min) + d_min

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