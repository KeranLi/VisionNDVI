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
    """Generate visualizations and evaluation metrics"""
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    metrics = []
    
    for i in tqdm(range(min(num_viz, len(predictions))), desc="Visualization & Evaluation"):
        # Load actual label
        features, actual, fp = dataset[i]
        actual = actual.numpy()
        pred = predictions[i]
        
        pred_viz = pred  # No need to denormalize for NDVI
        actual_viz = actual  # No need to denormalize for NDVI
        
        # Flatten the actual and predicted data to match the mask's shape
        actual_viz = actual_viz.flatten()  # Flatten to 1D array
        pred_viz = pred_viz.flatten()      # Flatten to 1D array
        
        # Create a mask based on valid data points (non-NaN, non-inf)
        mask = np.isfinite(actual_viz) & (actual_viz != -100.0)  # Creating a mask for valid values
        
        # Calculate metrics
        metric = calculate_metrics(pred_viz, actual_viz, mask)
        metric['file'] = os.path.basename(fp)
        metrics.append(metric)
        
        # Dynamically set the vmin and vmax for color scale based on data
        vmin = min(np.min(actual_viz), np.min(pred_viz))
        vmax = max(np.max(actual_viz), np.max(pred_viz))

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        im0 = axes[0].imshow(pred_viz.reshape(actual.shape), cmap='viridis', vmin=vmin, vmax=vmax)  # Reshape back for visualization
        axes[0].set_title('Prediction')

        # Remove the coordinate axes and borders
        axes[0].axis('off')

        # The color bar controls the length. Reduce the fraction to make it shorter
        plt.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.04)

        im1 = axes[1].imshow(actual_viz.reshape(actual.shape), cmap='viridis', vmin=vmin, vmax=vmax)  # Reshape back for visualization
        axes[1].set_title('Actual')

        # Remove the coordinate axes and borders
        axes[1].axis('off')

        # The color bar controls the length. Reduce the fraction to make it shorter
        plt.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.04)

        #plt.suptitle(f'NDVI Prediction - {os.path.basename(fp)}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'viz_NDVI_Prediction_{os.path.basename(fp)}.png'), dpi=300, bbox_inches='tight')
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

def run_inference_with_adapter(model, dataloader, device, output_dir, denormalize_output=True, stats=None, adapter=None, grid_size=30, accumulation_steps=4):
    """Run inference with adapter and fine-tune a small grid of the image."""
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    file_paths = []

    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)  # We optimize the adapter parameters

    model.eval()
    adapter.train()  # Make sure the adapter is in training mode (since we need to adjust weights)

    # Initialize the accumulation step counter
    accumulation_counter = 0

    with torch.no_grad():  # Disable gradients for inference
        with torch.set_grad_enabled(True):  # Ensure gradients are enabled for backpropagation
            for batch_idx, (features, _, file_path) in enumerate(tqdm(dataloader, desc="Inference")):
                features = features.to(device)

                # Step 1: Forward pass with the original model
                with torch.cuda.amp.autocast():
                    original_outputs = model(features)

                # Ensure that the original_outputs are on the same device as the adapter
                original_outputs = original_outputs.to(device)  # Make sure it's on the same device as the adapter
                original_outputs = original_outputs.to(torch.float32)  # Force the dtype to be float32

                # Check if the output is 3D (for example, single channel), and add an extra dimension for channels
                if original_outputs.dim() == 3:  # (batch_size, height, width)
                    original_outputs = original_outputs.unsqueeze(1)  # Add a channel dimension (batch_size, 1, height, width)

                # Set requires_grad to True for original_outputs to enable gradient calculation
                original_outputs.requires_grad_()  # Enable gradient calculation for original_outputs

                # Step 2: Fine-tune only a small region of the output (30x30 grid)
                if adapter:
                    # Define the 30x30 grid (small section of the image)
                    batch_size, channels, height, width = original_outputs.shape
                    grid_start_x = (height - grid_size) // 2  # Calculate the top-left corner for the grid
                    grid_start_y = (width - grid_size) // 2

                    # Extract the 30x30 region from the output tensor
                    small_grid = original_outputs[:, :, grid_start_x:grid_start_x+grid_size, grid_start_y:grid_start_y+grid_size]

                    # Apply adapter to this small region
                    adjusted_grid = adapter(small_grid)
                    original_outputs[:, :, grid_start_x:grid_start_x+grid_size, grid_start_y:grid_start_y+grid_size] = adjusted_grid

                # Step 3: Initialize adjusted_outputs before denormalizing
                adjusted_outputs = original_outputs

                # Denormalize if necessary
                if denormalize_output and stats and 'NDVI_Monthly' in stats:
                    adjusted_outputs = denormalize(original_outputs, 'NDVI_Monthly', stats)

                # Collect results
                predictions.append(adjusted_outputs.detach().cpu().numpy())  # Use detach() to avoid gradient tracking
                for i in range(adjusted_outputs.shape[0]):
                    base_name = os.path.basename(file_path[i]).replace('.npy', '_pred.npy')
                    save_path = os.path.join(output_dir, base_name)
                    np.save(save_path, adjusted_outputs[i].detach().cpu().numpy())  # Use detach() to avoid gradient tracking
                    file_paths.append(file_path[i])

                # Step 4: Optimize the adapter
                # Flatten both adjusted_outputs and original_outputs to match shapes
                adjusted_outputs_flat = adjusted_outputs.view(adjusted_outputs.size(0), -1)  # Flatten the output
                original_outputs_flat = original_outputs.view(original_outputs.size(0), -1)  # Flatten the original output

                # Now we can safely calculate the loss
                loss = torch.nn.functional.mse_loss(adjusted_outputs_flat, original_outputs_flat)  # Calculate loss based on original predictions

                # Perform backward pass
                loss.backward()  # Compute gradients

                # Perform gradient update every 'accumulation_steps' steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()  # Update adapter's parameters
                    optimizer.zero_grad()  # Zero the gradients
                    torch.cuda.empty_cache()  # Clear cached memory after each batch

                # Step 5: Clear the cache after each batch to avoid memory issues
                torch.cuda.empty_cache()  # Clear cached memory after each batch

    print(f"Saved {len(predictions)} predictions to {output_dir}")
    return predictions, file_paths