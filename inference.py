# inference.py
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Import from modules
from utils.helpers import (
    set_random_seeds, load_stats, denormalize, 
    filter_files_by_date, get_npy_files, calculate_metrics,
    TARGET_SHAPE, CATEGORIES
)
from utils.datasets import NDVIDataset
from models.models import load_model

def run_inference(model, dataloader, device, output_dir, denormalize_output=True, stats=None):
    """Run inference and save predictions"""
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    file_paths = []
    
    with torch.no_grad():
        for batch_idx, (features, _, file_path) in enumerate(tqdm(dataloader, desc="Inference")):
            features = features.to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(features)
            
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
        
        # Denormalize for visualization
        if stats and 'NDVI_Monthly' in stats:
            pred_viz = denormalize(torch.from_numpy(pred), 'NDVI_Monthly', stats).numpy()
            actual_viz = denormalize(torch.from_numpy(actual), 'NDVI_Monthly', stats).numpy()
        else:
            pred_viz = pred
            actual_viz = actual
        
        # Calculate metrics
        metric = calculate_metrics(pred_viz, actual_viz)
        metric['file'] = os.path.basename(fp)
        metrics.append(metric)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        im0 = axes[0].imshow(pred_viz, cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title('Prediction')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].imshow(actual_viz, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Actual')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'NDVI Prediction - {os.path.basename(fp)}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'viz_{i}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save metrics CSV
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, 'inference_metrics.csv'), index=False)
    print(f"Metrics saved. Mean RMSE: {df['rmse'].mean():.4f}, Mean R²: {df['r2'].mean():.4f}")

def main():
    parser = argparse.ArgumentParser(description='NDVI Prediction Inference')
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--checkpoint', type=str, default='ndvi_prediction_model.pth')
    parser.add_argument('--stats_file', type=str, default='training_stats.json')
    parser.add_argument('--output_dir', type=str, default='./inference_results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--denormalize', action='store_true')
    parser.add_argument('--start_date', type=str, default='201501')
    parser.add_argument('--num_viz', type=int, default=10)
    
    args = parser.parse_args()

    # Hardcode debug values
    print("RUNNING IN DEBUG MODE")
    args.dataset_dir = '/root/autodl-tmp/dataset'  # Change this to your actual path
    args.start_date = '198201'
    args.stats_file = 'training_stats.json'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(42)
    
    # Load checkpoint to verify it exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        return
    
    # Get files with debug info
    npy_files = get_npy_files(args.dataset_dir)
    print(f"Found files: { {k: len(v) for k, v in npy_files.items()} }")
    
    if sum(len(v) for v in npy_files.values()) == 0:
        print("ERROR: No .npy files found in dataset directory!")
        print(f"Directory contents: {os.listdir(args.dataset_dir)}")
        return
    
    # Continue with normal logic...
    stats = load_stats(args.stats_file) if os.path.exists(args.stats_file) else {}
    filtered_files = filter_files_by_date(npy_files, start_date=args.start_date, mode='after')
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(42)
    
    # Load stats
    stats = load_stats(args.stats_file) if os.path.exists(args.stats_file) else {}
    
    # Get files
    npy_files = get_npy_files(args.dataset_dir)
    filtered_files = filter_files_by_date(npy_files, start_date=args.start_date, mode='after')
    
    # Setup paths
    feature_files = {cat: filtered_files.get(cat, []) for cat in CATEGORIES}
    label_files = {'NDVI_Monthly': filtered_files.get('NDVI_Monthly', [])}
    slope_path = os.path.join(args.dataset_dir, "slope.npy")
    elevation_path = os.path.join(args.dataset_dir, "elevation.npy")
    mask_path = os.path.join(args.dataset_dir, "mask.npy")
    
    # Create dataset
    dataset = NDVIDataset(feature_files, label_files, slope_path, elevation_path, mask_path, stats, mode='inference')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Run inference
    pred_dir = os.path.join(args.output_dir, 'predictions')
    predictions, file_paths = run_inference(model, dataloader, device, pred_dir, args.denormalize, stats)
    
    # Visualize and evaluate if requested
    if args.visualize:
        print("Generating visualizations and metrics...")
        visualize_and_evaluate(predictions, file_paths, dataset, args.output_dir, stats, args.num_viz)
    
    print("Inference completed!")

if __name__ == '__main__':
    main()