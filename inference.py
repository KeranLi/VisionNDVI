# inference.py
import os
import argparse
import torch
from torch.utils.data import DataLoader

# Import from modules
from utils.helpers import (
    set_random_seeds, load_stats, denormalize, 
    filter_files_by_date, get_npy_files, calculate_metrics,
    TARGET_SHAPE, CATEGORIES
)

from utils.datasets import NDVIDataset
from utils.inference import run_inference_with_adapter, run_inference_with_time_adapter, run_inference_with_multi_history, visualize_and_evaluate  # Import functions from utils/inference
from models.models import load_model, load_adapter
from models.adapter import FineTuningAdapter, ResFineTuningAdapter, ConvResAdapter, TimeSpaceAdapter, DeepMultiTimeAdapter

import warnings

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='NDVI Prediction Inference')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/AWI-CM-1-1-MR/')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/AWI_prediction_model.pth')
    parser.add_argument('--stats_file', type=str, default='training_stats.json')
    parser.add_argument('--output_dir', type=str, default='./inference_results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--denormalize', action='store_true')
    parser.add_argument('--start_date', type=str, default='198201')
    parser.add_argument('--num_viz', type=int, default=10)
    parser.add_argument('--demo', action='store_true', help="Run inference on a demo set of 10 images")
    parser.add_argument('--adjustment_factor', type=float, default=0.1, help="Factor to adjust predictions")

    args = parser.parse_args()

    # Hardcode debug values
    print("RUNNING")

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

    # If demo mode is activated, limit dataset to 10 samples
    if args.demo:
        print(f"DEMO MODE: Processing only the first 10 samples.")
        dataset = torch.utils.data.Subset(dataset, range(10))  # in demo, only 10 samples were displayed
    else:
        print(f"FULL INFERENCE MODE: Processing all {len(dataset)} samples.")
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load adapter (for fine-tuning)
    input_shape = 30 * 30  # Adjust this value based on your model's output shape
    #adapter = FineTuningAdapter(input_size=input_shape)  # Use FineTuningAdapter with correct input size
    #adapter = ResFineTuningAdapter(input_size=input_shape)
    #adapter = ConvResAdapter()
    #adapter = TimeSpaceAdapter()
    adapter = DeepMultiTimeAdapter()
    adapter.to(device)  # Ensure it's on the correct device
    
    # Run inference with adapter
    pred_dir = os.path.join(args.output_dir, 'predictions')
    #predictions, file_paths = run_inference_with_adapter(model, dataloader, device, pred_dir, args.denormalize, stats, adapter=adapter)
    #predictions, file_paths = run_inference_with_time_adapter(model, dataloader, device, pred_dir, args.denormalize, stats, adapter=adapter)
    predictions, file_paths = run_inference_with_multi_history(
        model, 
        dataloader, 
        device, 
        pred_dir, 
        adapter=adapter,
        grid_size=30,
        mask_path='datasets/AWI-CM-1-1-MR/mask.npy',
        window_size=3,
        early_stop_threshold=1e-8, # Higher for more iterations.
        patience=10
    )

    # Visualize and evaluate if requested
    if args.visualize:
        print("Generating visualizations and metrics...")
        visualize_and_evaluate(predictions, file_paths, dataset, args.output_dir, stats, args.num_viz)
    
    print("Inference completed!")

if __name__ == '__main__':
    main()
