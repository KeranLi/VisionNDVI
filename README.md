
# VisionNDVI

> A script-driven repository for image/time-series processing and model training/inference (organized by scripts in the repository root directory: `tif2npy.py`, `inference.py`, various `train_OF_*.py` files, etc.).

Repository: [@KeranLi/VisonNDVI](https://github.com/KeranLi/VisonNDVI)

## Table of Contents
- [VisionNDVI](#visonndvi)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Quick Start](#quick-start)
  - [Main Scripts \& Usage Examples](#main-scripts--usage-examples)
  - [Dependencies](#dependencies)
  - [Data Format \& Suggested Workflow](#data-format--suggested-workflow)
  - [Development \& Training](#development--training)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

VisionNDVI is a codebase focused on image/time-series data processing and deep learning model training/inference (script-driven repository). The repository includes:
- **`tif2npy.py`**: Preprocessing script for converting GeoTIFF or raster images to NumPy format (for training/inference pipelines).
- **`check_bad_npy.py`**: Script to check/repair corrupted or anomalous `.npy` files (e.g., missing values, inconsistent shapes).
- **`train_OF_*.py`**: A set of scripts to train different model variants (e.g., CNN, UNet, 3D-CNN+MLP).
- **`inference.py`**: Script for running inference/prediction using a trained model.
- **`make_SOV.py`**: Computes/generates SOV or evaluation metrics (depending on implementation).
- **models/**, **loss/**, **utils/**: Directories for model definitions, loss functions, and utility functions.

**Note**: This repository is currently not packaged as a command-line tool (console_scripts); scripts are run via `python <script>.py`.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/KeranLi/VisionNDVI.git
   cd VisionNDVI
   ```

2. Create/install dependencies and activate the virtual environment:
   ```bash
   conda env create -f environment.yml -n VisionNDVI
   conda activate VisionNDVI
   ```

3. Run examples:
   - Convert GeoTIFF to `.npy`:
     ```bash
     python tif2npy.py --input-dir data/tifs --output-dir data/npy --pattern "*.tif"
     ```
   - Check `.npy` files:
     ```bash
     python check_bad_npy.py --dir data/npy
     ```
   - Train a model (see script comments for exact parameters):
     ```bash
     python train_OF_UnetCNN.py --config configs/train_unet.yaml
     ```
   - Run inference / generate results:
     ```bash
     python inference.py --model checkpoints/latest.pth --input data/npy --output results/predictions
     ```

## Main Scripts & Usage Examples

Here are common scripts and recommended usage (example parameters are placeholders, please replace them with actual values based on script implementation):

- **`tif2npy.py`**
  - Purpose: Converts raw GeoTIFF/raster data into `.npy`/array format for training/inference.
  - Example:
    ```bash
    python tif2npy.py --input-dir data/tifs --output-dir data/npy --band-mapping red:3,nir:4 --tile-size 256
    ```

- **`check_bad_npy.py`**
  - Purpose: Scans and reports (or fixes) corrupted/size anomalies/NaN values in `.npy` files.
  - Example:
    ```bash
    python check_bad_npy.py --dir data/npy --fix
    ```

- **`train_OF_*.py`**
  - Purpose: Trains different model architectures (e.g., `train_OF_UnetCNN.py`, `train_OF_CNNMLP.py`, etc.).
  - Recommended: Use YAML/JSON config files to manage training parameters (learning rate, batch size, data paths, logging/checkpoint directories, etc.).
  - Example:
    ```bash
    python train_OF_UnetCNN.py --config configs/train_unet.yaml --device cuda:0
    ```

- **`inference.py`**
  - Purpose: Loads a trained model and performs inference on new data, outputting predictions (e.g., `.npy`, `.png`, `.geotiff`).
  - Example:
    ```bash
    python inference.py --model checkpoints/best.pth --input data/npy --output results --batch-size 8
    ```

- **`make_SOV.py`**
  - Purpose: Computes SOV or other evaluation metrics (parameters depend on implementation).
  - Example:
    ```bash
    python make_SOV.py --pred-dir results --gt-dir ground_truth --out metrics/sov_report.csv
    ```

If needed, I can replace these examples with exact command-line options based on the actual parameters in the script header. Please paste the first few hundred lines of the script or allow me to access the file content.

## Dependencies

- Python 3.8+
- numpy
- torch / torchvision
- opencv-python (cv2)
- rasterio
- tqdm
- matplotlib / pillow
- scikit-image

## Data Format & Suggested Workflow

- Raw images should ideally be in GeoTIFF format with geospatial information (if GeoTIFF output is needed later).
- Use `tif2npy.py` to convert raw images into uniform NumPy arrays for training/inference.
- For large-scale or super-large images, it's recommended to use tile-based/window processing to manage memory usage.
- Keep the training/validation/test data paths consistent and clearly define `nodata` values and masking strategies in the configuration.

## Development & Training

- Recommended tools:
  - Code style: black, flake8
  - Testing: pytest (if the repository includes tests)
- Training debugging:
  - Start with small-scale datasets to verify loss reduction and output validity before scaling up to full training.
  - Use checkpointing and periodic validation set evaluation.

## Contributing

We welcome bug reports, feature suggestions, and patches via issues or PRs. Suggested contribution process:
1. Fork the repository and create a feature branch.
2. Add/modify code and include necessary documentation and tests.
3. Submit a PR with a description of the changes and their impact.

If you'd like me to add standard contributing guidelines (`CONTRIBUTING.md`) or replace the `README` script usage with precise parameters, I can complete the task and submit a PR once file content access is granted.

## License

Please refer to the `LICENSE` file in the repository for licensing information (currently recommended under the MIT License).
