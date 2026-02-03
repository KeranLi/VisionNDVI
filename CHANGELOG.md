# Change Log

## [Released]

## [1.0.0] - 2026-01-09

### Added

- Initial release, including training and inference.

### Known Issues

- Stage 1 of training focuses primarily on learning spatial distribution.
- Stage 2 introduces an Adapter to perform both numerical fine-tuning and inference.
- The structure and hyperparameters of the Adapter have not yet been finalized for experiments.

## [1.0.0] - 2026-01-10

### Added

- Fixed some bugs in denormalization and visualization.
- Add Residual convolution Adapter

### Known Issues

- NDVI ranges between 0 and 1. Denormalization is not a must.

## [1.0.0] - 2026-01-11

### Added

- Fixed and added Adapters with residual and convolutions. 
- Add Residual convolution Adapter.

### Known Issues

- Convolutions are the best.

## [1.0.0] - 2026-01-12

### Added

- Residual (Prediction - Actual) visulization maps.

### Known Issues

- Seems the residual is related the spatial patterns.

## [1.0.0] - 2026-01-13

### Added

- Residual (Prediction - Actual) visulization distribution bars.

### Known Issues

- Note that during the distribution statistics, must mask the ocean. The land is "1" while the ocean is "0".

## [1.0.0] - 2026-01-14

### Known Issues

- The 100-500 is the potential fine-tuning steps in stage 2.

## [1.0.0] - 2026-01-15

### Added

- Add physical constraint loss for leaving the continental NDVI ranges between 0 and 1.

### Known Issues

- Performance improved 13.4%.

## [1.0.0] - 2026-01-16

### Added

- Add TimeSpaceAdapter.

## [1.0.0] - 2026-01-17

### Added

- Fixed some bugs.
- Add multi-window TimeSpaceAdapter.

## [1.0.0] - 2026-01-18

### Added

- Fixed some bugs.
- Add dynamic training method.
- Modify new versions for converting tif file to npy file.

### Known Issues

- tif2npy_version_1.py aim to process tif data named by like 200111 (YYYYMM).
- tif2npy_version_2.py aim to process tif data named by like 2001-11 (YYYY-MM).

## [1.0.0] - 2026-01-24

### Added

- Add .bat manuscript pairs (tif2npy_batch.bat / tif2npy_batch.py) to automatically convert .tif files to .npy files.
- Add tif2npy_version_3_batch.py manuscript to automatically convert .tif files to .npy files.

### Known Issues

- On Window platform, the **Path** and **Environment** often take errors. We do not recommend using **.bat**.
- We recommend using **tif2npy_version_3_batch.py**.

## [1.0.0] - 2026-02-03

### Added

- Add `run_inference_with_multi_history_v2()` function in `utils/inference.py`.
  - Support custom finetune time period (e.g., `finetune_start_date="198001"`, `finetune_end_date="200101"`).
  - Auto-detect NDVI label availability: finetune + evaluate if label exists, direct prediction if not.
  - Only keep the last finetuned weights after all batches processed.

### Usage Example

```python
from utils.inference import run_inference_with_multi_history_v2

predictions, file_paths = run_inference_with_multi_history_v2(
    model=model,
    dataloader=test_loader,
    device=device,
    output_dir="results/finetune_1980_2001",
    adapter=adapter,
    finetune_start_date="198001",
    finetune_end_date="200101",
    save_adapter=True
)
```