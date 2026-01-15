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