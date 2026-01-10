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

### Known Issues

- NDVI ranges between 0 and 1. Denormalization is not a must.