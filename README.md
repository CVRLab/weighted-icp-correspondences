# Weighted ICP Correspondences

This repository holds the supplementary materials for the paper "*A mini-map alignment approach using landmark observation information intrinsic to the feature-based Visual SLAM pipeline*" in XIX Workshop de Visão Computacional 2024 - XIX WVC2024, Campus Rio Paranaíba da Universidade Federal de Viçosa - UFV-CRP.

## Important files

- Table of the individual scores and number of iterations obtained by each experiment as [PDF](./experiments_data.pdf) or [spreadsheet](./experiments_data.ods).
- Code to run the experiments described in the paper: [alignment_experiments.cpp](./alignment_experiments.cpp)
- Header file that implements the proposed modification to the ICP correspondence estimation step to be used in other PCL projects: [weighted_correspondence_estimation.h](./weighted_correspondence_estimation.h)

## Experiments build instructions

1. Install MRPT, OpenCV and PCL dependencies (may need sudo):

```shell
./install_dependencies.sh
```

2. Build:

```shell
./build.sh
```

2. Run experiments:
```shell
./run_experiments.sh
```

## Citation

> SARAIVA, F. P.; LAUREANO, G. T.; OLIVEIRA, T. H. de. **A mini-map alignment approach using landmark observation information intrinsic to the feature-based Visual SLAM pipeline**. In: XIX Workshop de Visão Computacional 2024 - XIX WVC2024, Campus Rio Paranaíba da Universidade Federal de Viçosa - UFV-CRP
