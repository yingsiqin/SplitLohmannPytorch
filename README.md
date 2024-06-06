# Split-Lohmann in Python

This is the python implementation of [Split-Lohmann Multifocal Displays](https://imaging.cs.cmu.edu/split_lohmann/). For the simulation code in matlab, please refer to [our original code release repo](https://github.com/Image-Science-Lab-cmu/SplitLohmann).

## Getting Started

#### Requirements

```
pip install numpy matplotlib
pip3 install torch torchvision torchaudio
pip install pytorch-lightning kornia
```

`simulation.ipynb` walks through the simulation pipeline to generate a focal stack using the Split-Lohmann system.
`time_multiplexed_multifocals.ipynb` walks through the pipeline to generate a focal stack using a time-multiplexed multifocal display.

This documentation for this repo is under development. For complete documentation, please refer to [our original code release repo](https://github.com/Image-Science-Lab-cmu/SplitLohmann).