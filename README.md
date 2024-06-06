# Split-Lohmann in Pytorch

This is the pytorch implementation of [Split-Lohmann Multifocal Displays](https://imaging.cs.cmu.edu/split_lohmann/). For the simulation code in matlab, please refer to [our original code release repo](https://github.com/Image-Science-Lab-cmu/SplitLohmann).

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

## Example Focal Stack Result

The focal stack result below was generated using `32` propagation iterations; each initialized with a wavefront having a random phase at each point.

To obtain the most realistic output, increase the `num_rounds` number in `simulation.ipynb`.\
Increasing this number will reduce the amount of speckle noise in the focal stack images.

![](results/Whiskey_splitlohmann_result.gif)

## Scene Credits

If you use any of our provided scenes, please also credit the courtesy source of the 3D scene or asset.

- Whiskey scene: "Dark Interior Scene" 3D scene courtesy of “Entity Designer” at Blender Market.
- Motorcycle scene: 3D scene courtesy of the Middlebury 2014 Stereo Dataset [Scharstein et al. 2014].
- CastleCity scene: "Scanlands" 3D scene courtesy of Piotr Krynski at Blender Studio.

## Citation

If you use our code or dataset, please cite our paper:
```
@article{Qin_SplitLohmann,
author = {Qin, Yingsi and Chen, Wei-Yu and O'Toole, Matthew and Sankaranarayanan, Aswin C.},
title = {Split-Lohmann Multifocal Displays},
year = {2023},
issue_date = {August 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3592110},
doi = {10.1145/3592110},
abstract = {This work provides the design of a multifocal display that can create a dense stack of focal planes in a single shot. We achieve this using a novel computational lens that provides spatial selectivity in its focal length, i.e, the lens appears to have different focal lengths across points on a display behind it. This enables a multifocal display via an appropriate selection of the spatially-varying focal length, thereby avoiding time multiplexing techniques that are associated with traditional focus tunable lenses. The idea central to this design is a modification of a Lohmann lens, a focus tunable lens created with two cubic phase plates that translate relative to each other. Using optical relays and a phase spatial light modulator, we replace the physical translation of the cubic plates with an optical one, while simultaneously allowing for different pixels on the display to undergo different amounts of translations and, consequently, different focal lengths. We refer to this design as a Split-Lohmann multifocal display. Split-Lohmann displays provide a large \'{e}tendue as well as high spatial and depth resolutions; the absence of time multiplexing and the extremely light computational footprint for content processing makes it suitable for video and interactive experiences. Using a lab prototype, we show results over a wide range of static, dynamic, and interactive 3D scenes, showcasing high visual quality over a large working range.},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {57},
numpages = {18},
keywords = {multifocal displays, computational displays, vergence-accomodation conflict, lohmann lenses}
}
```

Visit the links below for more information:\
 [[Paper](https://dl.acm.org/doi/abs/10.1145/3592110)] [[Supplemental PDF](https://yingsiqin.github.io/assets/pdfs/SplitLohmann_SIGGRAPH23-supp.pdf)] [[Project Website](https://imaging.cs.cmu.edu/split_lohmann/)]\
 [[6-Minute Video](https://youtu.be/9lbg8qOCjUM)] [[3-Minute Video](https://youtu.be/0Z4W1DJO_nw)] [[10-Minute Talk @ SIGGRAPH 2023](https://youtu.be/1qH6yvEWd5c)]
