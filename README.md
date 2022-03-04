# Scribble-Guided-Segmentation

To alleviate the high demand of human annotation for training medical image segmentation networks, we introduce human guidance into the learning process through a scribble-supervised training method without whole-image labeling. The segmentation network is initially trained with a small amount of fully-annotated data and then used to produce pseudo labels for many un-annotated data. Human supervisors draw scribbles at the locations of incorrect pseudo labels. The scribbles are converted into full pseudo label maps using a geodesic transform method. A confidence map of the pseudo labels is also generated according to the inter-pixel geodesic distance and the network output probability. Through an iterative process, the network model and the pseudo label maps are optimized alternately, while human proofreading is only needed in the first iteration.

## Table of Contents
- [Scribble-Guided-Segmentation](#scribble-guided-segmentation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Training of Preliminary Nerwork](#training-of-preliminary-nerwork)
  - [Annotating the Scribbles](#annotating-the-scribbles)
  - [Fine-tuning for the Final Model](#fine-tuning-for-the-final-model)

## Installation

The segmentation network part of this project is implemented based on the [nnU-Net framework](https://github.com/MIC-DKFZ/nnunet), but it is recommended that you clone this library directly to avoid the errors caused by nnUNet version changes. This project has been tested on Ubuntu (Ubuntu 20.04; Intel Core i9-10900KF; NVIDIA RTX 3090; 128G RAM). 

After cloning this project to a Linux device, please follow the [instructions of nnU-Net](https://github.com/MIC-DKFZ/nnunet) to install and configure it. If you are new to nnUNet, it is highly recommended that you read the documentations for nnUNet carefully.

The code for the geodesic transformation part is implemented in C/C++ to increase the speed. Both the source code and the compiled SO link library are given. If you need to debug or make changes to it, then you also need to configure the compilation environment. (*Note that the following steps are not necessary if you do not want to compile the SO link library by yourself.*)
1. [Qt](https://www.qt.io/) is used in the geodesic transformation programme, so first you have to install the Qt framework. My version is `Qt Creator 5.0.0 Based on Qt 5.15.2 (GCC 7.3.1 20180303 (Red Hat 7.3.1-5), 64 bit)`. Not strict on version requirements.
2. The required [pybind11](https://github.com/pybind/pybind11) is already stored by me under the geodesic_distance path, no additional manipulation is needed.
3. Open the project `./geodesic_distance/CMakeLists.txt` in Qt Creator. Generate the release compile and get the output `GeodesicDis.cpython-38-x86_64-linux-gnu.so`. Place this SO file under `nnunet`. Here is already the version I compiled, you can replace it.

## Training of Preliminary Nerwork


## Annotating the Scribbles


## Fine-tuning for the Final Model