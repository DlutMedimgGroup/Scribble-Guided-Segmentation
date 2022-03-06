# Scribble-Guided-Segmentation

To alleviate the high demand of human annotation for training medical image segmentation networks, we introduce human guidance into the learning process through a scribble-supervised training method without whole-image labeling. The segmentation network is initially trained with a small amount of fully-annotated data and then used to produce pseudo labels for many un-annotated data. Human supervisors draw scribbles at the locations of incorrect pseudo labels. The scribbles are converted into full pseudo label maps using a geodesic transform method. A confidence map of the pseudo labels is also generated according to the inter-pixel geodesic distance and the network output probability. Through an iterative process, the network model and the pseudo label maps are optimized alternately, while human proofreading is only needed in the first iteration.

*All example data and trained models can be downloaded here ``. In the following, 'example_data' is used to refer to the data downloaded here. In order to use this data, 'Task03_Liver' in the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset also needs to be downloaded.*

## Table of Contents
- [Scribble-Guided-Segmentation](#scribble-guided-segmentation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Training of Preliminary Network](#training-of-preliminary-network)
  - [Annotating the Scribbles](#annotating-the-scribbles)
  - [Fine-tuning for the Final Model](#fine-tuning-for-the-final-model)

## Installation

The segmentation network part of this project is implemented based on the [nnU-Net framework](https://github.com/MIC-DKFZ/nnunet), but it is recommended that you clone this library directly to avoid the errors caused by nnUNet version changes. This project has been tested on Ubuntu (Ubuntu 20.04; Intel Core i9-10900KF; NVIDIA RTX 3090; 128G RAM). 

After cloning this project to a Linux device, please follow the [instructions of nnU-Net](https://github.com/MIC-DKFZ/nnunet) to install and configure it. If you are new to nnUNet, it is highly recommended that you read the documentations for nnUNet carefully.

The code for the geodesic transformation part is implemented in C/C++ to increase the speed. Both the source code and the compiled SO link library are given. If you need to debug or make changes to it, then you also need to configure the compilation environment. (*Note that the following steps are not necessary if you do not want to compile the SO link library by yourself.*)
1. [Qt](https://www.qt.io/) is used in the geodesic transformation programme, so first you have to install the Qt framework. My version is `Qt Creator 5.0.0 Based on Qt 5.15.2 (GCC 7.3.1 20180303 (Red Hat 7.3.1-5), 64 bit)`. Not strict on version requirements.
2. Run `git clone https://github.com/pybind/pybind11.git` in the `geodesic_distance` folder.
3. Open the project `./geodesic_distance/CMakeLists.txt` in Qt Creator. Generate the release compile and get the output `GeodesicDis.cpython-38-x86_64-linux-gnu.so`. Place this SO file under `nnunet`. Here is already the version I compiled, you can replace it.

## Training of Preliminary Network
The first step of our pipeline is the training of preliminary network with a small fully annotated training set. Since only a small amount of data needs to be annotated, the annotation time is completely controllable. The annotation software tools are not restricted. 

Image data and annotated labels need to be placed in the `nnUNet_raw_data` folder created during the configuring of nnU-Net environment. 

`example_data` contains an example of training a preliminary network. Copy `example_data\Task11_Liver` to `nnUNet_raw_data`. To save download time, `example_data\Task11_Liver\imagesTr` is empty and you need to copy the CT files corresponding to `example_data\Task11_Liver\labelsTr` from downloaded Task03_Liver dataset to `example_data\Task11_Liver\imagesTr`.

To convert the dataset, run:
```
nnUNet_convert_decathlon_task -i /path_to_nnUNet_raw/nnUNet_raw_data/Task11_Liver -output_task_id 311
```
To run the preprocessor, run:
```
nnUNet_plan_and_preprocess -t 311
```
Unlike the original nnU-Net, the some of the preprocessing and training procedures are not run by command. Various operations are implemented through python files and configuration files. To train the preliminary network, run the following command in path `./nnunet/nnunet/run`.
```
python run_training_config.py -c geodesic_dis/small_311
```
The configurations used during training are included in `./nnunet/nnunet/Config_files/geodesic_dis/small_311`, which can be modified on demand.

After the training is completed, the model with the name `model_final_checkpoint.model` will be saved in `nnUNet_trained_models/nnUNet/3d_fullres/Task311_Liver/nnUNetTrainerV2_anatomy__nnUNetPlansv2.1_small_311/fold_0`. The preliminary network model that has been trained is given in `example_data/models/preliminary/`

Copy the CT files in downloaded Task03_Liver dataset with serial numbers ending with 4 and 9 to `example_data/inference/input_test` and rename the files to end with "_0000", e.g. "liver_4_0000.nii.gz", "liver_9_0000.nii.gz" and "liver_14_0000.nii.gz". Copy the CT files with serial numbers not ending with 4 and 9 to `example_data/inference/input_scribble` and rename the files to end with "_0000", e.g. "liver_0_0000.nii.gz", "liver_1_0000.nii.gz" and "liver_2_0000.nii.gz". 

To run the inference for scribble annotation data, run the following command in path `./nnunet/nnunet/inference`. Before running, change `input_folder` and `output_folder` in `./nnunet/nnunet/Config_files/geodesic_dis/small_311` to the path of `example_data/inference/input_scribble` and `example_data/inference/output_scribble`, respectively.
```
python predict_simple_config.py -c geodesic_dis/small_311
```
Change `input_folder` and `output_folder` in `./nnunet/nnunet/Config_files/geodesic_dis/small_311` to the path of `example_data/inference/input_test` and `example_data/inference/output_test_small` and run the commend again to run the inference for test set.

## Annotating the Scribbles

The labels in `example_data/inference/output_scribble` are the output of preliminary network. These results contain errors, due to the limited training set. Next, we need to annotate scribbles for these errors for subsequent fine-tuning. I used [AnatomyScktch software](https://github.com/DlutMedimgGroup/AnatomySketch-Software) for scribble annotation. It is possible to use other tools.

Examples of preliminary network outputs and scribble annotations are placed in `example_data/annotation`. `*.seed` is a special file formats of AnatomyScktch to represent scribbles. It is essentially a structured text file. You can also use the label maps to represent scribbles, but it requires some adjustments to the subsequent procedures. If this place is causing you trouble, feel free to ask me for advice.

## Fine-tuning for the Final Model

The procedure for fine tuning is very similar to the training. Image data, annotated labels and scribbles need to be placed in the `nnUNet_raw_data` folder. 

An example is given in `example_data/Scribble11_Liver`. Copy all ct images from Task03_Liver dataset to `example_data/Scribble11_Liver/imagesTr`. Then, you can copy `example_data/Scribble11_Liver` to `nnUNet_raw_data` directly or imitate it to build your own data folder.

To convert the dataset, run the following command in path `./nnunet/nnunet/experiment_planning/convert_data_config`. Some paths need to be adjusted to yours in the configuration file `./nnunet/nnunet/Config_files/geodesic_dis/geodesic_iteration_311`. Set `source_folder` to the path of `Scribble11_Liver`; set `plan_file_to_load` to the plan file generated by preliminary Network.
```
python convert_scribble_task_config.py -c geodesic_dis/geodesic_iteration_311
```
To run the preprocessor, run:
```
python plan_and_preprocess_multi_label.py -c geodesic_dis/geodesic_iteration_311
```
To train the preliminary network, run the following command in path `./nnunet/nnunet/run`.
```
python run_training_geodesic_config.py -c geodesic_dis/geodesic_iteration_311
```
To run the inference for test data and test data, run the following command in path `./nnunet/nnunet/inference`. Before running, change `input_folder` and `output_folder` in `./nnunet/nnunet/Config_files/geodesic_dis/geodesic_iteration_311` to the path of `example_data/inference/input_test` and `example_data/inference/output_test_large`, respectively.
```
python predict_simple_geodesic_config.py -c geodesic_dis/geodesic_iteration_311
```