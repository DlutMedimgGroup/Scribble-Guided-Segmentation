import os
import shutil
import argparse
import configparser
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

from multiprocessing import Pool
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

import nnunet
from nnunet.paths import *
from nnunet.configuration import default_num_threads
from nnunet.preprocessing.sanity_checks import *
from nnunet.training.model_restore import recursive_find_python_class

from nnunet.experiment_planning.multilabel.utils import crop_multilabel
from nnunet.experiment_planning.multilabel.DatasetAnalyzer_multilabel import DatasetAnalyzer_multilabel

def main():

    # 运行时通过参数选择配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', help="Path of config file.", required=True)
    args = parser.parse_args()

    # 读取配置文件
    config_file = args.config
    if not config_file.endswith(".ini"):
        config_file = config_file + ".ini"
    config_file = os.path.join("/home/mingrui/disk1/projects/nnUNet/my_nnunet/nnunet/Config_files/", config_file)    
    config = configparser.ConfigParser()
    config.read(config_file)

    # configs
    task_name = config.get('Convert', 'output_task_name')
    plans_folder = config.get('Convert', 'plan_file_to_load')
    plan_file_2D = ''
    plan_file_3D = ''
    if plans_folder != '':
        plan_file_2D = join(plans_folder, 'nnUNetPlansv2.1_plans_2D.pkl')
        plan_file_3D = join(plans_folder, 'nnUNetPlansv2.1_plans_3D.pkl')


    with open(plan_file_3D, 'rb') as f:
        properties = pickle.load(f)

    # crop
    crop_multilabel(task_name, False, num_threads=8)
    
    # planner
    planner_name3d = "ExperimentPlanner3D_v21_multilabel"
    planner_name2d = "ExperimentPlanner2D_v21_multilabel"

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    search_in = join(nnunet.__path__[0], "experiment_planning/multilabel")

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="nnunet.experiment_planning.multilabel")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="nnunet.experiment_planning.multilabel")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name2d)
    else:
        planner_2d = None

    # fingerprint
    print("\n\n\n", task_name)
    cropped_out_dir = os.path.join(nnUNet_cropped_data, task_name)
    preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, task_name)

    # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT.
    dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
    modalities = list(dataset_json["modality"].values())
    collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
    dataset_analyzer = DatasetAnalyzer_multilabel(cropped_out_dir, overwrite=False, num_processes=8)  # this class creates the fingerprint
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

    maybe_mkdir_p(preprocessing_output_dir_this_task)
    shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
    shutil.copy(join(nnUNet_raw_data, task_name, "dataset.json"), preprocessing_output_dir_this_task)

    # preprocess
    tl = 8 # Number of processes used for preprocessing the low resolution data for the 3D low resolution U-Net.
    tf = 8 # Number of processes used for preprocessing the full resolution data of the 2D U-Net and 3D U-Net.
    threads = (tl, tf)
    print("number of threads: ", threads, "\n")

    # Set this flag if you dont want to run the preprocessing. If this is set then this script will only run the experiment planning and create the plans file
    dont_run_preprocessing = False 

    if planner_3d is not None:
        exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        if plan_file_3D != '':
            exp_planner.load_experiment(plan_file_3D)
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads)
    if planner_2d is not None:
        exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        if plan_file_2D != '':
            exp_planner.load_experiment(plan_file_2D)
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads)



if __name__ == "__main__":
    main()
