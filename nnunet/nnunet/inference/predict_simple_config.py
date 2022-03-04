# 通过读配置文件的测试
import os
import argparse
import configparser
import torch
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


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
    input_folder = config.get('Predict', 'input_folder')
    output_folder = config.get('Predict', 'output_folder')
    task = config.get('Task', 'task_id')
    trainer_class_name = config.get('Task', 'network_trainer')
    model = config.get('Task', 'network')
    task_name_suffix = config.get('Task', 'task_name_suffix')
    plans_identifier = config.get('Task', 'plans_identifier')
    folds = config.get('Predict', 'folds')
    save_npz = config.getboolean('Predict', 'save_npz')
    lowres_segmentations = config.get('Predict', 'lowres_segmentations')
    part_id = config.getint('Predict', 'part_id')
    num_parts = config.getint('Predict', 'num_parts')
    num_threads_preprocessing = config.getint('Predict', 'num_threads_preprocessing')
    num_threads_nifti_save = config.getint('Predict', 'num_threads_nifti_save')
    disable_tta = config.getboolean('Predict', 'disable_tta')
    overwrite_existing = config.getboolean('Predict', 'overwrite_existing')
    mode = config.get('Predict', 'mode')
    all_in_gpu = config.get('Predict', 'all_in_gpu')
    step_size = config.getfloat('Predict', 'step_size')
    chk = config.get('Predict', 'chk')
    disable_mixed_precision = config.getboolean('Predict', 'disable_mixed_precision')

    gpus_id = config.get('Predict', 'gpus_id')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_id

    task_id = int(task)
    task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], "model must be 2d, 3d_lowres, 3d_fullres or " \
                                                                             "3d_cascade_fullres"
    if lowres_segmentations == "None":
        lowres_segmentations = None
    
    if folds == "all":
        folds = ["all"]
    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False
    
    if plans_identifier == "default_plans_identifier":
        plans_identifier = default_plans_identifier

    # we need to catch the case where model is 3d cascade fullres and the low resolution folder has not been set.
    # In that case we need to try and predict with 3d low res first
    if model == "3d_cascade_fullres" and lowres_segmentations is None:
        print("lowres_segmentations is None. Attempting to predict 3d_lowres first...")
        assert part_id == 0 and num_parts == 1, "if you don't specify a --lowres_segmentations folder for the " \
                                                "inference of the cascade, custom values for part_id and num_parts " \
                                                "are not supported. If you wish to have multiple parts, please " \
                                                "run the 3d_lowres inference first (separately)"
        model_folder_name = join(network_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" +
                                  plans_identifier)
        assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
        lowres_output_folder = join(output_folder, "3d_lowres_predictions")
        predict_from_folder(model_folder_name, input_folder, lowres_output_folder, folds, False,
                            num_threads_preprocessing, num_threads_nifti_save, None, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size)
        lowres_segmentations = lowres_output_folder
        torch.cuda.empty_cache()
        print("3d_lowres done")

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name

    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" +
                              plans_identifier)

    if task_name_suffix != "":
        model_folder_name = model_folder_name + "_" + task_name_suffix
    
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not disable_mixed_precision,
                        step_size=step_size, checkpoint_name=chk)

if __name__ == "__main__":
    main()
