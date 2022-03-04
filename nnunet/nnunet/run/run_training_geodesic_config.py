# 通过读配置文件的训练器
import os
import argparse
import configparser
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import network_training_output_dir
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes

from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir, nnUNet_cropped_data, network_training_output_dir
import numpy as np

def convert_id_to_task_name(task_id: int):
    startswith = "Scribble%03.0d" % task_id
    if preprocessing_output_dir is not None:
        candidates_preprocessed = subdirs(preprocessing_output_dir, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if nnUNet_raw_data is not None:
        candidates_raw = subdirs(nnUNet_raw_data, prefix=startswith, join=False)
    else:
        candidates_raw = []

    if nnUNet_cropped_data is not None:
        candidates_cropped = subdirs(nnUNet_cropped_data, prefix=startswith, join=False)
    else:
        candidates_cropped = []

    candidates_trained_models = []
    if network_training_output_dir is not None:
        for m in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']:
            if isdir(join(network_training_output_dir, m)):
                candidates_trained_models += subdirs(join(network_training_output_dir, m), prefix=startswith, join=False)

    all_candidates = candidates_cropped + candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one task name found for task id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (task_id, nnUNet_raw_data, preprocessing_output_dir,
                                                               nnUNet_cropped_data))
    if len(unique_candidates) == 0:
        raise RuntimeError("Could not find a task with the ID %d. Make sure the requested task ID exists and that "
                           "nnU-Net knows where raw and preprocessed data are located (see Documentation - "
                           "Installation). Here are your currently defined folders:\nnnUNet_preprocessed=%s\nRESULTS_"
                           "FOLDER=%s\nnnUNet_raw_data_base=%s\nIf something is not right, adapt your environemnt "
                           "variables." %
                           (task_id,
                            os.environ.get('nnUNet_preprocessed') if os.environ.get('nnUNet_preprocessed') is not None else 'None',
                            os.environ.get('RESULTS_FOLDER') if os.environ.get('RESULTS_FOLDER') is not None else 'None',
                            os.environ.get('nnUNet_raw_data_base') if os.environ.get('nnUNet_raw_data_base') is not None else 'None',
                            ))
    return unique_candidates[0]

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
    gpus_id = config.get('Task', 'gpus_id')
    num_gpus = config.getint('Task', 'num_of_gpus')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_id
    task = config.get('Task', 'task_id')
    network = config.get('Task', 'network')
    network_trainer = config.get('Task', 'network_trainer')
    validation_only = config.getboolean('Task', 'validation_only')
    continue_training = config.getboolean('Task', 'continue_training')
    plans_identifier = config.get('Task', 'plans_identifier')
    if plans_identifier == 'default_plans_identifier':
        plans_identifier = default_plans_identifier
    use_compressed_data = config.getboolean('Task', 'use_compressed_data')
    decompress_data = not use_compressed_data
    deterministic = config.getboolean('Task', 'deterministic')
    limited_tr_keys = config.getint('Task', 'limited_tr_keys')
    
    dbs = config.getboolean('Task', 'dbs')
    npz = config.getboolean('Task', 'npz')
    valbest = config.getboolean('Task', 'valbest')
    find_lr = config.getboolean('Task', 'find_lr')
    fp32 = config.getboolean('Task', 'fp32')
    disable_saving = config.getboolean('Task', 'disable_saving')
    fold = config.get('Task', 'fold')
    validation_path = config.get('Task', 'validation_path')

    task_name_suffix = config.get('Task', 'task_name_suffix')
    if not task.startswith("Scribble"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)
    
    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)
    
    if task_name_suffix != "":
        output_folder_name = output_folder_name + "_" + task_name_suffix

    retrain = config.getboolean('Retrain', 'retrain')
    if retrain:
        # 只训练部分层
        init_model = config.get('Retrain', 'init_model')
        init_model = os.path.join(network_training_output_dir, init_model)
        init_pkl = init_model + '.pkl'
        reinit = config.getboolean('Retrain', 'reinit')
        retrain_layer = config.get('Retrain', 'retrain_layer')
    else:
        init_model = 0
        reinit = 0
        retrain_layer = 0

    geodesic_cache = config.get('Task', 'geodesic_cache')

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class")

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class")

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name,
                            dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,
                            unpack_data=decompress_data, deterministic=deterministic, fp16=not fp32,
                            retrain=retrain, reinit=reinit, retrain_layer=retrain_layer, init_model=init_model, 
                            limited_tr_keys=limited_tr_keys, init_pkl=init_pkl, cache_path=geodesic_cache, validation_path=validation_path)

    if disable_saving:
        trainer.save_latest_only = False  # if false it will not store/overwrite _latest but separate files each
        trainer.save_intermediate_checkpoints = False  # whether or not to save checkpoint_latest
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        trainer.save_final_checkpoint = False  # whether or not to save the final checkpoint

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if continue_training:
                trainer.load_latest_checkpoint()
            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)

        trainer.network.eval()

        # # predict validation
        # trainer.validate(save_softmax=npz, validation_folder_name=val_folder,
        #                  run_postprocessing_on_folds=not disable_postprocessing_on_folds)

        if network == '3d_lowres':
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, jo
            in(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
    main()
    