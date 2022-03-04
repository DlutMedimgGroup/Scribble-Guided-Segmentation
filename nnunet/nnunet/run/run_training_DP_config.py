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
    limited_tr_keys_mod = config.get('Task', 'limited_tr_keys_mod')
    
    dbs = config.getboolean('Task', 'dbs')
    npz = config.getboolean('Task', 'npz')
    valbest = config.getboolean('Task', 'valbest')
    find_lr = config.getboolean('Task', 'find_lr')
    fp32 = config.getboolean('Task', 'fp32')
    disable_saving = config.getboolean('Task', 'disable_saving')
    fold = config.get('Task', 'fold')

    task_name_suffix = config.get('Task', 'task_name_suffix')

    if not task.startswith("Task"):
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
        reinit = config.getboolean('Retrain', 'reinit')
        retrain_layer = config.get('Retrain', 'retrain_layer')
    else:
        init_model = 0
        reinit = 0
        retrain_layer = 0


    if trainer_class is None:
        raise RuntimeError("Could not find trainer class")

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, nnUNetTrainerCascadeFullRes), "If running 3d_cascade_fullres then your " \
                                                                       "trainer class must be derived from " \
                                                                       "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class, nnUNetTrainer), "network_trainer was found but is not derived from " \
                                                         "nnUNetTrainer"

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name,
                            dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,
                            unpack_data=decompress_data, deterministic=deterministic,
                            distribute_batch_size=dbs, num_gpus=num_gpus, fp16=not fp32,
                            retrain=retrain, reinit=reinit, retrain_layer=retrain_layer, init_model=init_model, 
                            limited_tr_keys=limited_tr_keys, limited_tr_keys_mod=limited_tr_keys_mod)

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

        # predict validation
        trainer.validate(save_softmax=npz, validation_folder_name=val_folder,
                         run_postprocessing_on_folds=not disable_postprocessing_on_folds)

        if network == '3d_lowres':
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":

    main()