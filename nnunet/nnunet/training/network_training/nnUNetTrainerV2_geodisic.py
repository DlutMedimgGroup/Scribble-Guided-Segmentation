from _warnings import warn
from collections import OrderedDict
from numpy.lib.function_base import delete
from tqdm import trange
from time import time, sleep
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.network_training.nnUNetTrainerV2_anatomy import nnUNetTrainerV2_anatomy
from nnunet.training.dataloading.dataset_loading_multilabel import DataLoader3D_Geodesic
from nnunet.training.model_restore import restore_model
from multiprocessing import Pool
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti
from nnunet.experiment_planning.multilabel.preprocessing_multilabel import GenericPreprocessor_multilabel

from nnunet.training.loss_functions.soft_dice_loss import soft_DC_and_CE_loss
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.network_architecture.neural_network import SegmentationNetwork
import SimpleITK as sitk
import GeodesicDis
from nnunet.models.multi_threaded_geodesic import MultiThreadedGeodesic
from tqdm import tqdm

import psutil

class nnUNetTrainerV2_geodisic(nnUNetTrainerV2_anatomy):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, 
                 unpack_data=True, deterministic=True, fp16=False, retrain=False, reinit=True, retrain_layer="", 
                 init_model="", limited_tr_keys=0, fold_mod='r', init_pkl="", cache_path="", validation_path="None"):
        if retrain == True:
            info = load_pickle(init_pkl)
            self._plan_file_init = info['plans']
        super(nnUNetTrainerV2_geodisic, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                                unpack_data, deterministic, fp16, retrain, reinit, retrain_layer, init_model,
                                                limited_tr_keys, fold_mod)
        self.cache_path = cache_path
        self.ignore_label = 99
        # self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, ignore_label=self.ignore_label)
        self.loss = soft_DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.validation_path = validation_path

        self.initial_lr = 1e-3 # 降低微调时的初始学习率
        self.max_num_epochs = 300
        

    def _refresh_geodesic_distence(self):
        print('start to calculate geodesic...')
        geodesic_time_1 = time()
        self._inference_for_geodisic()
        geodesic_time_2 = time()
        self.print_to_log_file("Geodesic took %f s\n" % (geodesic_time_2 - geodesic_time_1))

    def _run_training_network_trainer(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        # _ = self.tr_gen.next()
        # _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)

            inference_start_time = time()

            # !!! inference for geodisic !!!
            if self.epoch <= 100 and (self.epoch) % 10 == 0:
            # if self.epoch <= 100 and (self.epoch) % 10 == 0 and self.epoch != 0:
                self._refresh_geodesic_distence()
            elif self.epoch > 100 and (self.epoch) % 20 == 0:
                self._refresh_geodesic_distence()

            epoch_start_time = time()
            self.print_to_log_file("This inference took %f s\n" % (epoch_start_time - inference_start_time))

            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break
 
            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))
    
    def _run_training_trainer(self):
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil

        shutil.copy(self.plans_file, join(self.output_folder_base, "plans.pkl"))

        self._run_training_network_trainer()

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = self._run_training_trainer()
        self.network.do_ds = ds
        return ret

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)
        self.plans['plans_per_stage'] = self._plan_file_init['plans_per_stage']


    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        num_modalities_label = len(self.plans['modalities_label'])
        all_seg_labels = [0] + self.classes
        if self.threeD:
            dl_tr = DataLoader3D_Geodesic(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size, self, self.ignore_label, all_seg_labels,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', label_mod=num_modalities_label)
            dl_val = DataLoader3D_Geodesic(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, self, self.ignore_label, all_seg_labels, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', label_mod=num_modalities_label)
        else:
            raise("nnUNetTrainer_geodisic doesn't support 2D data.")

        return dl_tr, dl_val

    # def initialize_optimizer_and_scheduler(self):
    #     assert self.network is not None, "self.initialize_network must be called first"
    #     self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
    #                                      momentum=0.99, nesterov=True)
    #     self.lr_scheduler = None

    def _inference_for_geodisic(self):
        num_modalities_label = len(self.plans['modalities_label'])

        GeodesicCalculater = MultiThreadedGeodesic()
        GeodesicCalculater.start()
        
        # for input_data in tqdm(self.dataset_tr):
        for input_data in tqdm(['liver_21', 'liver_25']):

            # T1 = time()
            case_all_data = np.load(self.dataset_tr[input_data]['data_file'][:-4] + ".npy", "r")
            # pkl = load_pickle(self.dataset_tr[input_data]['data_file'][:-4] + ".pkl")
            # case_all_data = np.load(self.dataset_tr[input_data]['data_file'][:-4] + ".npy")
            
            if case_all_data[1, 0, 0, 0] == -1:
                # T2 = time()
                # print('np.load 时间:%s毫秒' % ((T2 - T1)*1000))

                output = self.predict_preprocessed_data_return_seg_and_softmax(case_all_data[:-num_modalities_label, :, :, :], do_mirroring=False, mirror_axes=(0,1,2), step_size=0.5, verbose=False)[1]

                # T3 = time()
                # print('predict 时间:%s毫秒' % ((T3 - T2)*1000))

                transpose_forward = self.plans.get('transpose_forward')
                if transpose_forward is not None:
                    transpose_backward = self.plans.get('transpose_backward')
                    output = output.transpose([0] + [i + 1 for i in transpose_backward])
                # T4 = time()
                # print('transpose 时间:%s毫秒' % ((T4 - T3)*1000))
                
                properties = {'data_file_name': self.dataset_tr[input_data]['data_file'], 'input_data_name': input_data, 'epoch_int': int(self.epoch), 'cache_path': self.cache_path, 'num_modalities_label': num_modalities_label}
                item = [case_all_data, output, properties]
                GeodesicCalculater.add_result(item)
                # T5 = time()
                # print('add_result 时间:%s毫秒' % ((T5 - T4)*1000))
                
                # # Output Intermediate File
                # originlabel_filename = self.dataset_tr[input_data]['data_file'][:-4] + "-originlabel.npy"
                # epoch_int = int(self.epoch)                
                # if epoch_int == 0:
                #     originlabel = output.argmax(0).astype(np.int16)                    
                #     np.save(originlabel_filename, originlabel)
                #     # debug
                #     maybe_mkdir_p(self.cache_path)
                #     image_input_filename = join(self.cache_path, str(epoch_int) + "-" + input_data + "-image.nii.gz")
                #     image_image = sitk.GetImageFromArray(input_array[0, :, :, :])
                #     sitk.WriteImage(image_image, image_input_filename)
                # else:
                #     originlabel = np.load(originlabel_filename, "r")
                # netout_filename = join(self.cache_path, str(epoch_int) + "-" + input_data + "-netout.nii.gz")
                # netout_image = sitk.GetImageFromArray(output.argmax(0).astype(np.int16))
                # sitk.WriteImage(netout_image, netout_filename)

                # # Geodesic Distance
                # generater = GeodesicDis.GeoDisScibble()
                # generater.SetInputImage((input_array[0, :, :, :]))
                # generater.SetInputSeedMap(case_all_data[-1, :, :, :].astype(np.int))
                # generater.SetOriginLabelMap(originlabel)
                # generater.SetSpacing([1, 1, 1])
                # generater.SetOrigin([0, 0, 0])
                # generater.SetPropertyMap(output)
                # generater.SetProperties(0.01, 1, 0.1, 5)
                # generater.SetSortPeriod(10000)
                # generater.SetIgnoreLabel(self.ignore_label)
                # generater.DebugOff()
                # generater.Generate()

                # # output
                # fakelabel = np.empty([2]+list(input_array.shape[1:]), dtype=np.int16)
                # fakelabel[0] = generater.GetToughLabelMap()
                # fakelabel[1] = generater.GetConfidenceMap()
                # output_filename = self.dataset_tr[input_data]['data_file'][:-4] + "-fakelabel.npy"
                # np.save(output_filename, fakelabel)
                # focus = np.argwhere(fakelabel[1] == 1000)
                # focus_filename = self.dataset_tr[input_data]['data_file'][:-4] + "-focus.npy"
                # np.save(focus_filename, focus)

                # # debug
                # fakelabel_filename = join(self.cache_path, str(epoch_int) + "-" + input_data + "-fakelabel.nii.gz")
                # toughlabel_image = sitk.GetImageFromArray(fakelabel[0])
                # sitk.WriteImage(toughlabel_image, fakelabel_filename)
                # confidence_filename = join(self.cache_path, str(epoch_int) + "-" + input_data + "-confidence.nii.gz")
                # confidence_image = sitk.GetImageFromArray(fakelabel[1])
                # sitk.WriteImage(confidence_image, confidence_filename)

                # del generater
            
        GeodesicCalculater.wait_finish()
        GeodesicCalculater.stop()
        del GeodesicCalculater

    def preprocess_patient(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        from nnunet.training.model_restore import recursive_find_python_class
        preprocessor_name = self.plans.get('preprocessor_name')
        if preprocessor_name is None:
            if self.threeD:
                preprocessor_name = "GenericPreprocessor"
            else:
                preprocessor_name = "PreprocessorFor2D"

        print("using preprocessor", preprocessor_name)
        if preprocessor_name == 'GenericPreprocessor_multilabel':
            preprocessor_class = GenericPreprocessor_multilabel
        else:
            raise 'not found  preprocessor'
        assert preprocessor_class is not None, "Could not find preprocessor %s in nnunet.preprocessing" % \
                                               preprocessor_name
        preprocessor = preprocessor_class(self.normalization_schemes, self.use_mask_for_norm,
                                          self.transpose_forward, self.num_modalities_label, self.intensity_properties)
        self.num_classes
        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'])
        return d, s, properties

    def process_plans(self, plans):
        super(nnUNetTrainerV2_geodisic, self).process_plans(plans)
        self.num_modalities_label = len(plans['modalities_label'])

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        splits = []
        if self.validation_path != "None":
            splits_reference = load_pickle(self.validation_path)
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            num_folds = 3
            num_keys = len(all_keys_sorted)
            for i in range(num_folds):
                train_keys = []
                test_keys = splits_reference[i]['val']
                for key in all_keys_sorted:
                    if key not in test_keys:
                        train_keys.append(key)
                train_keys = np.array(train_keys)
                test_keys = np.array(test_keys)
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
        else:
            # 随机分配每一折的数据
            self.print_to_log_file("Creating new split (redom)...")
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
        save_pickle(splits, splits_file)

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        if self.limited_tr_keys > 0:
            tr_keys = tr_keys[0:self.limited_tr_keys]

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
    
    def initialize(self, training=True, force_load_plans=False):
        """
        和 nnUNetTrainer_V2 的唯一区别就是将 get_moreDA_augmentation 的 order_seg 参数设置为 0. 
        这影响了数据扩增 SpatialTransform > augment_spatial > modified_coords 中，对标签重采样时所使用的插值方法
        我们使用软标签，所以将 order_seg 设置为 0 来避免结果被四舍五入到整数
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val, 
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    order_seg=0   # !!! was changed from 1 to 0 for soft label
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        # !!! selected_seg_channels was changed for soft label !!!
        self.data_aug_params['selected_seg_channels'] = None
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        # stronger gpu for softlabel augmentation
        self.data_aug_params["num_cached_per_thread"] = 2
        self.data_aug_params["num_threads"] = 16
        
