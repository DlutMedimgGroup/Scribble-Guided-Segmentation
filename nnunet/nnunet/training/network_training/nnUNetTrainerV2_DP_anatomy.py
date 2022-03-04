
# 用于研究Unet特性实验的训练器
# 多GPU


import numpy as np
import random
from sklearn.model_selection import KFold
from collections import OrderedDict
import torch
from torch import nn
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.network_training.nnUNetTrainerV2_DP import nnUNetTrainerV2_DP
from nnunet.network_architecture.my_UNet_DP import my_UNet_DP
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from typing import Union, Tuple, List


class nnUNetTrainerV2_DP_anatomy(nnUNetTrainerV2_DP):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                unpack_data=True, deterministic=True, num_gpus=1, distribute_batch_size=False, fp16=False,
                retrain=False, reinit=True, retrain_layer="", init_model="", limited_tr_keys=0, limited_tr_keys_mod='f'):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                                        unpack_data, deterministic, num_gpus, distribute_batch_size, fp16)
        self.retrain = retrain
        self.reinit = reinit
        self.retrain_layer = retrain_layer
        self.init_model = init_model
        self.limited_tr_keys = limited_tr_keys
        self.limited_tr_keys_mod = limited_tr_keys_mod

    def initialize_network(self):
        """
        replace genericUNet with the implementation of above for super speeds
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = my_UNet_DP(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        if self.retrain:
            # 导入预训练模型
            self.load_checkpoint_to_retrain(self.init_model, train=True)
            # 设置再训练的层
            self.network.set_retrain_layer(self.retrain_layer, self.reinit)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None
        
    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        if not isfile(splits_file):
            self.print_to_log_file("Creating new split...")
            splits = []
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
            if self.limited_tr_keys_mod == 'f':
                tr_keys = tr_keys[0:self.limited_tr_keys]
            elif self.limited_tr_keys_mod == 'r':
                random.shuffle(tr_keys)
                tr_keys = tr_keys[0:self.limited_tr_keys]
            else:
                assert False, "limited_tr_keys_mod is illegal."

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def load_checkpoint_to_retrain(self, fname, train=True):
        self.print_to_log_file("loading checkpoint to retrain", fname, "train=", train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        checkpoint = torch.load(fname, map_location=torch.device('cpu'))

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.network.load_state_dict(new_state_dict)
        self.epoch = 0
        self._maybe_init_amp()

