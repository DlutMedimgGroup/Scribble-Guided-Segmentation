# 每x个epoch优化一次

import torch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast

class nnUNetTrainerV2_obs(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 obs_size=1):
        super(nnUNetTrainerV2_obs, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                                unpack_data, deterministic, fp16)
        self.obs_size = obs_size # optimizer batch size
        if self.obs_size < 1:
            self.obs_size = 1
        self.crnt_obs_index = 0

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        if self.crnt_obs_index == self.obs_size:
            self.optimizer.zero_grad()
            self.crnt_obs_index = 0
        self.crnt_obs_index = self.crnt_obs_index + 1

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                if self.crnt_obs_index == self.obs_size:
                    self.amp_grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                if self.crnt_obs_index == self.obs_size:
                    self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()
