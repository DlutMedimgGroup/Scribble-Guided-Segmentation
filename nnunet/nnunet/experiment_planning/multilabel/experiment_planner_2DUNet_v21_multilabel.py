import os
import shutil
import numpy as np
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import *
import nnunet
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21


class ExperimentPlanner2D_v21_multilabel(ExperimentPlanner2D_v21):

    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D_v21_multilabel, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.preprocessor_name = "PreprocessorFor2D_multilabel"

    def plan_experiment(self):
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero maks for normalizaion?", use_nonzero_mask_for_normalization)

        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']
        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))
        modalities_label = self.dataset_properties['modalities_label']  # changed
        num_modalities_label = len(list(modalities_label.keys()))  # changed

        target_spacing = self.get_target_spacing()
        new_shapes = np.array([np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)])

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("we don't want feature maps smaller than ", self.unet_featuremap_min_edge_length, " in the bottleneck")

        # how many stages will the image pyramid have?
        self.plans_per_stage = []

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("the transposed median shape of the dataset is ", median_shape_transposed)

        self.plans_per_stage.append(
            self.get_properties_for_stage(target_spacing_transposed, target_spacing_transposed, median_shape_transposed,
                                          num_cases=len(self.list_of_cropped_npz_files),
                                          num_modalities=num_modalities,
                                          num_classes=len(all_classes) + 1),
            )

        print(self.plans_per_stage)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        normalization_schemes = self.determine_normalization_scheme()
        # deprecated
        only_keep_largest_connected_component, min_size_per_class, min_region_size_per_class = None, None, None

        # these are independent of the stage
        plans = {'num_stages': len(list(self.plans_per_stage.keys())), 'num_modalities': num_modalities,
                 'modalities': modalities, 'modalities_label': modalities_label, 'normalization_schemes': normalization_schemes, # changed
                 'dataset_properties': self.dataset_properties, 'list_of_npz_files': self.list_of_cropped_npz_files,
                 'original_spacings': spacings, 'original_sizes': sizes,
                 'preprocessed_data_folder': self.preprocessed_output_folder, 'num_classes': len(all_classes),
                 'all_classes': all_classes, 'base_num_features': self.unet_base_num_features,
                 'use_mask_for_norm': use_nonzero_mask_for_normalization,
                 'keep_only_largest_region': only_keep_largest_connected_component,
                 'min_region_size_per_class': min_region_size_per_class, 'min_size_per_class': min_size_per_class,
                 'transpose_forward': self.transpose_forward, 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage,
                 'preprocessor_name': self.preprocessor_name,
                 }

        self.plans = plans
        self.save_my_plans()

    def load_experiment(self, file_name):
        with open(file_name, 'rb') as f:
            properties = pickle.load(f)
        self.plans['dataset_properties']['intensityproperties'] = properties['dataset_properties']['intensityproperties']

    def run_preprocessing(self, num_threads):
        if os.path.isdir(join(self.preprocessed_output_folder, "gt_segmentations")):
            shutil.rmtree(join(self.preprocessed_output_folder, "gt_segmentations"))
        shutil.copytree(join(self.folder_with_cropped_data, "gt_segmentations"),
                        join(self.preprocessed_output_folder, "gt_segmentations"))
        normalization_schemes = self.plans['normalization_schemes'] 
        use_nonzero_mask_for_normalization = self.plans['use_mask_for_norm']
        intensityproperties = self.plans['dataset_properties']['intensityproperties']
        num_modalities_label = len(self.plans['modalities_label'].keys()) # changed
        preprocessor_class = recursive_find_python_class([join(nnunet.__path__[0], "experiment_planning/multilabel")], # changed
                                                         self.preprocessor_name, current_module="nnunet.experiment_planning.multilabel") # changed
        assert preprocessor_class is not None
        preprocessor = preprocessor_class(normalization_schemes, use_nonzero_mask_for_normalization,
                                         self.transpose_forward, num_modalities_label, # changed
                                          intensityproperties)
        target_spacings = [i["current_spacing"] for i in self.plans_per_stage.values()]
        if self.plans['num_stages'] > 1 and not isinstance(num_threads, (list, tuple)):
            num_threads = (default_num_threads, num_threads)
        elif self.plans['num_stages'] == 1 and isinstance(num_threads, (list, tuple)):
            num_threads = num_threads[-1]
        preprocessor.run(target_spacings, self.folder_with_cropped_data, self.preprocessed_output_folder,
                         self.plans['data_identifier'], num_threads)