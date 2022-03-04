from nnunet.preprocessing.preprocessing import *

class GenericPreprocessor_multilabel(GenericPreprocessor):
    def __init__(self, normalization_scheme_per_modality, use_nonzero_mask, transpose_forward: (tuple, list), num_modalities_label, intensityproperties=None):
        """

        :param normalization_scheme_per_modality: dict {0:'nonCT'}
        :param use_nonzero_mask: {0:False}
        :param intensityproperties:
        """
        self.transpose_forward = transpose_forward
        self.intensityproperties = intensityproperties
        self.normalization_scheme_per_modality = normalization_scheme_per_modality
        self.num_modalities_label = num_modalities_label
        self.use_nonzero_mask = use_nonzero_mask

        self.resample_separate_z_anisotropy_threshold = RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD

    @staticmethod
    def load_cropped(cropped_output_dir, case_identifier, num_modalities_label):
        all_data = np.load(os.path.join(cropped_output_dir, "%s.npz" % case_identifier))['data']
        data = all_data[:-num_modalities_label].astype(np.float32)
        seg = all_data[-num_modalities_label:]
        with open(os.path.join(cropped_output_dir, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return data, seg, properties

    def _run_internal(self, target_spacing, case_identifier, output_folder_stage, cropped_output_dir, force_separate_z,
                      all_classes, num_modalities_label):
        data, seg, properties = self.load_cropped(cropped_output_dir, case_identifier, num_modalities_label)

        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(data, target_spacing,
                                                            properties, seg, force_separate_z)

        all_data = np.vstack((data, seg)).astype(np.float32)

        # we need to find out where the classes are and sample some random locations
        # let's do 10.000 samples per class
        # seed this for reproducibility!
        num_samples = 10000
        min_percent_coverage = 0.01 # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
        rndst = np.random.RandomState(1234)
        class_locs = {}
        for c in all_classes:
            all_locs = np.argwhere(all_data[-num_modalities_label] == c)
            if len(all_locs) == 0:
                class_locs[c] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[c] = selected
            print(c, target_num_samples)
        properties['class_locations'] = class_locs

        print("saving: ", os.path.join(output_folder_stage, "%s.npz" % case_identifier))
        np.savez_compressed(os.path.join(output_folder_stage, "%s.npz" % case_identifier),
                            data=all_data.astype(np.float32))
        with open(os.path.join(output_folder_stage, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        """

        :param target_spacings: list of lists [[1.25, 1.25, 5]]
        :param input_folder_with_cropped_npz: dim: c, x, y, z | npz_file['data'] np.savez_compressed(fname.npz, data=arr)
        :param output_folder:
        :param num_threads:
        :param force_separate_z: None
        :return:
        """
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
        maybe_mkdir_p(output_folder)
        num_stages = len(target_spacings)
        if not isinstance(num_threads, (list, tuple, np.ndarray)):
            num_threads = [num_threads] * num_stages

        assert len(num_threads) == num_stages

        # we need to know which classes are present in this dataset so that we can precompute where these classes are
        # located. This is needed for oversampling foreground
        all_classes = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']

        for i in range(num_stages):
            all_args = []
            output_folder_stage = os.path.join(output_folder, data_identifier + "_stage%d" % i)
            maybe_mkdir_p(output_folder_stage)
            spacing = target_spacings[i]
            for j, case in enumerate(list_of_cropped_npz_files):
                case_identifier = get_case_identifier_from_npz(case)
                args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes, self.num_modalities_label # changed
                all_args.append(args)
            p = Pool(num_threads[i])
            p.starmap(self._run_internal, all_args)
            p.close()
            p.join()

class PreprocessorFor2D_multilabel(GenericPreprocessor_multilabel):
    def __init__(self, normalization_scheme_per_modality, use_nonzero_mask, transpose_forward: (tuple, list), num_modalities_label, intensityproperties=None):
        super(PreprocessorFor2D_multilabel, self).__init__(normalization_scheme_per_modality, use_nonzero_mask,
                                                transpose_forward, num_modalities_label, intensityproperties)

    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
        assert len(list_of_cropped_npz_files) != 0, "set list of files first"
        maybe_mkdir_p(output_folder)
        all_args = []
        num_stages = len(target_spacings)

        # we need to know which classes are present in this dataset so that we can precompute where these classes are
        # located. This is needed for oversampling foreground
        all_classes = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']

        for i in range(num_stages):
            output_folder_stage = os.path.join(output_folder, data_identifier + "_stage%d" % i)
            maybe_mkdir_p(output_folder_stage)
            spacing = target_spacings[i]
            for j, case in enumerate(list_of_cropped_npz_files):
                case_identifier = get_case_identifier_from_npz(case)
                args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes, self.num_modalities_label
                all_args.append(args)
        p = Pool(num_threads)
        p.starmap(self._run_internal, all_args)
        p.close()
        p.join()

    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }
        target_spacing[0] = original_spacing_transposed[0]
        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        print("normalization...")

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
        print("normalization done")
        return data, seg, properties