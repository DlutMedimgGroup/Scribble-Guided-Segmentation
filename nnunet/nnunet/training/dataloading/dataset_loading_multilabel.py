from nnunet.training.dataloading.dataset_loading import *
from nnunet.utilities.one_hot_encoding import to_one_hot

class DataLoader3D_multilabel(DataLoader3D):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None, label_mod = 1):

        self._num_seg = label_mod
        super(DataLoader3D_multilabel, self).__init__(data, patch_size, final_patch_size, batch_size, has_prev_stage, oversample_foreground_percent,
                                                      memmap_mode, pad_mode, pad_kwargs_data, pad_sides)
        

    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = self._num_seg + 1
        else:
            num_seg = self._num_seg

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - self._num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
            # which is much faster to access
            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            # If we are doing the cascade then we will also need to load the segmentation of the previous stage and
            # concatenate it. Here it will be concatenates to the segmentation because the augmentations need to be
            # applied to it in segmentation mode. Later in the data augmentation we move it from the segmentations to
            # the last channel of the data
            if self.has_prev_stage:
                if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                       mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]
                # we theoretically support several possible previsous segmentations from which only one is sampled. But
                # in practice this feature was never used so it's always only one segmentation
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None

            # do you trust me? You better do. Otherwise you'll have to go through this mess and honestly there are
            # better things you could do right now

            # (above) documentation of the day. Nice. Even myself coming back 1 months later I have not friggin idea
            # what's going on. I keep the above documentation just for fun but attempt to make things clearer now

            need_to_pad = self.need_to_pad
            for d in range(3):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint
            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]

                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    # If the image does not contain any foreground classes, we fall back to random cropping
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                          valid_bbox_y_lb:valid_bbox_y_ub,
                                          valid_bbox_z_lb:valid_bbox_z_ub]

            data[j] = np.pad(case_all_data[:-self._num_seg], ((0, 0),
                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                  (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)

            seg[j, 0:self._num_seg] = np.pad(case_all_data[-self._num_seg:], ((0, 0),
                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': -1})
            if seg_from_previous_stage is not None:
                seg[j, 1] = np.pad(seg_from_previous_stage, ((0, 0),
                                                             (-min(0, bbox_x_lb),
                                                              max(bbox_x_ub - shape[0], 0)),
                                                             (-min(0, bbox_y_lb),
                                                              max(bbox_y_ub - shape[1], 0)),
                                                             (-min(0, bbox_z_lb),
                                                              max(bbox_z_ub - shape[2], 0))),
                                   'constant', **{'constant_values': 0})

        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}

class DataLoader3D_Geodesic(DataLoader3D_multilabel):
    def __init__(self, data, patch_size, final_patch_size, batch_size, network, ignore_label, all_seg_labels, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None, label_mod = 1):

        self.ignore_label = ignore_label
        self._network_for_predict = network
        self._all_seg_labels = all_seg_labels
        super(DataLoader3D_Geodesic, self).__init__(data, patch_size, final_patch_size, batch_size, has_prev_stage, oversample_foreground_percent,
                                                      memmap_mode, pad_mode, pad_kwargs_data, pad_sides, label_mod)
        self._distinguish_srcribble()

        self._proportion_fakelabel = 0.5 # proportion of fake label
        self._proportion_fakelabel_focus = 0.25 # proportion focus area in fake label

    def _distinguish_srcribble(self):
        # # 指定数据
        # self.list_of_fulllabel = []
        # self.list_of_scribble = ['liver_18', 'liver_31', 'liver_36', 'liver_57', 'liver_83', 'liver_101', 'liver_106', 'liver_110', 'liver_123']
        # for selected_key in self.list_of_keys:
        #     case_all_data = np.load(self._data[selected_key]['data_file'][:-4] + ".npy", "r")
        #     if case_all_data[1, 0, 0, 0] == -1:
        #         pass
        #     else:
        #         self.list_of_fulllabel.append(selected_key)
        # 使用全部弱监督
        self.list_of_fulllabel = []
        self.list_of_scribble = []
        for selected_key in self.list_of_keys:
            # case_all_data = np.load(self._data[selected_key]['data_file'][:-4] + ".npy", "r")
            if isfile(self._data[selected_key]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[selected_key]['data_file'][:-4] + ".npy", "r")
            else:
                case_all_data = np.load(self._data[selected_key]['data_file'])['data']
            if case_all_data[1, 0, 0, 0] == -1:
                self.list_of_scribble.append(selected_key)
            else:
                self.list_of_fulllabel.append(selected_key)

    def determine_shapes(self):
        if self.has_prev_stage:
            print('error: no pre starge supportted in DataLoader3D_Geodesic')
            num_seg = 2
        else:
            num_seg = len(self._all_seg_labels)

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - self._num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        # changed for softlabel
        seg_shape = (self.batch_size, 2, *self.patch_size)
        # seg_shape = (self.batch_size, 1, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        r = np.random.uniform()
        if len(self.list_of_scribble) > 0 and r < self._proportion_fakelabel:
            selected_keys = np.random.choice(self.list_of_scribble, self.batch_size, True, None)
        else:
            selected_keys = np.random.choice(self.list_of_fulllabel, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
            # which is much faster to access
            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode) 
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']
            
            # !!! FakeLabel for godesic !!!
            fake_label_name = self._data[i]['data_file'][:-4] + "-fakelabel.npy"
            pkl_name = self._data[i]['data_file'][:-4] + "-pkl.pkl"
            is_fake_label = False
            if isfile(fake_label_name):
                is_fake_label = True

            # If we are doing the cascade then we will also need to load the segmentation of the previous stage and
            # concatenate it. Here it will be concatenates to the segmentation because the augmentations need to be
            # applied to it in segmentation mode. Later in the data augmentation we move it from the segmentations to
            # the last channel of the data
            if self.has_prev_stage:
                if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                       mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]
                # we theoretically support several possible previsous segmentations from which only one is sampled. But
                # in practice this feature was never used so it's always only one segmentation
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None

            # do you trust me? You better do. Otherwise you'll have to go through this mess and honestly there are
            # better things you could do right now

            # (above) documentation of the day. Nice. Even myself coming back 1 months later I have not friggin idea
            # what's going on. I keep the above documentation just for fun but attempt to make things clearer now

            need_to_pad = self.need_to_pad
            for d in range(3):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint
            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            if is_fake_label:
                focus_patch = False
                pkl_file = load_pickle(pkl_name)
                r2 = np.random.uniform()
                if r2 < self._proportion_fakelabel_focus:
                    focus_mask = pkl_file['focus_area']
                    if len(focus_mask) > 0:
                        p = np.random.randint(focus_mask.shape[0])
                        bbox_x_lb = focus_mask[p][0] - self.patch_size[0] // 2
                        bbox_y_lb = focus_mask[p][1] - self.patch_size[1] // 2
                        bbox_z_lb = focus_mask[p][2] - self.patch_size[2] // 2
                        focus_patch = True
                if not focus_patch:
                    # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
                    # at least one of the foreground classes in the patch
                    if not force_fg:
                        bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                        bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                        bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
                    else:
                        class_locations = pkl_file['class_locations']

                        # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                        foreground_classes = np.array(
                            [i for i in class_locations.keys() if len(class_locations[i]) != 0])
                        foreground_classes = foreground_classes[foreground_classes > 0]

                        if len(foreground_classes) == 0:
                            # this only happens if some image does not contain foreground voxels at all
                            selected_class = None
                            voxels_of_that_class = None
                            # print('case does not contain any foreground classes', i)
                        else:
                            selected_class = np.random.choice(foreground_classes)

                            voxels_of_that_class = class_locations[selected_class]

                        if voxels_of_that_class is not None:
                            selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                            # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                            # Make sure it is within the bounds of lb and ub
                            bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                            bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                            bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                        else:
                            # If the image does not contain any foreground classes, we fall back to random cropping
                            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
                    

            else:
                # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
                # at least one of the foreground classes in the patch
                if not force_fg:
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
                else:
                    # these values should have been precomputed
                    if 'class_locations' not in properties.keys():
                        raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                    # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                    foreground_classes = np.array(
                        [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                    foreground_classes = foreground_classes[foreground_classes > 0]

                    if len(foreground_classes) == 0:
                        # this only happens if some image does not contain foreground voxels at all
                        selected_class = None
                        voxels_of_that_class = None
                        # print('case does not contain any foreground classes', i)
                    else:
                        selected_class = np.random.choice(foreground_classes)

                        voxels_of_that_class = properties['class_locations'][selected_class]

                    if voxels_of_that_class is not None:
                        selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                        # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                        # Make sure it is within the bounds of lb and ub
                        bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                        bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                        bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                    else:
                        # If the image does not contain any foreground classes, we fall back to random cropping
                        bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                        bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                        bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                          valid_bbox_y_lb:valid_bbox_y_ub,
                                          valid_bbox_z_lb:valid_bbox_z_ub]

            data[j] = np.pad(case_all_data[:1], ((0, 0),
                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                  (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)

            # !!! FakeLabel for godesic !!!
            if is_fake_label:
                fake_label = np.load(fake_label_name, self.memmap_mode)
                fake_label =  np.copy(fake_label[:,valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])
                seg[j] = np.pad(fake_label,  ((0, 0),
                                                        (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                        (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                        (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                'constant', **{'constant_values': -1})
                seg[j, 1, :, :, :] /= 1000 # max confidence is 1000 in so

            else:
                seg[j, 0] = np.pad(case_all_data[1], (
                                                        (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                        (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                        (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                'constant', **{'constant_values': -1})
                seg[j, 1, :, :, :] = 1 # max confidence


        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}
    