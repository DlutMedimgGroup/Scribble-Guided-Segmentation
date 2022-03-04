import json
import shutil
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, subdirs, isfile
from nnunet.configuration import default_num_threads
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
from nnunet.experiment_planning.multilabel.cropping_multilabel import ImageCropper_multilabel

def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    num_modalities_label = len(d['modality_label'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        for mod in range(num_modalities_label):
            cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        lists.append(cur_pat)
    return lists, num_modalities_label, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}

def crop_multilabel(task_string, override=False, num_threads=default_num_threads):
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
    lists, num_modalities_label, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    imgcrop = ImageCropper_multilabel(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, num_modalities_label, overwrite_existing=override)
    shutil.copy(join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)