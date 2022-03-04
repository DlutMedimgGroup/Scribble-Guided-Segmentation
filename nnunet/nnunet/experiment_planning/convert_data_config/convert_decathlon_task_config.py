import os
import shutil
import argparse
import configparser
import re
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool, pool

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.nnUNet_convert_decathlon_task import crawl_and_remove_hidden_from_decathlon
from nnunet.utilities.file_endings import remove_trailing_slash
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
from nnunet.configuration import default_num_threads

def cat_and_split_label(file_base, label_path, output_folder):
    if (not isfile(label_path)):
        return False
    if isfile(label_path):
        label_itk = sitk.ReadImage(label_path)
        dim = label_itk.GetDimension()
        if dim != 3:
            raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, label_path))
        shutil.copy(label_path, join(output_folder, 'labelsTr', file_base + ".nii.gz"))
    return True

def cat_and_split(image_path, label_path, output_folder):
    img_itk = sitk.ReadImage(image_path)
    dim = img_itk.GetDimension()
    size = img_itk.GetSize()
    origin = list(img_itk.GetOrigin())
    spacing = img_itk.GetSpacing()
    direction = np.array(img_itk.GetDirection()).reshape(3,3)
    direction = tuple(direction.reshape(-1))
    # origin[0] = origin[0] - spacing[0] * size[0]
    # origin[1] = origin[1] - spacing[1] * size[1]
    file_base = image_path.split("/")[-1]
    file_index = re.split('[_\.]', file_base)
    file_base = file_index[0] + '_' + file_index[1]

    if not cat_and_split_label(file_base, label_path, output_folder):
        # 如果不存在label，直接返回
        return

    if dim == 3:
        shutil.copy(image_path, join(join(output_folder, 'imagesTr'), file_base + "_0000.nii.gz"))
        return
    elif dim != 4:
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, image_path))
    else:
        img_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = np.array(img_itk.GetDirection()).reshape(4,4)
        # now modify these to remove the fourth dimension
        spacing = tuple(list(spacing[:-1]))
        origin = tuple(list(origin[:-1]))
        direction = tuple(direction[:-1, :-1].reshape(-1))
        for i, t in enumerate(range(img_npy.shape[0])):
            img = img_npy[t]
            img_itk_new = sitk.GetImageFromArray(img)
            img_itk_new.SetSpacing(spacing)
            img_itk_new.SetOrigin(origin)
            img_itk_new.SetDirection(direction)
            sitk.WriteImage(img_itk_new, join(output_folder, file_base + "_%04.0d.nii.gz" % i))

def cat_and_split_data(source_folder, output_task_name, num_processes=default_num_threads):
    '''
    拼接和分割数据。
    '''
    # paths
    path_imagesTr = join(source_folder, 'imagesTr')
    path_imagesTs = join(source_folder, 'imagesTs')
    path_labelsTr = join(source_folder, 'labelsTr')

    # names and folders
    full_task_name = source_folder.split("/")[-1]
    assert full_task_name.startswith("Task"), "The input folder must point to a folder that starts with TaskXX_"
    first_underscore = full_task_name.find("_")
    assert first_underscore == 6, "Input folder start with TaskXX with XX being a 2-digit id: 00, 01, 02 etc"
    output_folder = join(nnUNet_raw_data, output_task_name)
    if isdir(output_folder):
        shutil.rmtree(output_folder)
    maybe_mkdir_p(output_folder)
    maybe_mkdir_p(join(output_folder, 'imagesTr'))
    maybe_mkdir_p(join(output_folder, 'imagesTs'))
    maybe_mkdir_p(join(output_folder, 'labelsTr'))

    # convert
    images = []
    labels = []
    scribbles = []
    output_dirs = []
    for image_name in os.listdir(path_imagesTr):
        if not image_name.endswith(".nii.gz"):
            continue
        base_name = os.path.basename(image_name)
        if len(re.split('_', base_name)) == 3:
            base_name_l = re.split('_', base_name)
            base_name = base_name_l[0] + '_' + base_name_l[1]
        if not base_name.endswith('.nii.gz'):
            base_name = base_name + '.nii.gz'
        images.append(join(path_imagesTr, image_name))
        labels.append(join(path_labelsTr, base_name))
        output_dirs.append(output_folder)

    # multi thread
    p = Pool(num_processes)
    p.starmap(cat_and_split, zip(images, labels, output_dirs))
    p.close()
    p.join()

    # json
    path_json = join(source_folder, 'dataset.json')
    dataset = load_json(path_json)
    train_list = listdir(join(output_folder, 'imagesTr'))
    train_list = [{'image': '.imagesTr/' + x[:-12] + '.nii.gz', 'label': './labelsTr/' + x[:-12] + '.nii.gz'} for x in train_list]
    test_list = listdir(join(output_folder, 'imagesTs'))
    test_list = ['.imagesTs/' + x[:-12] + '.nii.gz' for x in test_list]
    dataset['numTraining'] = len(train_list)
    dataset['numTest'] = len(test_list)
    dataset['training'] = train_list
    dataset['test'] = test_list
    dataset['modality_label'] = {'0': 'label', '1': 'scribble'}
    save_json(dataset, join(output_folder, 'dataset.json'))

def main():
    ''' 转换数据，将4D数据拆分
    输入文件夹内要包含imagesTr, imagesTs, labelsTr 和 scribbleTr    
    '''

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
    folder = config.get('Convert', 'source_folder')
    output_task_name = config.get('Convert', 'output_task_name')

    # confirm 
    folder = remove_trailing_slash(folder)
    assert folder.split('/')[-1].startswith("Task"), "This does not seem to be a scribble folder. Please give me a "\
                                                         "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                         "labelsTr, imagesTs and scribbleTr"
    subf = subfolders(folder, join=False)
    assert 'imagesTr' in subf, "This does not seem to be a scribble folder. Please give me a "\
                                                         "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                         "labelsTr, imagesTs and scribbleTr"
    assert 'imagesTs' in subf, "This does not seem to be a scribble folder. Please give me a "\
                                                         "folder that starts with TaskXX and has the subfolders imagesTs, " \
                                                         "labelsTr, imagesTs and scribbleTr"
    assert 'labelsTr' in subf, "This does not seem to be a scribble folder. Please give me a "\
                                                         "folder that starts with TaskXX and has the subfolders labelsTr, " \
                                                         "labelsTr, imagesTs and scribbleTr"

    # convert
    cat_and_split_data(folder, output_task_name)

if __name__ == "__main__":

    main()