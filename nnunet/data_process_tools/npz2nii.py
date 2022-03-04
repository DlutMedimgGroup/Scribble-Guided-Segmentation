# covert npz file to nii

import os
from posixpath import basename
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import pandas as pd
import SimpleITK as sitk
from numpy.lib.type_check import imag
from tqdm import tqdm

# def main():
#     file_list = os.listdir(npz_path)
#     for file_name in tqdm(file_list): 
#         if not file_name.endswith('.npz'):
#             continue
#         full_path = os.path.join(npz_path, file_name)
#         npz_data = np.load(full_path)
#         softmax = npz_data['softmax']
#         softmax = softmax.astype(np.float32)
#         softmax = softmax[:,:,::-1,::-1]
#         reference_path = os.path.join(npz_path, file_name[:-4] + '.nii.gz')
#         reference_img = sitk.ReadImage(reference_path)

#         out_filename = os.path.join(output_path, file_name[:-4] + '_label-')
#         for label in range(3):
#             image = sitk.GetImageFromArray(softmax[label])
#             origin = reference_img.GetOrigin()
#             spacing = reference_img.GetSpacing()
#             dim = reference_img.GetSize()
#             image.SetOrigin((origin[0] - (spacing[0] * dim[0]), origin[1] - (spacing[1] * dim[1]), origin[2]))
#             image.SetSpacing(reference_img.GetSpacing())
#             sitk.WriteImage(image, out_filename + str(label) + '.nii.gz')

def main1():
    npz_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/geodisic_cache'
    output_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/geodisic_cache/mhd'
    file_list = subfiles(npz_path, join=True, suffix='.npy')
    for file_name in tqdm(file_list):
        data = np.load(file_name).astype(np.float32)
        for i in range(data.shape[0]):
            image = sitk.GetImageFromArray(data[i])
            out_name = join(output_path, basename(file_name)[:-4] + '-' + str(i) + '.mhd')
            print(out_name)
            sitk.WriteImage(image, out_name)

def main2():
    npz_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/geodisic_cache'
    output_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/geodisic_cache/mhd'
    file_list = subfiles(npz_path, join=True, suffix='.npy')
    for file_name in tqdm(file_list):
        # pkl
        # pkl_name = file_name[:-4] + '.pkl'
        # pkl = pd.read_pickle(pkl_name)
        # npz
        # npz_data = np.load(file_name)
        # data = npz_data['data']
        data = np.load(file_name)
        data_label = data.argmax(0).astype(np.int32)
        image_label = sitk.GetImageFromArray(data_label)
        out_name = join(output_path, basename(file_name)[:-4] + '-label.mhd')
        sitk.WriteImage(image_label, out_name)
        # for i in range(data.shape[0]):
        #     image = sitk.GetImageFromArray(data[i])
        #     # image.SetOrigin(pkl['itk_origin'])
        #     # image.SetSpacing(pkl['itk_spacing'])
        #     out_name = join(output_path, basename(file_name)[:-4] + '-' + str(i) + '.mhd')
        #     print(out_name)
        #     sitk.WriteImage(image, out_name)

def main3():
    npz_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_inference/Task311_Liver/debug_deleteme/liver_1.npz'
    output_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_inference/Task311_Liver/debug_deleteme'

    data = np.load(npz_path)
    data_array = data['softmax'].astype(np.float32)
    for i in range(data_array.shape[0]):
        image = sitk.GetImageFromArray(data_array[i])
        out_name = join(output_path, 'liver_1-' + str(i) + '.mhd')
        print(out_name)
        sitk.WriteImage(image, out_name)

def main4():
    npy_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_cropped_data/Scribble304_Liver/'
    output_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/geodisic_cache/mhd/liver_2-new_map.mhd'
    data = np.load(npy_path).astype(np.int32)
    image = sitk.GetImageFromArray(data)
    sitk.WriteImage(image, output_path)

def main5():
    npy_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/geodisic_cache'
    output_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/geodisic_cache/mhd'
    file_list = subfiles(npy_path, join=True, suffix='.npy')
    for file_name in tqdm(file_list):
        data = np.load(file_name).astype(np.float32)
        image = sitk.GetImageFromArray(data)
        out_name = join(output_path, basename(file_name)[:-4] + '.mhd')
        print(out_name)
        sitk.WriteImage(image, out_name)


if __name__ == "__main__":
    main3()
 