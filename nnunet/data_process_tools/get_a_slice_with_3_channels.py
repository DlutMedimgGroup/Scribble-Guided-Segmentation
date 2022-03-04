import os
import numpy as np
import SimpleITK as sitk
from numpy.lib.type_check import imag
from tqdm import tqdm
from PIL import Image

def get_a_slice_with_3_channels():
    image_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task003_Liver/imagesTr/liver_11_0000.nii.gz'
    npz_path = '/mnt/sda/mingrui/projects/nnUNet/DATASET/nnUNet_inference/Task03_Liver/l_to_s_tail5/l_to_s_tail5_inference_npz/liver_11.npz'
    output_dir = '/mnt/sda/mingrui/projects/nnUNet/DATASET/nnUNet_inference/Task03_Liver/l_to_s_tail5/2d_glabcut_test'

    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    npz_data = np.load(npz_path)
    softmax = npz_data['softmax']
    softmax = softmax.astype(np.float32)
    # softmax = softmax[:,:,::-1,::-1]

    image_array_2d = image_array[:, 312, :]
    channel_1_2d = softmax[1, :, 312, :]
    channel_2_2d = softmax[2, :, 312, :]

    min_max = [-300, 700]
    image_array_2d = (image_array_2d - min_max[0]) / (min_max[1] - min_max[0]) * 255
    channel_1_2d = channel_1_2d * 255 / channel_1_2d.max()
    channel_2_2d = channel_2_2d * 25500000
    channel_2_2d[channel_2_2d>255] = 255

    shape = [image_array_2d.shape[0], image_array_2d.shape[1], 3]
    out_array = np.zeros(shape)
    out_array[:, :, 0] = image_array_2d
    out_array[:, :, 1] = channel_1_2d
    out_array[:, :, 2] = channel_2_2d
    out_array = out_array.astype(np.uint8)
    out_array[out_array==0]=1

    output_full_path = os.path.join(output_dir, 'test.bmp')
    im = Image.fromarray(out_array)
    im.save(output_full_path)

    pause = 1


def export_property_to_nii():
    image_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task003_Liver/imagesTr/liver_11_0000.nii.gz'
    npz_path = '/mnt/sda/mingrui/projects/nnUNet/DATASET/nnUNet_inference/Task03_Liver/l_to_s_tail5/l_to_s_tail5_inference_npz/liver_11.npz'
    output_dir = '/mnt/sda/mingrui/projects/nnUNet/DATASET/nnUNet_inference/Task03_Liver/l_to_s_tail5/2d_glabcut_test'

    image = sitk.ReadImage(image_path)

    npz_data = np.load(npz_path)
    softmax = npz_data['softmax']
    softmax = softmax.astype(np.float32)

    liver_property = softmax[1, :, ::-1, ::-1]
    tumor_property = softmax[2, :, ::-1, ::-1]
    liver_property_img = sitk.GetImageFromArray(liver_property)
    liver_property_img.SetOrigin(image.GetOrigin())
    liver_property_img.SetSpacing(image.GetSpacing())
    tumor_property_img = sitk.GetImageFromArray(tumor_property)
    tumor_property_img.SetOrigin(image.GetOrigin())
    tumor_property_img.SetSpacing(image.GetSpacing())

    liver_path = os.path.join(output_dir, 'liver_property.nii')
    sitk.WriteImage(liver_property_img, liver_path)
    tumor_path = os.path.join(output_dir, 'tumor_property.nii')
    sitk.WriteImage(tumor_property_img, tumor_path)

    pause = 1


if __name__ == "__main__":
    # get_a_slice_with_3_channels()
    export_property_to_nii()