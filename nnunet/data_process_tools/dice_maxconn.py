'''
计算统计数据Dice系数
'''
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk


def Get_Dice_Coefficient(pred_path, gt_path, organ):
    pred_img = sitk.ReadImage(pred_path)
    gt_img = sitk.ReadImage(gt_path)
    pred_array = sitk.GetArrayFromImage(pred_img).astype(int)
    gt_array = sitk.GetArrayFromImage(gt_img).astype(int)

    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    mask_img = cca.Execute(pred_img)
    mask_array = sitk.GetArrayFromImage(mask_img).astype(int)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_img)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label +1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    final_label = num_list_sorted[0]
    pred_array[mask_array != final_label] = 0
 
    output = dict()
    for target in organ:
        label = organ[target]
        pred_target = pred_array.copy()
        gt_target = gt_array.copy()
        pred_target[pred_target != label] = 0
        pred_target[pred_target == label] = 1
        gt_target[gt_target != label] = 0
        gt_target[gt_target == label] = 1
        ints = np.sum(gt_target * pred_target)
        sums = np.sum((pred_target * 1) + (gt_target * 1))
        if sums == 0:
            dice = 'null'
        else:
            dice = ((2.0 * ints) / sums)
        output[target] = dice
    return output


def main():
    cvs_full_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_inference/Task331_Liver/scribble_331-3_Liver_best/scribble_331-3_Liver_best-maxconn.csv'
    pred_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_inference/Task331_Liver/scribble_331-3_Liver_best'
    gt_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task003_Liver/labelsTr/'
    organ = {'liver': 1, 'liver_tumor': 2}

    # cvs_full_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_inference/Task411_Hippocampus/small_411_distta/small_411_distta.csv'
    # pred_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_inference/Task411_Hippocampus/small_411_distta'
    # gt_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task004_Hippocampus/labelsTr/'
    # organ = {'Anterior': 1, 'Posterior': 2}

    # cvs_full_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_inference/Task511_Pancreas/small_511_distta/small_511_distta.csv'
    # pred_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_inference/Task511_Pancreas/small_511_distta'
    # gt_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task007_Pancreas/labelsTr/'
    # organ = {'Pancreas': 1, 'Tumor': 2}

    file_list = os.listdir(pred_path)
    data_name_list = []
    dict_out = {'data_id': data_name_list}
    for key in organ:
        dict_out[key] = []

    for pred_name in file_list:
        if not pred_name.endswith('.nii.gz'):
            continue
        pred_full_path = os.path.join(pred_path, pred_name)
        gt_full_path = os.path.join(gt_path, pred_name)        

        dice = Get_Dice_Coefficient(pred_full_path, gt_full_path, organ)

        data_name_list.append(pred_name)
        for key in organ:
            dict_out[key].append(dice[key])

        print(pred_name + ' finished')

    dataframe = pd.DataFrame(dict_out)
    dataframe.to_csv(cvs_full_path, index=False, sep=',')


if __name__ == '__main__':
    main()