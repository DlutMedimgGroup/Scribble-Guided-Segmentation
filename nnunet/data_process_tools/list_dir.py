from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
import numpy as np
l1 = np.array(range(15))
l2 = (l1*(208/15)).astype(np.int)+2


target_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task07_Pancreas/imagesTr'
cvs_full_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task07_Pancreas/data_ids.csv'

data_list = subfiles(target_path, join=False, prefix='pancreas_', suffix='.nii.gz')
id_list = [int(x[9:12]) for x in data_list]
id_list = id_list[:280]
id_list_1 = []
id_list_2 = []
id_list_3 = []
id_list_4 = []
id_list_5 = []
id_list_list = [id_list_1, id_list_2, id_list_3, id_list_4, id_list_5]
for i in range(len(id_list)):
    id_list_list[i%5].append(id_list[i])

dict_out = {'data_id1': id_list_1, 'data_id2': id_list_2, 'data_id3': id_list_3, 'data_id4': id_list_4, 'data_id5': id_list_5}

dataframe = pd.DataFrame(dict_out)
dataframe.to_csv(cvs_full_path, index=False, sep=',')
pause = 1