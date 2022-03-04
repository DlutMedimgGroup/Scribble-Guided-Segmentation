"""
Batch process to delete npy files to free up hard drive space.

In the nnUNet project, npz files are converted to npy files when they are used to increase the speed of operations. If the npy files are deleted, the npz files will be retained without affecting the operation.
"""

import os
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm

def main():
    # working_path = '/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_preporcessed_archive'
    working_path = '/mnt/nvme0n1/mingrui/nnUNet_preprocessed'
    folders = subfolders(working_path)
    for folder in tqdm(folders):
        for_one_folder(folder)


def for_one_folder(working_path):
    subfolders = ['nnUNetData_plans_v2.1_2D_stage0', 'nnUNetData_plans_v2.1_stage0', 'nnUNetData_plans_v2.1_stage1']
    for f in subfolders:
        path = join(working_path, f)
        if not isdir(path):
            continue
        files = subfiles(path, join=True, suffix='.npy')
        for file_name in files:
            os.remove(file_name)

if __name__ == "__main__":
    main()