from batchgenerators.utilities.file_and_folder_operations import *


splits_file = join("/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_cropped_data/Task003_Liver", "intensityproperties.pkl")
splits = load_pickle(splits_file)

pause = 1
