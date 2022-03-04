from batchgenerators.utilities.file_and_folder_operations import *

def main1():
    '''
    根据目录中 imagesTr, imagesTs, labelsTr 下的内容自动生成 dataset.json 文件，其中对应的图像和标签的文件名要相同
    其余的信息由 reference_json 中获得，可以将其设置为数据集中自带的那个 dataset.json 文件
    会自动校验 imagesTr 和 labelsTr 的文件是否一一对应，不对应报错
    '''
    working_foler = "/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task53_Pancreas"
    reference_json_path = "/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task07_Pancreas/dataset.json"

    imagesTr_path = join(working_foler, 'imagesTr')
    imagesTs_path = join(working_foler, 'imagesTs')
    labelsTr_path = join(working_foler, 'labelsTr')
    output_name = join(working_foler, 'dataset.json')

    reference_json = load_json(reference_json_path)
    imagesTr_list = listdir(imagesTr_path)
    imagesTs_list = listdir(imagesTs_path)
    labelsTr_list = listdir(labelsTr_path)

    if len(imagesTr_list[0].split('_')) == 2:
        imagesTr_list_c = imagesTr_list
    else:
        imagesTr_list_c = [name[:-12]+".nii.gz" for name in imagesTr_list]

    for imagesTr_name in imagesTr_list_c:
        if imagesTr_name not in labelsTr_list:
            print("Error: " + imagesTr_name + " is not in labelsTr but in imagesTr.")
            return
    for labelsTr_name in labelsTr_list:
        if labelsTr_name not in imagesTr_list_c:
            print("Error: " + labelsTr_name + " is not in imagesTr but in labelsTr.")
            return

    training_list = []
    for imagesTr_name in imagesTr_list_c:
        # imagesTr 和 labelsTr 中文件名相同
        training_list.append({"image": "./imagesTr/" + imagesTr_name, "label": "./labelsTr/" + imagesTr_name})
    test_list = []
    for imagesTs_name in imagesTs_list:
        test_list.append("./imagesTs/" + imagesTr_name)

    reference_json["numTraining"] = len(training_list)
    reference_json["numTest"] = len(test_list)
    reference_json["training"] = training_list
    reference_json["test"] = test_list

    save_json(reference_json, output_name)

if __name__ == "__main__":
    main1()