import os
import re
import csv

log_file_path = "/home/mingrui/disk1/projects/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task003_Liver/nnUNetTrainerV2_anatomy__nnUNetPlansv2.1_small/fold_0"
log_file_name = "training_log_2021_5_28_10_44_48.txt"
csv_file_name = log_file_name[:-3]+"csv"
log_full_path = os.path.join(log_file_path, log_file_name)
csv_full_path = os.path.join(log_file_path, csv_file_name)


epoch_data_title = ["epoch", "train loss", "validation loss", "Dice_0", "Dice_1", "lr", "time"]
epoch_data_list = []
f = open(log_full_path)
lines = f.readlines()
epoch = -1
epoch_status = 0 # 0: epoch; 1: train loss; 2: validation loss; 3: Average global foreground Dice; 4: lr; 5: time
for line in lines:
    line=line.strip('\n')
    if epoch_status == 0:
        if line[0:6] == "epoch:":
            # assert int(line[6:]) == epoch + 1, "epoch序号错误：" + str(epoch)
            # epoch = epoch + 1
            epoch = int(line[6:])
            epoch_data = []
            epoch_data.append(int(epoch))
            epoch_status = 1
    elif epoch_status == 1:
        if len(line) < 28:
            continue
        line = line[28:]
        if line[0:12] == "train loss :":
            epoch_data.append(float(line[12:]))
            epoch_status = 2
    elif epoch_status == 2:
        if len(line) < 28:
            continue
        line = line[28:]
        if line[0:16] == "validation loss:":
            epoch_data.append(float(line[16:]))
            epoch_status = 3
    elif epoch_status == 3:
        if len(line) < 28:
            continue
        line = line[28:]
        if line[0:31] == "Average global foreground Dice:":
            dice = line[31:]
            dice = re.findall(r'[[](.*?)[]]', dice)
            dice = dice[0].split(', ')
            for i, d in enumerate(dice):
                epoch_data.append(float(d))
            epoch_status = 4
    elif epoch_status == 4:
        if len(line) < 28:
            continue
        line = line[28:]
        if line[0:3] == "lr:":
            epoch_data.append(float(line[3:]))
            epoch_status = 5
    elif epoch_status == 5:
        if len(line) < 28:
            continue
        line = line[28:]
        if line[0:15] == "This epoch took":
            epoch_data.append(float(line[15:26]))
            epoch_status = 0
            epoch_data_list.append(epoch_data)

with open(csv_full_path,'w',newline='') as t:#numline是来控制空的行数的
    writer=csv.writer(t)#这一步是创建一个csv的写入器
    writer.writerow(epoch_data_title)#写入标签
    writer.writerows(epoch_data_list)#写入样本数据

print("Succeeded!")
