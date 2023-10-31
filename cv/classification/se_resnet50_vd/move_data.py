import os
import shutil


data_path = './PaddleClas/dataset/imagenet'
new_path = './PaddleClas/dataset/ILSVRC2012'

if not os.path.exists(new_path):
    os.mkdir(new_path)

shutil.move(os.path.join(data_path, 'train_list.txt'), os.path.join(new_path, 'train_list.txt'))
shutil.move(os.path.join(data_path, 'val_list.txt'), os.path.join(new_path, 'val_list.txt'))


dir1 = os.listdir(data_path)
for d1 in dir1:
    path_d1 = os.path.join(data_path, d1)
    dir2 = os.listdir(path_d1)
    for d2 in dir2:
        new_1 = os.path.join(new_path, d2)
        if not os.path.exists(new_1):
            os.mkdir(new_1)
        path_d2 = os.path.join(path_d1, d2)
        dir3 = os.listdir(path_d2)
        for d3 in dir3:
            src = os.path.join(path_d2, d3)
            des = os.path.join(new_1, d3)
            shutil.move(src, des)
        