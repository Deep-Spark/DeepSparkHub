'''
Description: 
Author: Liwei Dai
Date: 2021-05-10 19:35:41
LastEditors: VSCode
LastEditTime: 2021-05-10 19:38:34
'''
import os
import argparse
import xml.etree.ElementTree as ET

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    # Rename the folder containing images
    try:
        os.rename(os.path.join(data_path, 'JPEGImages'), os.path.join(data_path, 'images'))
    except FileNotFoundError:
        print("JPEGImages folder has already been renamed to images")
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]
    
    os.makedirs(os.path.join(data_path, 'labels'), exist_ok=True)
    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'images', image_ind + '.jpg')
            label_path = os.path.join(data_path, 'labels', image_ind + '.txt') # This will be created
            anno_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(anno_path).getroot()
            objects = root.findall('object')
            labels = []
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = int(bbox.find('xmin').text.strip())
                xmax = int(bbox.find('xmax').text.strip())
                ymin = int(bbox.find('ymin').text.strip())
                ymax = int(bbox.find('ymax').text.strip())
                annotation = os.path.join(data_path, 'labels', image_ind + '.txt')
                img_size = root.find('size')
                h, w = int(img_size.find('height').text.strip()), int(img_size.find('width').text.strip())

                # Prepare for labels
                x_center, y_center = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
                h_obj, w_obj = abs(xmax - xmin) /w , abs(ymax - ymin) /h

                label = ' '.join(str(i) for i in [class_ind, x_center, y_center, w_obj, h_obj])     
                labels.append(label)
            with open(label_path, 'w') as f_label:
                f_label.writelines("%s\n" % l for l in labels)
            f.write(image_path + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./VOC/")
    parser.add_argument("--train_annotation", default="./data/voc/train.txt")
    parser.add_argument("--test_annotation",  default="./data/voc/valid.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)
    if os.path.dirname(flags.train_annotation):
        os.makedirs(os.path.dirname(flags.train_annotation), exist_ok=True)
    if os.path.dirname(flags.train_annotation):
        os.makedirs(os.path.dirname(flags.test_annotation), exist_ok=True)

    num1 = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit/VOC2007'), 'trainval', flags.train_annotation, False)
    num2 = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit/VOC2012'), 'trainval', flags.train_annotation, False)
    num3 = convert_voc_annotation(os.path.join(flags.data_path, 'test/VOCdevkit/VOC2007'),  'test', flags.test_annotation, False)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1 + num2, num3))


