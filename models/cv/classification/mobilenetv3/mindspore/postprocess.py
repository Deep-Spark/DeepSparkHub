# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''post process for 310 inference'''
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='post process for 310 inference')
parser.add_argument("--dataset", type=str, default="imagenet", help="result file path")
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--label_file", type=str, required=True, help="label file")
args = parser.parse_args()

def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count

def read_label(label_file):
    '''read label file'''
    f = open(label_file, "r")
    lines = f.readlines()

    img_label = {}
    for line in lines:
        img_id = line.split(":")[0]
        label = line.split(":")[1]
        img_label[img_id] = label

    return img_label

def cal_acc(dataset, result_path, label_file):
    '''main acc calculation function'''
    img_label = read_label(label_file)

    img_tot = 0
    top1_correct = 0
    top5_correct = 0

    files = os.listdir(result_path)
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(1, 1000)
            gt_classes = int(img_label[file[:-6]])

            top1_output = np.argmax(result, (-1))
            top5_output = np.argsort(result)[:, -5:]

            t1_correct = np.equal(top1_output, gt_classes).sum()
            top1_correct += t1_correct
            top5_correct += get_top5_acc(top5_output, [gt_classes])
            img_tot += 1

    results = [[top1_correct], [top5_correct], [img_tot]]

    results = np.array(results)
    top1_correct = results[0, 0]
    top5_correct = results[1, 0]
    img_tot = results[2, 0]
    acc1 = 100.0 * top1_correct / img_tot
    print('Eval: top1_correct={}, tot={}, acc={:.2f}%'.format(top1_correct, img_tot, acc1))

if __name__ == "__main__":
    cal_acc(args.dataset, args.result_path, args.label_file)
