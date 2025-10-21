# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
import os
import shutil
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_datasets_path", type=str, default=None)
    parser.add_argument("--datasets_path", type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    train_dir = os.path.join(args.datasets_path, 'train')
    test_dir = os.path.join(args.datasets_path, 'test')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    

    # 处理train
    for file in os.listdir(os.path.join(args.origin_datasets_path, 'bounding_box_train')):
        print(file)
        id = file.split('_')[0]
        if not os.path.exists(os.path.join(train_dir, id)):
            os.mkdir(os.path.join(train_dir, id))
        else:
            shutil.copy(os.path.join(args.origin_datasets_path, 'bounding_box_train', file), os.path.join(train_dir, id))

    # 处理test
    for file in os.listdir(os.path.join(args.origin_datasets_path, 'bounding_box_test')):
        id = file.split('_')[0]
        if not os.path.exists(os.path.join(test_dir, id)):
            os.mkdir(os.path.join(test_dir, id))
        else:
            shutil.copy(os.path.join(args.origin_datasets_path, 'bounding_box_test', file), os.path.join(test_dir, id))
    print("Process down!")