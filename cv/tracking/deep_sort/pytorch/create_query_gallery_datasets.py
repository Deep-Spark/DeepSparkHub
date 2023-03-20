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
import re
import os
import shutil
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_datasets_path", type=str, default=None)
    parser.add_argument("--datasets_path", type=str, default=None)
    return parser.parse_args()

def extract_market(src_path, dst_root):
    img_names = os.listdir(src_path)
    pattern = re.compile(r'([-\d]+)_c(\d)')
    pid_container = set()
    for img_name in img_names:
        if '.jpg' not in img_name:
            continue
        print(img_name)
        # pid: 每个人的标签编号 1
        # _  : 摄像头号 2
        pid, _ = map(int, pattern.search(img_name).groups())
        # 去掉没用的图片
        if pid == 0 or pid == -1:
            continue
        dst_dir = os.path.join(dst_root, str(pid))
        if os.path.exists(dst_dir) == False:
            os.makedirs(dst_dir)
        shutil.copy(os.path.join(src_path, img_name), os.path.join(dst_dir, img_name))
        
if __name__ == '__main__':
    args = arg_parse()
    src_query = os.path.join(args.origin_datasets_path, 'query')
    dst_query = os.path.join(args.datasets_path, 'query')
    src_gallery = os.path.join(args.origin_datasets_path, 'bounding_box_test')
    dst_gallery = os.path.join(args.datasets_path, 'gallery')
    if not os.path.exists(dst_query):
        os.mkdir(dst_query)
    if not os.path.exists(dst_gallery):
        os.mkdir(dst_gallery)

    extract_market(src_query, dst_query)
    extract_market(src_gallery, dst_gallery)
    print("Process down!")