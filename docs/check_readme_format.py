# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

def print_error(text, content, file_path):
    if text not in content:
        print(f"Error: Missing '{text}' in {file_path}")

def check_readme_files(directory):
    # 一级目录
    for root, dirs, files in os.walk(directory):
        current_level = root[len(directory):].count(os.sep)
        # if current_level > 3:  # 只检查四级子目录
        #     break
            
        if 'README.md' in files:
            file_path = os.path.join(root, 'README.md')
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print_error('## Model Description', content, file_path)
                print_error('## Model Preparation', content, file_path)
                print_error('## Model Training', content, file_path)
                print_error('### Prepare Resources', content, file_path)
                print_error('### Install Dependencies', content, file_path)
                print_error('## Model Results', content, file_path)
                print_error('## References', content, file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_readme.py <directory>")
        sys.exit(1)
        
    target_directory = sys.argv[1]
    if not os.path.isdir(target_directory):
        print(f"Error: {target_directory} is not a valid directory")
        sys.exit(1)
        
    check_readme_files(target_directory)
