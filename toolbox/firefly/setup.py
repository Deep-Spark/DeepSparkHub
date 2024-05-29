# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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


import setuptools
import os

    
def req_file(filename, folder="./"):
    with open(os.path.join(folder, filename), encoding='utf-8') as f:
        content = f.readlines()
    return [x.strip() for x in content]

install_requires = req_file("requirements.txt")

setuptools.setup(
    name="firefly",
    version="0.1.0",
    description="Firefly is an open-source project for large-scale model training, supporting pre-training and fine-tuning of state-of-the-art large models.",
    packages=setuptools.find_packages(),
    python_requires='>=3.7, <4',
    install_requires=install_requires
)
