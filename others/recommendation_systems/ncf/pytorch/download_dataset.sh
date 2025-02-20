# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
#    under the License

# 新建目录 data
mkdir data 
cd data

# Creates ml-20.zip
curl -O http://files.grouplens.org/datasets/movielens/ml-20m.zip

# Unzip
unzip ml-20m.zip

# delete zip
rm -rf ml-20m.zip 

# 返回上一级目录
cd ..
