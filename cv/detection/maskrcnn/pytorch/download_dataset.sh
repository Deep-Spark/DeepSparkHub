#!/bin/bash
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
#    under the License.
# Get COCO 2017 data sets
mkdir -p datasets/coco
pushd datasets/coco

curl -O https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
tar xzf coco_annotations_minival.tgz

curl -O http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

curl -O http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# TBD: MD5 verification
# $md5sum *.zip *.tgz
#f4bbac642086de4f52a3fdda2de5fa2c  annotations_trainval2017.zip
#cced6f7f71b7629ddf16f17bbcfab6b2  train2017.zip
#442b8da7639aecaf257c1dceb8ba8c80  val2017.zip
#2d2b9d2283adb5e3b8d25eec88e65064  coco_annotations_minival.tgz

popd
