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

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(type='ResNet', 
        depth=34),
    decode_head=dict(
        type='ATTUNetHead',
        in_channels=[64, 128, 256, 512],
        channels=64,
        input_transform=None,
        # dropout_ratio=None,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    use_sigmoid=False,
                    loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
                dict(type="FocalLoss")
            ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)