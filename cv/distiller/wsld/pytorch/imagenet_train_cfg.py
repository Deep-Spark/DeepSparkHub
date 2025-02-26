# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

from tools.collections import AttrDict

__C = AttrDict()

cfg = __C

__C.net_type='resnet'   # choose resnet or mobilenet

__C.train_params=AttrDict()
__C.train_params.epochs = 100
__C.train_params.use_seed=False
__C.train_params.seed=0
__C.train_params.print_freq = 50

__C.optim=AttrDict()
__C.optim.init_lr=0.1
__C.optim.min_lr=0
__C.optim.momentum=0.9
__C.optim.weight_decay=1e-4
__C.optim.use_grad_clip=False
__C.optim.grad_clip=10
__C.optim.label_smooth=False
__C.optim.smooth_alpha=0.1

__C.optim.if_resume=False
__C.optim.resume_path=''

__C.data=AttrDict()
__C.data.data_path = '/home/datasets/cv/imagenet'
__C.data.num_workers=32
__C.data.batch_size=256
__C.data.dataset='imagenet'
__C.data.train_data_type='lmdb'  # set 'img' to read original images
__C.data.val_data_type='lmdb'  # set 'img' to read original images
__C.data.patch_dataset=False
__C.data.num_examples=1281167
__C.data.input_size=(3,224,224)
__C.data.type_of_data_aug='random_sized'  # random_sized / rand_scale
__C.data.random_sized=AttrDict()
__C.data.random_sized.min_scale=0.08
__C.data.mean=[0.485, 0.456, 0.406]
__C.data.std=[0.229, 0.224, 0.225]
__C.data.color=False

__C.optim.cosine=AttrDict()
__C.optim.cosine.use_restart=False
__C.optim.cosine.restart=AttrDict()
__C.optim.cosine.restart.lr_period=[10, 20, 40, 80, 160, 320]
__C.optim.cosine.restart.lr_step=[0, 10, 30, 70, 150, 310]
