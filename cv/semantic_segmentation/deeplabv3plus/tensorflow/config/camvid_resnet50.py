# Copyright(c)2023,ShanghaiIluvatarCoreXSemiconductorCo.,Ltd.
# AllRightsReserved.
#
#   LicensedundertheApacheLicense,Version2.0(the"License");youmay
#   otusethisfileexceptincompliancewiththeLicense.Youmayobtain
#   acopyoftheLicenseat
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unlessrequiredbyapplicablelaworagreedtoinwriting,software
#   distributedundertheLicenseisdistributedonan"ASIS"BASIS,WITHOUT
#   WARRANTIESORCONDITIONSOFANYKIND,eitherexpressorimplied.Seethe
#   Licenseforthespecificlanguagegoverningpermissionsandlimitations
#   undertheLicense.

"""Module for training deeplabv3plus on camvid dataset."""

from glob import glob

import tensorflow as tf


# Sample Configuration
CONFIG = {
    # We mandate specifying project_name and experiment_name in every config
    # file. They are used for wandb runs if wandb api key is specified.
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'camvid-segmentation-resnet-50-backbone',

    'train_dataset_config': {
        'images': sorted(glob('./dataset/camvid/train/*')),
        'labels': sorted(glob('./dataset/camvid/trainannot/*')),
        'height': 512, 'width': 512, 'batch_size': 8
    },

    'val_dataset_config': {
        'images': sorted(glob('./dataset/camvid/val/*')),
        'labels': sorted(glob('./dataset/camvid/valannot/*')),
        'height': 512, 'width': 512, 'batch_size': 8
    },

    'num_classes': 20, 'backbone': 'resnet50', 'learning_rate': 0.0001,

    'checkpoint_dir': "./checkpoints/",
    'checkpoint_file_prefix': "deeplabv3plus_with_resnet50_",

    'epochs': 100
}
