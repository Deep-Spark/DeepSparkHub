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

"""Module providing configuration for training on human parsing with resnet50
backbone"""

from glob import glob

import tensorflow as tf


CONFIG = {
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'human-parsing-resnet-50-backbone',

    'train_dataset_config': {
        'images': sorted(
            glob(
                './dataset/instance-level_human_parsing/'
                'instance-level_human_parsing/Training/Images/*'
            )
        ),
        'labels': sorted(
            glob(
                './dataset/instance-level_human_parsing/'
                'instance-level_human_parsing/Training/Category_ids/*'
            )
        ),
        'height': 512, 'width': 512, 'batch_size': 8
    },

    'val_dataset_config': {
        'images': sorted(
            glob(
                './dataset/instance-level_human_parsing/'
                'instance-level_human_parsing/Validation/Images/*'
            )
        ),
        'labels': sorted(
            glob(
                './dataset/instance-level_human_parsing/'
                'instance-level_human_parsing/Validation/Category_ids/*'
            )
        ),
        'height': 512, 'width': 512, 'batch_size': 8
    },

    'num_classes': 20,
    'backbone': 'resnet50',
    'learning_rate': 0.0001,

    'checkpoint_dir': "./checkpoints/",
    'checkpoint_file_prefix':
    'deeplabv3-plus-human-parsing-resnet-50-backbone_',

    'epochs': 100
}
