#############################################################################
"""You have to fill these args.

_base_(str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""

_base_ = ['mmcls::resnet/resnet34_8xb32_in1k.py']

un_prune = 1.0
stage_ratio_1 = 0.5
stage_ratio_2 = 0.4
stage_ratio_3 = 0.6
stage_ratio_4 = un_prune

fix_subnet = {
    # stage 1
    'backbone.conv1_(0, 64)_64': un_prune,  # short cut layers
    'backbone.layer1.0.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.1.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.2.conv1_(0, 64)_64': un_prune,
    # stage 2
    'backbone.layer2.0.conv1_(0, 128)_128': un_prune,
    'backbone.layer2.0.conv2_(0, 128)_128': un_prune,  # short cut layers
    'backbone.layer2.1.conv1_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.2.conv1_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.3.conv1_(0, 128)_128': un_prune,
    # stage 3
    'backbone.layer3.0.conv1_(0, 256)_256': un_prune,
    'backbone.layer3.0.conv2_(0, 256)_256': un_prune,  # short cut layers
    'backbone.layer3.1.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.2.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.3.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.4.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.5.conv1_(0, 256)_256': un_prune,
    # stage 4
    'backbone.layer4.0.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.0.conv2_(0, 512)_512': un_prune,  # short cut layers
    'backbone.layer4.1.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.2.conv1_(0, 512)_512': stage_ratio_4
}

divisor = 8
##############################################################################

architecture = _base_.model

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherDeploySubModel',
    architecture=architecture,
    fix_subnet=fix_subnet,
    divisor=divisor,
)
