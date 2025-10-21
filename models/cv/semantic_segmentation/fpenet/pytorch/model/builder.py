from .models import *

SEGMENTRON_CFG = dict(
    SOLVER = dict(
        AUX = False
    ),
    MODEL = dict(
        MODEL_NAME = '',
        BACKBONE = '',
        # model backbone channel scale
        BACKBONE_SCALE = 1.0,
        # support ['BN', 'SyncBN', 'FrozenBN', 'GN', 'nnSyncBN']
        BN_TYPE = 'BN',
        # batch norm epsilon for encoder, if set None will use api default value.
        BN_EPS_FOR_ENCODER = None,
        # BatchNorm momentum, if set None will use api default value.
        BN_MOMENTUM = None,
        # backbone output stride
        OUTPUT_STRIDE = 16,
        NUM_CLASS = 19,
        MULTI_DILATION = None,
        MULTI_GRID = False,
        # DeepLab config
        DEEPLABV3_PLUS = dict(
            USE_ASPP = True,
            ENABLE_DECODER = True,
            ASPP_WITH_SEP_CONV = True,
            DECODER_USE_SEP_CONV = True
        ),
        # OCNet config
        OCNet = dict(
            OC_ARCH = 'base'
        ),
        # EncNet config
        ENCNET = dict(
            SE_LOSS = True,
            SE_WEIGHT = 0.2,
            LATERAL = True
        ),
        # CGNET config
        CGNET = dict(
            STAGE2_BLOCK_NUM = 3,
            STAGE3_BLOCK_NUM = 21
        ),
        # PointRend config
        POINTREND = dict(
            BASEMODEL = 'DeepLabV3_Plus'
        ),
        # hrnet config
        HRNET = dict(
            PRETRAINED_LAYERS = ['*'],
            STEM_INPLANES = 64,
            FINAL_CONV_KERNEL = 1,
            WITH_HEAD = True,
            # stage 1
            STAGE1 = dict(
                NUM_MODULES = 1,
                NUM_BRANCHES = 1,
                NUM_BLOCKS = [1],
                NUM_CHANNELS = [32],
                BLOCK = 'BOTTLENECK',
                FUSE_METHOD = 'SUM',
            ),
            # stage 2
            STAGE2 = dict(
                NUM_MODULES = 1,
                NUM_BRANCHES = 2,
                NUM_BLOCKS = [4, 4],
                NUM_CHANNELS = [32, 64],
                BLOCK = 'BASIC',
                FUSE_METHOD = 'SUM'
            ),
            # stage 3
            STAGE3 = dict(
                NUM_MODULES = 1,
                NUM_BRANCHES = 3,
                NUM_BLOCKS = [4, 4, 4],
                NUM_CHANNELS = [32, 64, 128],
                BLOCK = 'BASIC',
                FUSE_METHOD = 'SUM'
            ),
            # stage 4
            STAGE4 = dict(
                NUM_MODULES = 1,
                NUM_BRANCHES = 4,
                NUM_BLOCKS = [4, 4, 4, 4],
                NUM_CHANNELS = [32, 64, 128, 256],
                BLOCK = 'BASIC',
                FUSE_METHOD = 'SUM'
            )
        )
    )
)


class ConfigDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict2obj(obj):
    if isinstance(obj, list):
        return [dict2obj(x) for x in obj]
    elif isinstance(obj, dict):
        _cfg = ConfigDict()
        for k in obj:
            _cfg[k] = dict2obj(obj[k])
        return _cfg
    else:
        return obj


def merge_dict1_into_dict2(dict1, dict2):
    dict2 = dict2.copy()
    for k, v in dict1.items():
        if not isinstance(v, dict):
            dict2[k] = v
        elif k not in dict2:
            dict2[k] = v
        elif not isinstance(dict2[k], dict):
            raise TypeError('You must set '
                            f'`{OVERWRITE_KEY}=True` to force an overwrite for %s.' %k)
        else:
            dict2[k] = merge_dict1_into_dict2(v, dict2[k])
    return dict2


def build_segmentron_config(config):
    if isinstance(config, dict):
        base_cfg = SEGMENTRON_CFG.copy()
        config = merge_dict1_into_dict2(config, base_cfg)
        config = dict2obj(config)

    return config


def fpenet(num_classes, **kwargs):
    _cfg = dict(
        MODEL = dict(
            MODEL_NAME = "FPENet"
        )
    )
    cfg = build_segmentron_config(_cfg)
    cfg.MODEL.NUM_CLASS = num_classes
    return FPENet(cfg, **kwargs)
