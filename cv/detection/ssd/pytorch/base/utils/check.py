import os

import torch


def get_config_arg(config, name):
    if hasattr(config, name):
        value = getattr(config, name)
        if value is not None:
            return value

    if name in os.environ:
        return os.environ[name]

    return None


def check_config(config):
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        config.device, config.n_gpu, config.local_rank != -1, config.fp16))

    if config.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            config.gradient_accumulation_steps))

    # if config.fp16:
    #     assert config.opt_level == 2

    # nhwc can only be used with fp16
    if config.nhwc:
        assert config.fp16

        # input padding can only be used with NHWC
    if config.pad_input:
        assert config.nhwc

        # no dali can only be used with NCHW and no padding
    if not config.dali:
        assert (not config.nhwc)
        assert (not config.pad_input)
        assert (not config.use_nvjpeg)
        assert (not config.dali_cache)

    if config.dali_cache > 0:
        assert config.use_nvjpeg

    if config.jit:
        assert config.nhwc  # jit can not be applied with apex::syncbn used for non-nhwc


# Check that the run is valid for specified group BN arg
def validate_group_bn(bn_groups):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    # Can't have larger group than ranks
    assert(bn_groups <= world_size)

    # must have only complete groups
    assert(world_size % bn_groups == 0)










