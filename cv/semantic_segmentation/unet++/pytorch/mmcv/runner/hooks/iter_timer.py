# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

import time
import torch.distributed as dist

from .hook import HOOKS, Hook


@HOOKS.register_module()
class IterTimerHook(Hook):

    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        self.batch_size = runner.cfg.data.samples_per_gpu
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        iter_info = {'time': time.time() - self.t}
        fps = self.batch_size / iter_info["time"]

        if dist.is_initialized():
            fps = fps * dist.get_world_size()

        iter_info["fps"] = fps

        runner.log_buffer.update(iter_info)
        self.t = time.time()
