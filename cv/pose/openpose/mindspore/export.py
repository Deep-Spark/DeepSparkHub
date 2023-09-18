# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""export"""
import os
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.openposenet import OpenPoseNet
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)


def modelarts_pre_process():
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=None)
def model_export():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    # define net
    net = OpenPoseNet()

    # load checkpoint
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)
    inputs = np.ones([config.batch_size, 3, config.insize, config.insize]).astype(np.float32)
    export(net, Tensor(inputs), file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    model_export()
