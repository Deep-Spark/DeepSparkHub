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
"""export checkpoint file into air, onnx, mindir models"""
import numpy as np

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id
from src.inception_v3 import InceptionV3

import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context


config.batch_size = 1

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

def modelarts_process():
    pass

@moxing_wrapper(pre_process=modelarts_process)
def export_inceptionv3():
    """ export_inceptionv3 """
    net = InceptionV3(num_classes=config.num_classes, is_training=False)
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=[config.batch_size, 3, config.width, \
        config.height]), ms.float32)
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_inceptionv3()
