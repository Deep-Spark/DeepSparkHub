# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""Train SSD and get checkpoint files."""

import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.common import set_seed, dtype
from src.ssd import SSD300, SsdInferWithDecoder, SSDWithLossCell, TrainingWrapper, ssd_mobilenet_v2,\
    ssd_mobilenet_v1_fpn, ssd_mobilenet_v1, ssd_resnet50_fpn, ssd_vgg16
from src.dataset import create_ssd_dataset, create_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter_by_list
from src.eval_callback import EvalCallBack
from src.eval_utils import apply_eval
from src.box_utils import default_boxes
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

set_seed(1)

def ssd_model_build():
    if config.model_name == "ssd300":
        backbone = ssd_mobilenet_v2()
        ssd = SSD300(backbone=backbone, config=config)
        init_net_param(ssd)
        if config.freeze_layer == "backbone":
            for param in backbone.feature_1.trainable_params():
                param.requires_grad = False
    elif config.model_name == "ssd_mobilenet_v1_fpn":
        ssd = ssd_mobilenet_v1_fpn(config=config)
        init_net_param(ssd)
        if config.feature_extractor_base_param != "":
            param_dict = ms.load_checkpoint(config.feature_extractor_base_param)
            for x in list(param_dict.keys()):
                param_dict["network.feature_extractor.mobilenet_v1." + x] = param_dict[x]
                del param_dict[x]
            ms.load_param_into_net(ssd.feature_extractor.mobilenet_v1.network, param_dict)
    elif config.model_name == "ssd_mobilenet_v1":
        ssd = ssd_mobilenet_v1(config=config)
        init_net_param(ssd)
        if config.feature_extractor_base_param != "":
            param_dict = ms.load_checkpoint(config.feature_extractor_base_param)
            for x in list(param_dict.keys()):
                param_dict["network.feature_extractor.mobilenet_v1." + x] = param_dict[x]
                del param_dict[x]
            ms.load_param_into_net(ssd.feature_extractor.mobilenet_v1.network, param_dict)
    elif config.model_name == "ssd_resnet50_fpn":
        ssd = ssd_resnet50_fpn(config=config)
        init_net_param(ssd)
        if config.feature_extractor_base_param != "":
            param_dict = ms.load_checkpoint(config.feature_extractor_base_param)
            for x in list(param_dict.keys()):
                param_dict["network.feature_extractor.resnet." + x] = param_dict[x]
                del param_dict[x]
            ms.load_param_into_net(ssd.feature_extractor.resnet, param_dict)
    elif config.model_name == "ssd_vgg16":
        ssd = ssd_vgg16(config=config)
        init_net_param(ssd)
        if config.feature_extractor_base_param != "":
            param_dict = ms.load_checkpoint(config.feature_extractor_base_param)
            from src.vgg16 import ssd_vgg_key_mapper
            for k in ssd_vgg_key_mapper:
                v = ssd_vgg_key_mapper[k]
                param_dict["network.backbone." + v + ".weight"] = param_dict[k + ".weight"]
                del param_dict[k + ".weight"]
            ms.load_param_into_net(ssd.backbone, param_dict)
    else:
        raise ValueError(f'config.model: {config.model_name} is not supported')
    return ssd

def set_graph_kernel_context(device_target, model):
    if device_target == "GPU" and model == "ssd300":
        # Enable graph kernel for default model ssd300 on GPU back-end.
        ms.set_context(enable_graph_kernel=True,
                       graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=Conv2D")
    if device_target == "GPU" and model == "ssd_mobilenet_v1":
        # Enable graph kernel for default model ssd300 on GPU back-end.
        ms.context.set_context(enable_graph_kernel=True,
                               graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=Conv2D")

@moxing_wrapper()
def train_net():
    if hasattr(config, 'num_ssd_boxes') and config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.steps[i]) * (w // config.steps[i]) * config.num_default[i]
        config.num_ssd_boxes = num

    rank = 0
    device_num = 1
    loss_scale = float(config.loss_scale)
    if config.device_target == "CPU":
        loss_scale = 1.0
        ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    else:
        ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)
        set_graph_kernel_context(config.device_target, config.model_name)
        if config.run_distribute:
            device_num = config.device_num
            ms.reset_auto_parallel_context()
            ms.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                         device_num=device_num)
            init()
            if config.all_reduce_fusion_config:
                ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            rank = get_rank()

    mindrecord_file = create_mindrecord(config.dataset, "ssd.mindrecord", True)

    if config.only_create_dataset:
        return

    # When create MindDataset, using the fitst mindrecord file, such as ssd.mindrecord0.
    use_multiprocessing = (config.device_target != "CPU")
    dataset = create_ssd_dataset(mindrecord_file, batch_size=config.batch_size,
                                 device_num=device_num, rank=rank, use_multiprocessing=use_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print(f"Create dataset done! dataset size is {dataset_size}")
    ssd = ssd_model_build()
    if (hasattr(config, 'use_float16') and config.use_float16):
        ssd.to_float(dtype.float16)
    net = SSDWithLossCell(ssd, config)

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs)
    ckpt_save_dir = config.output_path +'/ckpt_{}/'.format(rank)
    ckpoint_cb = ModelCheckpoint(prefix="ssd", directory=ckpt_save_dir, config=ckpt_config)

    if config.pre_trained:
        param_dict = ms.load_checkpoint(config.pre_trained)
        if config.filter_weight:
            filter_checkpoint_parameter_by_list(param_dict, config.checkpoint_filter_list)
        ms.load_param_into_net(net, param_dict, True)

    lr = Tensor(get_lr(global_step=config.pre_trained_epoch_size * dataset_size,
                       lr_init=config.lr_init, lr_end=config.lr_end_rate * config.lr, lr_max=config.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=dataset_size))

    if hasattr(config, 'use_global_norm') and config.use_global_norm:
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, 1.0)
        net = TrainingWrapper(net, opt, loss_scale, True)
    else:
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)

    callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]
    if config.run_eval:
        eval_net = SsdInferWithDecoder(ssd, Tensor(default_boxes), config)
        eval_net.set_train(False)
        mindrecord_file = create_mindrecord(config.dataset, "ssd_eval.mindrecord", False)
        eval_dataset = create_ssd_dataset(mindrecord_file, batch_size=config.batch_size,
                                          is_training=False, use_multiprocessing=False)
        if config.dataset == "coco":
            anno_json = os.path.join(config.coco_root, config.instances_set.format(config.val_data_type))
        elif config.dataset == "voc":
            anno_json = os.path.join(config.voc_root, config.voc_json)
        else:
            raise ValueError('SSD eval only support dataset mode is coco and voc!')
        eval_param_dict = {"net": eval_net, "dataset": eval_dataset, "anno_json": anno_json}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_map.ckpt",
                               metrics_name="mAP")
        callback.append(eval_cb)
    model = Model(net)
    dataset_sink_mode = False
    if config.mode_sink == "sink" and config.device_target != "CPU":
        print("In sink mode, one epoch return a loss.")
        dataset_sink_mode = True
    print("Start train SSD, the first epoch will be slower because of the graph compilation.")
    model.train(config.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)

if __name__ == '__main__':
    train_net()
