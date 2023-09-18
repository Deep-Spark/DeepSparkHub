# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Run evaluation for a model exported to ONNX"""

import os
import numpy as np
import onnxruntime as ort
from mindspore import context

from src.dataset import create_ssd_dataset, create_mindrecord
from src.eval_utils import COCOMetrics
from src.model_utils.config import config


def create_session(checkpoint_path, target_device):
    """Create onnxruntime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def ssd_eval(dataset_path, ckpt_path, anno_json):
    """SSD evaluation."""
    # Silence false positive
    # pylint: disable=unexpected-keyword-arg
    ds = create_ssd_dataset(dataset_path, batch_size=config.batch_size,
                            is_training=False, use_multiprocessing=False)

    session, input_name = create_session(ckpt_path, config.device_target)
    total = ds.get_dataset_size() * config.batch_size
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")

    metrics = COCOMetrics(anno_json=anno_json,
                          classes=config.classes,
                          num_classes=config.num_classes,
                          max_boxes=config.max_boxes,
                          nms_threshold=config.nms_threshold,
                          min_score=config.min_score)

    for batch in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        img_id = batch['img_id']
        img_np = batch['image']
        image_shape = batch['image_shape']

        output = session.run(None, {input_name: batch['image']})

        for batch_idx in range(img_np.shape[0]):
            pred = {"boxes": output[0][batch_idx],
                    "box_scores": output[1][batch_idx],
                    "img_id": int(np.squeeze(img_id[batch_idx])),
                    "image_shape": image_shape[batch_idx]
                    }
            metrics.update(pred)
    print(f"mAP: {metrics.get_metrics()}")


def eval_net():
    """Eval ssd model"""
    if hasattr(config, 'num_ssd_boxes') and config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.steps[i]) * (w // config.steps[i]) * config.num_default[i]
        config.num_ssd_boxes = num

    if config.dataset == "coco":
        coco_root = os.path.join(config.data_path, config.coco_root)
        json_path = os.path.join(coco_root, config.instances_set.format(config.val_data_type))
    elif config.dataset == "voc":
        voc_root = os.path.join(config.data_path, config.voc_root)
        json_path = os.path.join(voc_root, config.voc_json)
    else:
        raise ValueError('SSD eval only support dataset mode is coco and voc!')

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

    mindrecord_file = create_mindrecord(config.dataset, "ssd_eval.mindrecord", False)

    print("Start Eval!")
    ssd_eval(mindrecord_file, config.file_name, json_path)


if __name__ == '__main__':
    eval_net()
