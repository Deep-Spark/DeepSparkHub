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

"""
Functional Cells used in Bert finetune and evaluation.
"""

import os
import math
import collections
import numpy as np
import mindspore.nn as nn
from mindspore import log as logger
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.callback import Callback
from mindspore.nn.metrics import Metric
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR


class CrossEntropyCalculation(nn.Cell):
    """
    Cross Entropy loss
    """
    def __init__(self, is_training=True):
        super(CrossEntropyCalculation, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.is_training = is_training

    def construct(self, logits, label_ids, num_labels):
        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx)
            one_hot_labels = self.onehot(label_ids, num_labels, self.on_value, self.off_value)
            per_example_loss = self.neg(self.reduce_sum(one_hot_labels * logits, self.last_idx))
            loss = self.reduce_mean(per_example_loss, self.last_idx)
            return_value = self.cast(loss, mstype.float32)
        else:
            return_value = logits * 1.0
        return return_value


def make_directory(path: str):
    """Make directory."""
    if path is None or not isinstance(path, str) or path.strip() == "":
        logger.error("The path(%r) is invalid type.", path)
        raise TypeError("Input path is invalid type")

    # convert the relative paths
    path = os.path.realpath(path)
    logger.debug("The abs path is %r", path)

    # check the path is exist and write permissions?
    if os.path.exists(path):
        real_path = path
    else:
        # All exceptions need to be caught because create directory maybe have some limit(permissions)
        logger.debug("The directory(%s) doesn't exist, will create it", path)
        try:
            os.makedirs(path, exist_ok=True)
            real_path = path
        except PermissionError as e:
            logger.error("No write permission on the directory(%r), error = %r", path, e)
            raise TypeError("No write permission on the directory.")
    return real_path

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, dataset_size=-1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            print("epoch: {}, current epoch percent: {}, step: {}, outputs are {}"
                  .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, str(cb_params.net_outputs)),
                  flush=True)
        else:
            print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)), flush=True)


def LoadNewestCkpt(load_finetune_checkpoint_dir, prefix):
    """
    Find the ckpt finetune generated and load it into eval network.
    """
    files = os.listdir(load_finetune_checkpoint_dir)
    max_time = 0
    for filename in files:
        if filename.startswith(prefix) and filename.endswith(".ckpt"):
            full_path = os.path.join(load_finetune_checkpoint_dir, filename)
            mtime = os.path.getmtime(full_path)
            if mtime > max_time:
                max_time = mtime
                load_finetune_checkpoint_path = full_path
    print("Find the newest checkpoint: ", load_finetune_checkpoint_path)
    return load_finetune_checkpoint_path


class BertLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def convert_labels_to_index(label_list):
    """
    Convert label_list to indices for NER task.
    """
    label2id = collections.OrderedDict()
    label2id["O"] = 0
    prefix = ["S_", "B_", "M_", "E_"]
    index = 0
    for label in label_list:
        for pre in prefix:
            index += 1
            sub_label = pre + label
            label2id[sub_label] = index
    return label2id

def _get_poly_lr(global_step, lr_init, lr_end, lr_max, warmup_steps, total_steps, poly_power):
    """
    generate learning rate array

    Args:
       global_step(int): current step
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_steps(int): number of warmup epochs
       total_steps(int): total epoch of training
       poly_power(int): poly learning rate power

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr = float(lr_max - lr_end) * (base ** poly_power)
            lr = lr + lr_end
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)

    learning_rate = np.array(lr_each_step).astype(np.float32)
    current_step = global_step
    learning_rate = learning_rate[current_step:]
    return learning_rate


def get_bert_thor_lr(lr_max=0.0034, lr_min=3.244e-05, lr_power=1.0, lr_total_steps=30000):
    learning_rate = _get_poly_lr(global_step=0, lr_init=0.0, lr_end=lr_min, lr_max=lr_max, warmup_steps=0,
                                 total_steps=lr_total_steps, poly_power=lr_power)
    return Tensor(learning_rate)


def get_bert_thor_damping(damping_max=5e-2, damping_min=1e-6, damping_power=1.0, damping_total_steps=30000):
    damping = _get_poly_lr(global_step=0, lr_init=0.0, lr_end=damping_min, lr_max=damping_max, warmup_steps=0,
                           total_steps=damping_total_steps, poly_power=damping_power)
    return Tensor(damping)


class EvalCallBack(Callback):
    """
    Evaluate after a certain amount of training samples.
    Args:
        model (Model): The network model.
        eval_ds (Dataset): The eval dataset.
        global_batch (int): The batchsize of the sum of all devices.
        eval_samples (int): The number of eval interval samples.
    """
    def __init__(self, model, eval_ds, global_batch, eval_samples):
        super(EvalCallBack, self).__init__()
        self.model = model
        self.eval_ds = eval_ds
        self.global_batch = global_batch
        self.eval_samples = eval_samples
        self.last_eval_step = 0

    def epoch_end(self, run_context):
        """
        Evaluate after training a certain number of samples.
        """
        cb_params = run_context.original_args()
        num_samples = (cb_params.cur_step_num - self.last_eval_step) * self.global_batch
        if num_samples < self.eval_samples:
            return
        self.last_eval_step = cb_params.cur_step_num
        total_sumples = cb_params.cur_step_num * self.global_batch
        res = self.model.eval(self.eval_ds, dataset_sink_mode=True)
        res = res['bert_acc']
        print("====================================", flush=True)
        print("Accuracy is: ", "%.6f" % res, ", current samples is: ", total_sumples)
        print("====================================", flush=True)

class BertMetric(Metric):
    """
    The metric of bert network.
    Args:
        batch_size (int): The batchsize of each device.
    """
    def __init__(self, batch_size):
        super(BertMetric, self).__init__()
        self.clear()
        self.batch_size = batch_size

    def clear(self):
        self.mlm_total = 0
        self.mlm_acc = 0

    def update(self, *inputs):
        mlm_acc = self._convert_data(inputs[0])
        mlm_total = self._convert_data(inputs[1])
        self.mlm_acc += mlm_acc
        self.mlm_total += mlm_total

    def eval(self):
        return self.mlm_acc / self.mlm_total
