# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
from torch.optim import Optimizer

from .base import BaseTrainingEventInterface


class BaseTrainingEventMix:

    def launch(self):
        pass

    def create_optimizer(self, optimizer: Optimizer):
        pass

    def on_init_evaluate(self, result: dict):
        pass

    def on_evaluate(self, result: dict):
        pass

    def on_step_end(self, step: int, result: dict = None):
        pass


class BaseTrainingEventAdapter(BaseTrainingEventMix, BaseTrainingEventInterface):
    pass





