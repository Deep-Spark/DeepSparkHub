# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


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





